"""
Per-Session Budget Limiter for LiteLLM Proxy.

Enforces a dollar-amount cap per session (identified by `session_id` /
`x-litellm-trace-id`). After each successful LLM call the response cost is
accumulated against the session. When the accumulated spend exceeds
`max_budget_per_session` (configured in agent litellm_params), subsequent
requests for that session receive a 429.

Includes atomic session budget reservations to prevent concurrent request
bursts from bypassing the session budget floor (Issue #34732).

Note: trace-id enforcement (require_trace_id_on_calls_by_agent) is handled
separately in auth_checks.py at the agent level, not in this hook.

Works across multiple proxy instances via DualCache (in-memory + Redis).
Follows the same pattern as max_iterations_limiter.py and budget_reservation.py.
"""

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Final

from pydantic import ConfigDict, TypeAdapter

from litellm import DualCache
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import log_redis_failure
from litellm.exceptions import RateLimitType
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit

if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    InternalUsageCache = _InternalUsageCache
else:
    InternalUsageCache = Any


_CACHE_AMOUNT_ADAPTER: Final[TypeAdapter[str | bytes | int | float]] = TypeAdapter(
    str | bytes | int | float, config=ConfigDict(strict=True, hide_input_in_errors=True)
)
_OPTIONAL_CACHE_AMOUNT_ADAPTER: Final[TypeAdapter[str | bytes | int | float | None]] = TypeAdapter(
    str | bytes | int | float | None, config=ConfigDict(strict=True, hide_input_in_errors=True)
)


# Redis Lua script for atomic float increment with TTL.
# INCRBYFLOAT returns the new value as a string.
# Only sets EXPIRE on first call (when prior value was nil).
MAX_BUDGET_SESSION_INCREMENT_SCRIPT: Final = """
local key = KEYS[1]
local amount = ARGV[1]
local ttl = tonumber(ARGV[2])

local existed = redis.call('EXISTS', key)
local new_val = redis.call('INCRBYFLOAT', key, amount)
if existed == 0 then
    redis.call('EXPIRE', key, ttl)
end

return new_val
"""

# Redis Lua script for atomic session budget reservation.
# KEYS[1]: spend key ({session_budget:<session_id>}:spend)
# KEYS[2]: reservation key ({session_budget:<session_id>}:reserved)
# ARGV[1]: reserve_amount (float string)
# ARGV[2]: max_budget (float string)
# ARGV[3]: ttl (seconds)
MAX_BUDGET_SESSION_RESERVE_SCRIPT: Final = """
local spend_key = KEYS[1]
local reserve_key = KEYS[2]
local amount = tonumber(ARGV[1])
local max_budget = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

local spend = tonumber(redis.call('GET', spend_key) or 0)
local reserved = tonumber(redis.call('GET', reserve_key) or 0)
local total = spend + reserved

if (total + amount) > max_budget then
    return -1
end

local new_reserved = redis.call('INCRBYFLOAT', reserve_key, amount)
redis.call('EXPIRE', reserve_key, ttl)

return tonumber(new_reserved)
"""

# Redis Lua script for reconciling reservation and recording actual spend.
# KEYS[1]: spend key
# KEYS[2]: reservation key
# ARGV[1]: reserve_amount (to decrement)
# ARGV[2]: actual_cost (to increment)
# ARGV[3]: ttl (seconds)
MAX_BUDGET_SESSION_RECONCILE_SCRIPT: Final = """
local spend_key = KEYS[1]
local reserve_key = KEYS[2]
local reserve_amount = tonumber(ARGV[1])
local actual_cost = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

if reserve_amount > 0 then
    local reserved = tonumber(redis.call('GET', reserve_key) or 0)
    local new_reserved = reserved - reserve_amount
    if new_reserved <= 0.000001 then
        redis.call('DEL', reserve_key)
    else
        redis.call('SET', reserve_key, tostring(new_reserved))
        redis.call('EXPIRE', reserve_key, ttl)
    end
end

if actual_cost > 0 then
    local existed = redis.call('EXISTS', spend_key)
    local new_spend = redis.call('INCRBYFLOAT', spend_key, actual_cost)
    if existed == 0 then
        redis.call('EXPIRE', spend_key, ttl)
    end
    return new_spend
else
    local spend = tonumber(redis.call('GET', spend_key) or 0)
    return spend
end
"""

# Redis Lua script for releasing reservation on call failure.
# KEYS[1]: reservation key
# ARGV[1]: reserve_amount (to decrement)
# ARGV[2]: ttl (seconds)
MAX_BUDGET_SESSION_RELEASE_SCRIPT: Final = """
local reserve_key = KEYS[1]
local reserve_amount = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])

if reserve_amount > 0 then
    local reserved = tonumber(redis.call('GET', reserve_key) or 0)
    local new_reserved = reserved - reserve_amount
    if new_reserved <= 0.000001 then
        redis.call('DEL', reserve_key)
    else
        redis.call('SET', reserve_key, tostring(new_reserved))
        redis.call('EXPIRE', reserve_key, ttl)
    end
end
return 1
"""

# Default TTL for session budget counters (1 hour)
DEFAULT_MAX_BUDGET_PER_SESSION_TTL: Final = 3600


class _PROXY_MaxBudgetPerSessionHandler(CustomLogger):
    """
    Pre-call hook that enforces max_budget_per_session.

    Configuration (set in agent litellm_params):
        - max_budget_per_session: dollar cap per session_id
        - session_reservation_amount: optional estimated cost reservation per call

    Cache key patterns:
        {session_budget:<session_id>}:spend
        {session_budget:<session_id>}:reserved
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        self.ttl: Final[int] = int(
            os.getenv(
                "LITELLM_MAX_BUDGET_PER_SESSION_TTL",
                DEFAULT_MAX_BUDGET_PER_SESSION_TTL,
            )
        )
        self._in_memory_lock: asyncio.Lock | None = None

        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            self.increment_script: Final = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                MAX_BUDGET_SESSION_INCREMENT_SCRIPT
            )
            self.reserve_script: Final = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                MAX_BUDGET_SESSION_RESERVE_SCRIPT
            )
            self.reconcile_script: Final = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                MAX_BUDGET_SESSION_RECONCILE_SCRIPT
            )
            self.release_script: Final = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                MAX_BUDGET_SESSION_RELEASE_SCRIPT
            )
        else:
            self.increment_script = None
            self.reserve_script = None
            self.reconcile_script = None
            self.release_script = None

    @property
    def in_memory_lock(self) -> asyncio.Lock:
        if self._in_memory_lock is None:
            self._in_memory_lock = asyncio.Lock()
        return self._in_memory_lock

    @with_service_target("session_budgets")
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Exception | str | dict | None:
        """
        Before each LLM call, check if max_budget_per_session is set and
        whether accumulated spend + in-flight reservations exceed the budget (429 if so).
        Atomically leases a reservation to guarantee fail-closed budget caps under concurrency.
        """
        raw_max_budget: Final = self._get_max_budget_per_session(user_api_key_dict)
        session_id: Final = self._get_session_id(data)

        if raw_max_budget is None or session_id is None:
            return None

        max_budget: Final[float] = float(raw_max_budget)
        spend_key: Final = self._make_cache_key(session_id)
        reserve_key: Final = self._make_reservation_cache_key(session_id)

        current_spend: Final = await self._get_current_spend(spend_key)

        verbose_proxy_logger.debug(
            "MaxBudgetPerSessionHandler: session_id=%s, spend=%.4f, max=%.2f",
            session_id,
            current_spend,
            max_budget,
        )

        if current_spend >= max_budget:
            resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(data.get("model") if data else None)
            raise ProxyRateLimitError(
                detail=(
                    f"Session budget exceeded for session {session_id}. "
                    f"Current spend: ${current_spend:.4f}, "
                    f"max_budget_per_session: ${max_budget:.2f}."
                ),
                rate_limit_type=RateLimitType.BUDGET,
                model=resolved_model,
                llm_provider=llm_provider,
            )

        reserve_amount: Final = self._get_reservation_amount(data, user_api_key_dict, max_budget)

        if reserve_amount > 0:
            allowed: Final = await self._reserve_spend(
                spend_key=spend_key,
                reserve_key=reserve_key,
                reserve_amount=reserve_amount,
                max_budget=max_budget,
            )
            if not allowed:
                current_reserved: Final = await self._get_current_reserved(reserve_key)
                total_spend: Final = current_spend + current_reserved
                resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(data.get("model") if data else None)
                raise ProxyRateLimitError(
                    detail=(
                        f"Session budget exceeded for session {session_id}. "
                        f"Current spend: ${total_spend:.4f}, "
                        f"max_budget_per_session: ${max_budget:.2f}."
                    ),
                    rate_limit_type=RateLimitType.BUDGET,
                    model=resolved_model,
                    llm_provider=llm_provider,
                )

            # Attach reservation context to request metadata for post-call reconciliation
            if "metadata" not in data or not isinstance(data["metadata"], dict):
                data["metadata"] = {}
            data["metadata"]["_session_budget_reserved"] = reserve_amount
            data["metadata"]["_session_budget_session_id"] = session_id
            data["metadata"]["_session_budget_released"] = False
            if user_api_key_dict.agent_id is not None and "agent_id" not in data["metadata"]:
                data["metadata"]["agent_id"] = user_api_key_dict.agent_id

            if "litellm_metadata" not in data or not isinstance(data["litellm_metadata"], dict):
                data["litellm_metadata"] = {}
            data["litellm_metadata"]["_session_budget_reserved"] = reserve_amount
            data["litellm_metadata"]["_session_budget_session_id"] = session_id
            data["litellm_metadata"]["_session_budget_released"] = False
            if user_api_key_dict.agent_id is not None and "agent_id" not in data["litellm_metadata"]:
                data["litellm_metadata"]["agent_id"] = user_api_key_dict.agent_id

        return None

    @with_service_target("session_budgets")
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """
        After a successful LLM call, release reservation and increment session spend by response cost.
        """
        try:
            litellm_params: Final = kwargs.get("litellm_params") or {}
            metadata: Final = litellm_params.get("metadata") or {}
            session_id: Final = metadata.get("session_id") or metadata.get("_session_budget_session_id")
            if session_id is None:
                return

            agent_id: Final = metadata.get("agent_id")
            if agent_id is None:
                return

            from litellm.proxy.agent_endpoints.agent_registry import (
                global_agent_registry,
            )

            agent: Final = global_agent_registry.get_agent_by_id(agent_id=str(agent_id))
            if agent is None:
                return

            agent_litellm_params: Final = agent.litellm_params or {}
            max_budget: Final = agent_litellm_params.get("max_budget_per_session")
            if max_budget is None:
                return

            reserve_amount: Final[float] = float(metadata.get("_session_budget_reserved") or 0.0)
            metadata["_session_budget_released"] = True

            response_cost: Final[float] = float(kwargs.get("response_cost") or 0.0)

            spend_key: Final = self._make_cache_key(str(session_id))
            reserve_key: Final = self._make_reservation_cache_key(str(session_id))

            if reserve_amount > 0 or response_cost > 0:
                await self._reconcile_spend(
                    spend_key=spend_key,
                    reserve_key=reserve_key,
                    reserve_amount=reserve_amount,
                    actual_cost=response_cost,
                )

            verbose_proxy_logger.debug(
                "MaxBudgetPerSessionHandler: reconciled session %s (reserved=%.6f, cost=%.6f)",
                session_id,
                reserve_amount,
                response_cost,
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                "MaxBudgetPerSessionHandler: error in async_log_success_event: %s",
                str(e),
            )

    @with_service_target("session_budgets")
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """
        On call failure, release any reserved budget so capacity is not stranded.
        """
        try:
            litellm_params: Final = kwargs.get("litellm_params") or {}
            metadata: Final = litellm_params.get("metadata") or {}
            if metadata.get("_session_budget_released") is True:
                return

            session_id: Final = metadata.get("session_id") or metadata.get("_session_budget_session_id")
            if session_id is None:
                return

            reserve_amount: Final[float] = float(metadata.get("_session_budget_reserved") or 0.0)
            if reserve_amount <= 0:
                return

            metadata["_session_budget_released"] = True
            reserve_key: Final = self._make_reservation_cache_key(str(session_id))
            await self._release_reservation(reserve_key, reserve_amount)

            verbose_proxy_logger.debug(
                "MaxBudgetPerSessionHandler: released reservation %.6f for session %s on failure",
                reserve_amount,
                session_id,
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                "MaxBudgetPerSessionHandler: error in async_log_failure_event: %s",
                str(e),
            )

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        error_type: Exception,
    ) -> None:
        """Release reservation if proxy rejected downstream after pre_call_hook."""
        try:
            metadata: Final = request_data.get("metadata") or {}
            litellm_metadata: Final = request_data.get("litellm_metadata") or {}
            if metadata.get("_session_budget_released") is True or litellm_metadata.get("_session_budget_released") is True:
                return

            session_id: Final = self._get_session_id(request_data)
            if session_id is None:
                return

            reserve_amount: Final[float] = float(
                metadata.get("_session_budget_reserved")
                or litellm_metadata.get("_session_budget_reserved")
                or 0.0
            )
            if reserve_amount <= 0:
                return

            metadata["_session_budget_released"] = True
            litellm_metadata["_session_budget_released"] = True
            reserve_key: Final = self._make_reservation_cache_key(str(session_id))
            await self._release_reservation(reserve_key, reserve_amount)
        except Exception as e:
            verbose_proxy_logger.warning(
                "MaxBudgetPerSessionHandler: error in async_post_call_failure_hook: %s",
                str(e),
            )

    def _get_reservation_amount(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        max_budget: float,
    ) -> float:
        """Determine reservation amount for in-flight concurrency protection."""
        metadata: Final = data.get("metadata") or {}
        litellm_metadata: Final = data.get("litellm_metadata") or {}
        custom_amount: Final = metadata.get("session_reservation_amount") or litellm_metadata.get("session_reservation_amount")
        if custom_amount is not None:
            return min(float(custom_amount), max_budget)

        if user_api_key_dict.agent_id is not None:
            from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

            agent: Final = global_agent_registry.get_agent_by_id(agent_id=user_api_key_dict.agent_id)
            if agent and agent.litellm_params:
                agent_res: Final = agent.litellm_params.get("session_reservation_amount")
                if agent_res is not None:
                    return min(float(agent_res), max_budget)

        env_amount: Final = os.getenv("LITELLM_SESSION_BUDGET_RESERVATION_AMOUNT")
        if env_amount is not None:
            return min(float(env_amount), max_budget)

        return min(0.0001, max_budget)

    def _get_session_id(self, data: dict) -> str | None:
        """Extract session_id from request metadata."""
        metadata: Final = data.get("metadata") or {}
        session_id = metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        litellm_metadata: Final = data.get("litellm_metadata") or {}
        session_id = litellm_metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        return None

    def _get_max_budget_per_session(self, user_api_key_dict: UserAPIKeyAuth) -> float | None:
        """Extract max_budget_per_session from agent litellm_params."""
        agent_id: Final = user_api_key_dict.agent_id
        if agent_id is None:
            return None

        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        agent: Final = global_agent_registry.get_agent_by_id(agent_id=agent_id)
        if agent is None:
            return None

        litellm_params: Final = agent.litellm_params or {}
        max_budget: Final = litellm_params.get("max_budget_per_session")
        if max_budget is not None:
            return float(max_budget)
        return None

    def _make_cache_key(self, session_id: str) -> str:
        return f"{{session_budget:{session_id}}}:spend"

    def _make_reservation_cache_key(self, session_id: str) -> str:
        return f"{{session_budget:{session_id}}}:reserved"

    @with_service_target("session_budgets")
    async def _get_current_spend(self, cache_key: str) -> float:
        """Read current accumulated spend for a session."""
        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            try:
                redis_result: Final[str | bytes | int | float | None] = _OPTIONAL_CACHE_AMOUNT_ADAPTER.validate_python(
                    await self.internal_usage_cache.dual_cache.redis_cache.async_get_cache(key=cache_key)
                )
                if redis_result is not None:
                    return float(redis_result)
                return 0.0
            except Exception as e:
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    "MaxBudgetPerSessionHandler: Redis GET failed, falling back to in-memory",
                    e,
                )

        cache_result: Final[str | bytes | int | float | None] = _OPTIONAL_CACHE_AMOUNT_ADAPTER.validate_python(
            await self.internal_usage_cache.async_get_cache(
                key=cache_key,
                litellm_parent_otel_span=None,
                local_only=True,
            )
        )
        if cache_result is not None:
            return float(cache_result)
        return 0.0

    @with_service_target("session_budgets")
    async def _get_current_reserved(self, cache_key: str) -> float:
        """Read current active reservations for a session."""
        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            try:
                redis_result: Final[str | bytes | int | float | None] = _OPTIONAL_CACHE_AMOUNT_ADAPTER.validate_python(
                    await self.internal_usage_cache.dual_cache.redis_cache.async_get_cache(key=cache_key)
                )
                if redis_result is not None:
                    return float(redis_result)
                return 0.0
            except Exception as e:
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    "MaxBudgetPerSessionHandler: Redis GET reserved failed, falling back to in-memory",
                    e,
                )

        cache_result: Final[str | bytes | int | float | None] = _OPTIONAL_CACHE_AMOUNT_ADAPTER.validate_python(
            await self.internal_usage_cache.async_get_cache(
                key=cache_key,
                litellm_parent_otel_span=None,
                local_only=True,
            )
        )
        if cache_result is not None:
            return float(cache_result)
        return 0.0

    async def _reserve_spend(
        self,
        spend_key: str,
        reserve_key: str,
        reserve_amount: float,
        max_budget: float,
    ) -> bool:
        """Atomically reserve capacity against session budget."""
        if self.reserve_script is not None:
            try:
                script_result: Final[str | bytes | int | float] = _CACHE_AMOUNT_ADAPTER.validate_python(
                    await self.reserve_script(
                        keys=[spend_key, reserve_key],
                        args=[str(reserve_amount), str(max_budget), self.ttl],
                    )
                )
                return float(script_result) >= 0
            except Exception as e:
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    "MaxBudgetPerSessionHandler: Redis reserve script failed, falling back to in-memory",
                    e,
                )

        return await self._in_memory_reserve_spend(spend_key, reserve_key, reserve_amount, max_budget)

    async def _in_memory_reserve_spend(
        self,
        spend_key: str,
        reserve_key: str,
        reserve_amount: float,
        max_budget: float,
    ) -> bool:
        async with self.in_memory_lock:
            current_spend: Final[float] = await self._get_current_spend(spend_key)
            current_reserved: Final[float] = await self._get_current_reserved(reserve_key)
            if (current_spend + current_reserved + reserve_amount) > max_budget:
                return False

            new_reserved: Final[float] = current_reserved + reserve_amount
            await self.internal_usage_cache.async_set_cache(
                key=reserve_key,
                value=new_reserved,
                ttl=self.ttl,
                litellm_parent_otel_span=None,
                local_only=True,
            )
            return True

    async def _release_reservation(self, reserve_key: str, reserve_amount: float) -> None:
        """Release reservation capacity on call completion or failure."""
        if reserve_amount <= 0:
            return

        if self.release_script is not None:
            try:
                await self.release_script(
                    keys=[reserve_key],
                    args=[str(reserve_amount), self.ttl],
                )
                return
            except Exception as e:
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    "MaxBudgetPerSessionHandler: Redis release script failed, falling back to in-memory",
                    e,
                )

        await self._in_memory_release_reservation(reserve_key, reserve_amount)

    async def _in_memory_release_reservation(self, reserve_key: str, reserve_amount: float) -> None:
        async with self.in_memory_lock:
            current_reserved: Final[float] = await self._get_current_reserved(reserve_key)
            new_reserved: Final[float] = max(0.0, current_reserved - reserve_amount)
            await self.internal_usage_cache.async_set_cache(
                key=reserve_key,
                value=new_reserved,
                ttl=self.ttl,
                litellm_parent_otel_span=None,
                local_only=True,
            )

    async def _reconcile_spend(
        self,
        spend_key: str,
        reserve_key: str,
        reserve_amount: float,
        actual_cost: float,
    ) -> float:
        """Atomically decrement reservation and increment actual spend."""
        if self.reconcile_script is not None:
            try:
                script_result: Final[str | bytes | int | float] = _CACHE_AMOUNT_ADAPTER.validate_python(
                    await self.reconcile_script(
                        keys=[spend_key, reserve_key],
                        args=[str(reserve_amount), str(actual_cost), self.ttl],
                    )
                )
                return float(script_result)
            except Exception as e:
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    "MaxBudgetPerSessionHandler: Redis reconcile script failed, falling back to in-memory",
                    e,
                )

        return await self._in_memory_reconcile_spend(spend_key, reserve_key, reserve_amount, actual_cost)

    async def _in_memory_reconcile_spend(
        self,
        spend_key: str,
        reserve_key: str,
        reserve_amount: float,
        actual_cost: float,
    ) -> float:
        async with self.in_memory_lock:
            if reserve_amount > 0:
                current_reserved: Final[float] = await self._get_current_reserved(reserve_key)
                new_reserved: Final[float] = max(0.0, current_reserved - reserve_amount)
                await self.internal_usage_cache.async_set_cache(
                    key=reserve_key,
                    value=new_reserved,
                    ttl=self.ttl,
                    litellm_parent_otel_span=None,
                    local_only=True,
                )

            if actual_cost > 0:
                return await self._in_memory_increment_spend(spend_key, actual_cost)
            return await self._get_current_spend(spend_key)

    async def _increment_spend(self, cache_key: str, amount: float) -> float:
        """Atomically increment the session spend and return the new value."""
        if self.increment_script is not None:
            try:
                script_result: Final[str | bytes | int | float] = _CACHE_AMOUNT_ADAPTER.validate_python(
                    await self.increment_script(
                        keys=[cache_key],
                        args=[str(amount), self.ttl],
                    )
                )
                return float(script_result)
            except Exception as e:
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    "MaxBudgetPerSessionHandler: Redis INCRBYFLOAT failed, falling back to in-memory",
                    e,
                )

        return await self._in_memory_increment_spend(cache_key, amount)

    async def _in_memory_increment_spend(self, cache_key: str, amount: float) -> float:
        current: Final[str | bytes | int | float | None] = _OPTIONAL_CACHE_AMOUNT_ADAPTER.validate_python(
            await self.internal_usage_cache.async_get_cache(
                key=cache_key,
                litellm_parent_otel_span=None,
                local_only=True,
            )
        )
        current_spend: Final = float(current) if current is not None else 0.0
        new_value: Final = current_spend + amount
        await self.internal_usage_cache.async_set_cache(
            key=cache_key,
            value=new_value,
            ttl=self.ttl,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        return new_value


PROXY_MaxBudgetPerSessionHandler: Final = _PROXY_MaxBudgetPerSessionHandler
