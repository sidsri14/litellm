"""
Unit Tests for the per-session budget limiter for the proxy.

Tests that session-scoped budget tracking works correctly:
- Enforces max_budget_per_session per session_id (read from agent litellm_params)
- Different sessions have independent budgets
- Requests under budget pass through
- Requests without agent_id pass through
"""

import logging
from typing import Final
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import _redis_circuit_breaker_guard
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.max_budget_per_session_limiter import (
    PROXY_MaxBudgetPerSessionHandler,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse


def _make_mock_agent(max_budget_per_session: float) -> AgentResponse:
    return AgentResponse(
        agent_id="agent-budget-123",
        agent_name="budget-agent",
        litellm_params={"max_budget_per_session": max_budget_per_session},
        agent_card_params={"name": "budget-agent", "version": "1.0.0"},
    )


@pytest.mark.asyncio
async def test_budget_per_session_under_budget_passes():
    """
    Requests under budget should pass through without error.
    """
    local_cache = DualCache()
    handler = PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    mock_agent = _make_mock_agent(max_budget_per_session=5.0)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"metadata": {"session_id": "session-budget-1"}},
            call_type="",
        )
        assert result is None


@pytest.mark.asyncio
async def test_budget_per_session_exceeds_budget():
    """
    After accumulating spend beyond max_budget_per_session, the next
    pre-call check should raise 429.
    """
    local_cache = DualCache()
    handler = PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    session_id = "session-over-budget"
    cache_key = handler._make_cache_key(session_id)
    await handler._increment_spend(cache_key, 1.50)

    mock_agent = _make_mock_agent(max_budget_per_session=1.0)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": session_id}},
                call_type="",
            )
        assert exc_info.value.status_code == 429
        assert "budget exceeded" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_budget_per_session_independent_sessions():
    """
    Different session_ids have independent budget counters.
    Exhausting session A does not affect session B.
    """
    local_cache = DualCache()
    handler = PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    cache_key_a = handler._make_cache_key("session-A")
    await handler._increment_spend(cache_key_a, 3.0)

    mock_agent = _make_mock_agent(max_budget_per_session=2.0)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

        # Session A should be blocked
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": "session-A"}},
                call_type="",
            )
        assert exc_info.value.status_code == 429

        # Session B should still pass
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"metadata": {"session_id": "session-B"}},
            call_type="",
        )
        assert result is None


@pytest.mark.asyncio
async def test_no_agent_id_passes():
    """
    When no agent_id is set on the key, all requests pass through.
    """
    local_cache = DualCache()
    handler = PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-no-agent",
    )

    result = await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"metadata": {"session_id": "any-session"}},
        call_type="",
    )
    assert result is None


class _OpenBreakerRedis:
    def __init__(self) -> None:
        from litellm.caching.redis_cache import RedisCircuitBreaker

        self._circuit_breaker = RedisCircuitBreaker(failure_threshold=3, recovery_timeout=60)
        for _ in range(3):
            self._circuit_breaker.record_failure()

    @_redis_circuit_breaker_guard
    async def async_get_cache(self, key, **kwargs):
        raise AssertionError("never reached")

    def async_register_script(self, script):
        @_redis_circuit_breaker_guard
        async def refused(_self, keys, args):
            raise AssertionError("never reached")

        return lambda keys, args: refused(self, keys, args)


@pytest.mark.asyncio
async def test_an_open_circuit_breaker_reads_session_spend_locally_without_a_warning(caplog):
    cache = DualCache(redis_cache=_OpenBreakerRedis())  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
    handler = PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(cache))
    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
        spend = await handler._get_current_spend("{session_budget:quiet}:spend")

    assert spend == 0.0
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []
    assert any("circuit breaker is open" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_concurrent_burst_enforces_fail_closed_budget_reservation():
    """
    Issue #34732: Concurrent requests must not bypass max_budget_per_session.
    Under a 0.005 budget with 0.001 reservation per call, exactly 5 of 20
    concurrent requests must be admitted, and 15 must be rejected with 429.
    """
    import asyncio

    local_cache: Final = DualCache()
    handler: Final = PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict: Final = UserAPIKeyAuth(
        api_key="sk-test-burst-key",
        agent_id="agent-budget-burst",
    )

    budget: Final[float] = 0.005
    reserve_per_call: Final[float] = 0.001
    mock_agent: Final = _make_mock_agent(max_budget_per_session=budget)
    session_id: Final = "concurrent-burst-session"

    admitted: list[dict] = []
    rejected: list[Exception] = []

    async def attempt_call(index: int) -> None:
        data: Final[dict] = {
            "metadata": {
                "session_id": session_id,
                "session_reservation_amount": reserve_per_call,
            }
        }
        try:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data=data,
                call_type="",
            )
            admitted.append(data)
        except Exception as exc:
            rejected.append(exc)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent
        await asyncio.gather(*(attempt_call(i) for i in range(20)))

    assert len(admitted) == 5, f"Expected exactly 5 admitted, got {len(admitted)}"
    assert len(rejected) == 15, f"Expected 15 rejected, got {len(rejected)}"
    for err in rejected:
        assert isinstance(err, HTTPException)
        assert err.status_code == 429
        assert "budget exceeded" in str(err.detail).lower()

    # Reconcile all admitted calls
    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent
        for req_data in admitted:
            await handler.async_log_success_event(
                {
                    "call_type": "acompletion",
                    "response_cost": reserve_per_call,
                    "litellm_params": {"metadata": req_data["metadata"]},
                },
                None,
                None,
                None,
            )

    spend_key: Final = handler._make_cache_key(session_id)
    reserve_key: Final = handler._make_reservation_cache_key(session_id)
    final_spend: Final = await handler._get_current_spend(spend_key)
    final_reserved: Final = await handler._get_current_reserved(reserve_key)

    assert final_spend == pytest.approx(budget)
    assert final_reserved == 0.0


@pytest.mark.asyncio
async def test_failed_call_releases_reservation():
    """
    When a call fails, async_log_failure_event must release the reservation
    so subsequent calls can use the remaining capacity.
    """
    local_cache: Final = DualCache()
    handler: Final = PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict: Final = UserAPIKeyAuth(
        api_key="sk-test-fail-key",
        agent_id="agent-budget-fail",
    )

    budget: Final[float] = 0.001
    mock_agent: Final = _make_mock_agent(max_budget_per_session=budget)
    session_id: Final = "fail-release-session"

    data_1: Final[dict] = {
        "metadata": {
            "session_id": session_id,
            "session_reservation_amount": 0.001,
            "agent_id": "agent-budget-fail",
        }
    }

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

        # Call 1 reserves full budget
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data=data_1,
            call_type="",
        )

        # Call 2 should be rejected as capacity is reserved
        data_2: Final[dict] = {
            "metadata": {
                "session_id": session_id,
                "session_reservation_amount": 0.001,
            }
        }
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data=data_2,
                call_type="",
            )
        assert exc_info.value.status_code == 429

        # Call 1 fails and releases reservation
        await handler.async_log_failure_event(
            {
                "call_type": "acompletion",
                "litellm_params": {"metadata": data_1["metadata"]},
            },
            None,
            None,
            None,
        )

        # Call 3 should now succeed
        data_3: Final[dict] = {
            "metadata": {
                "session_id": session_id,
                "session_reservation_amount": 0.001,
            }
        }
        res = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data=data_3,
            call_type="",
        )
        assert res is None

