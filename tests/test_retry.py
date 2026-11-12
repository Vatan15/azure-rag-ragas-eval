"""Tests for retry with exponential backoff (src/retry.py). Offline: time is faked."""
from types import SimpleNamespace

import pytest
from azure.core.exceptions import HttpResponseError, ServiceResponseError

from src import retry
from src.config import MAX_RETRIES
from src.stats import stats


@pytest.fixture
def delays(monkeypatch):
    """Replace sleeping with recording, and jitter with zero, so tests are instant and exact."""
    recorded = []
    monkeypatch.setattr(retry.time, "sleep", recorded.append)
    monkeypatch.setattr(retry.random, "uniform", lambda a, b: 0.0)
    return recorded


def flaky(failures, error):
    """A function that raises `error` the first `failures` times, then returns 'ok'."""
    state = {"calls": 0}

    def func():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise error
        return "ok"

    return func, state


def test_succeeds_after_two_failures_with_backoff_1_then_2_seconds(delays):
    func, state = flaky(2, ServiceResponseError("Connection aborted"))
    before = stats["retries"]
    assert retry.with_retry(func) == "ok"
    assert state["calls"] == 3
    assert delays == [1.0, 2.0]
    assert stats["retries"] - before == 2


def test_gives_up_after_max_retries(delays):
    func, state = flaky(99, ServiceResponseError("Connection aborted"))
    with pytest.raises(ServiceResponseError):
        retry.with_retry(func)
    assert state["calls"] == MAX_RETRIES + 1  # first try + 3 retries
    assert delays == [1.0, 2.0, 4.0]


def test_non_retryable_error_is_raised_immediately(delays):
    func, state = flaky(99, ValueError("bad input"))
    with pytest.raises(ValueError):
        retry.with_retry(func)
    assert state["calls"] == 1
    assert delays == []


def test_retry_after_header_is_honored(delays):
    err = ServiceResponseError("slow down")
    err.response = SimpleNamespace(headers={"retry-after": "7"})
    func, _ = flaky(1, err)
    assert retry.with_retry(func) == "ok"
    assert delays == [7.0]  # max(1.0 backoff, 7.0 requested by the server)


def test_http_429_is_retryable_but_500_is_not():
    too_many = HttpResponseError("rate limited")
    too_many.status_code = 429
    server_error = HttpResponseError("boom")
    server_error.status_code = 500
    assert retry.is_retryable(too_many) is True
    assert retry.is_retryable(server_error) is False
