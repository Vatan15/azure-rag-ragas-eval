"""Retry with exponential backoff and jitter for rate-limit and connection errors."""
import random
import time

from azure.core.exceptions import (
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)
from openai import APIConnectionError, RateLimitError

from .config import BASE_DELAY, MAX_RETRIES
from .stats import stats

# Errors worth retrying: rate limits (429) and dropped connections.
RETRYABLE = (RateLimitError, APIConnectionError, ServiceRequestError, ServiceResponseError)


def is_retryable(err):
    # Azure Search reports a 429 as a generic HttpResponseError, so check its status code.
    if isinstance(err, HttpResponseError):
        return err.status_code == 429
    return isinstance(err, RETRYABLE)


def get_retry_after(err):
    """Seconds the server asked us to wait (Retry-After header), or None."""
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def with_retry(func, *args, **kwargs):
    """Call func(*args, **kwargs); on a retryable error wait 1, 2, 4 s (+ jitter) and retry."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            if not is_retryable(err) or attempt == MAX_RETRIES:
                raise
            delay = BASE_DELAY * 2 ** attempt + random.uniform(0, 1)
            retry_after = get_retry_after(err)
            if retry_after is not None:
                delay = max(delay, retry_after)
            stats["retries"] += 1
            print(f"retry {attempt + 1}/{MAX_RETRIES} after {type(err).__name__}, waiting {delay:.1f}s")
            time.sleep(delay)
