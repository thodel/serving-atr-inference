"""Surviving a hub rate limit in prepare (#89).

Job 20260822T143612Z-kraken-medieval-german-v2 died in `prepare` on:

    429 Too Many Requests: you have reached your 'api' rate limit.
    Retry after 9 sec

Nine seconds, against a stage that had been running for minutes and would have
run for hours. The limit is easy to reach honestly: verifying a four-dataset
corpus lists every repo and sizes ~1,800 shards before a page is read.
"""

import pytest

from atr_serving.training.prepare import (
    HUB_RETRY_CAP_S,
    _retry_after,
    with_hub_retry,
)

LIMIT = ("429 Too Many Requests: you have reached your 'api' rate limit. "
         "Retry after 9 sec")


class TestRecognisingTheLimit:
    def test_the_real_message_yields_the_hubs_own_backoff(self):
        assert _retry_after(RuntimeError(LIMIT)) == 9.0

    def test_a_rate_limit_without_a_number_gets_a_default(self):
        assert _retry_after(RuntimeError("429 Too Many Requests")) == 5.0

    def test_the_wording_alone_is_enough(self):
        """`datasets` wraps hub errors, so the class does not survive but the
        message does."""
        assert _retry_after(RuntimeError("You have reached your rate limit")) == 5.0

    @pytest.mark.parametrize("message", [
        "connection reset by peer",
        "404 Client Error: Entry Not Found",
        "ValueError: Coordinate 'right' is less than 'left'",
    ])
    def test_anything_else_is_not_retried(self, message):
        assert _retry_after(RuntimeError(message)) is None


class TestRetrying:
    def test_it_returns_once_the_limit_clears(self):
        attempts = []

        def flaky():
            attempts.append(len(attempts))
            if len(attempts) < 3:
                raise RuntimeError(LIMIT)
            return "loaded"

        assert with_hub_retry(flaky, sleep=lambda _: None) == "loaded"
        assert len(attempts) == 3

    def test_it_waits_what_the_hub_asked_for(self):
        waits = []
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError(LIMIT)
            return "ok"

        with_hub_retry(flaky, sleep=waits.append)
        assert waits == [9.0]

    def test_the_wait_grows_but_is_capped(self):
        waits = []
        with pytest.raises(RuntimeError):
            with_hub_retry(lambda: (_ for _ in ()).throw(RuntimeError(LIMIT)),
                           attempts=6, sleep=waits.append)
        assert waits == sorted(waits)
        assert max(waits) <= HUB_RETRY_CAP_S

    def test_a_persistent_limit_still_fails_rather_than_hanging(self):
        """Retrying is not the same as never giving up."""
        with pytest.raises(RuntimeError, match="rate limit"):
            with_hub_retry(lambda: (_ for _ in ()).throw(RuntimeError(LIMIT)),
                           attempts=2, sleep=lambda _: None)

    def test_an_unrelated_failure_is_raised_immediately(self):
        calls = []

        def broken():
            calls.append(1)
            raise ValueError("Coordinate 'right' is less than 'left'")

        with pytest.raises(ValueError):
            with_hub_retry(broken, sleep=lambda _: None)
        assert len(calls) == 1          # not retried
