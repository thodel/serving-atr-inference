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


# ── the second shape of 429: a window, not a backoff (#89) ───────────────────

QUOTA = (
    "429 Client Error: Too Many Requests for url: "
    "https://huggingface.co/api/datasets/dh-unibe/"
    "image-text_koenigsfelden-charters-post-1500/tree/fac683b/"
    "data%2Ftrain%2Fu-17_0877?recursive=True&expand=False\n\n"
    "We had to rate limit you, you hit the quota of 1000 api requests per "
    "5 minutes period. Upgrade to a PRO user or Team/Enterprise organization "
    "account to get higher limits."
)


class TestQuotaWindow:
    """What killed the first v4 attempt on 15.09.2026.

    This 429 states no backoff at all. The old reader fell through to its 5-second
    default and the cap held every wait under 60 s, so every retry came back while
    the five-minute window was still running.
    """

    def test_a_quota_message_waits_the_window_not_five_seconds(self):
        from atr_serving.training.prepare import _retry_after
        assert _retry_after(RuntimeError(QUOTA)) == 300.0

    def test_the_cap_is_large_enough_to_hold_a_window(self):
        assert HUB_RETRY_CAP_S >= 300.0

    def test_the_first_wait_really_is_the_whole_window(self):
        waits = []
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError(QUOTA)
            return "ok"

        assert with_hub_retry(flaky, sleep=waits.append) == "ok"
        assert waits == [300.0]

    def test_a_stated_backoff_still_wins_over_a_window(self):
        """Both in one message: the hub's own number is the more specific answer."""
        from atr_serving.training.prepare import _retry_after
        both = QUOTA + "\nRetry after 12 sec"
        assert _retry_after(RuntimeError(both)) == 12.0

    def test_an_hour_long_window_is_read_as_an_hour(self):
        from atr_serving.training.prepare import _retry_after
        message = "429: you hit the quota of 50,000 api requests per 1 hour period"
        assert _retry_after(RuntimeError(message)) == 3600.0

    def test_a_call_that_cannot_fit_the_quota_still_fails(self):
        """Retrying cannot fix a call that overspends the quota by itself.

        Resolving 1,185 project directories costs 1,185 requests against a quota
        of 1,000. Every attempt spends them again. The backoff is not the fix —
        collapse_complete_selection is — and this test pins that the retry does
        not pretend otherwise.
        """
        with pytest.raises(RuntimeError, match="quota"):
            with_hub_retry(lambda: (_ for _ in ()).throw(RuntimeError(QUOTA)),
                           attempts=3, sleep=lambda _: None)
