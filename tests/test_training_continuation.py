"""Continuing while the curve still improves (#88).

kraken has had `--quit early --min-epochs --lag` all along; the VLM backend had a
fixed epoch count, so a run either stopped mid-improvement or burned hours after
it had plateaued. `kraken-medieval-shard00-std` is the reference behaviour: a
`--epochs 30` schedule that ran to 66 and peaked at 21.
"""

import pytest

from atr_serving.training.continuation import (
    ContinuationPolicy,
    should_stop,
)


def policy(**kw):
    base = dict(min_epochs=1, max_epochs=100, patience=2, min_delta=0.0)
    base.update(kw)
    return ContinuationPolicy(**base)


class TestTheThreeBounds:
    def test_an_improving_curve_keeps_going(self):
        v = should_stop([1.0, 0.9, 0.8, 0.7], policy())
        assert v.stop is False and v.best_epoch == 4

    def test_a_plateau_stops_once_patience_runs_out(self):
        v = should_stop([1.0, 0.9, 0.91, 0.92], policy(patience=2))
        assert v.stop is True and v.best_epoch == 2 and v.since_best == 2

    def test_patience_is_not_yet_exhausted_after_one_bad_epoch(self):
        v = should_stop([1.0, 0.9, 0.91], policy(patience=2))
        assert v.stop is False and v.since_best == 1

    def test_the_ceiling_stops_a_curve_that_is_still_improving(self):
        """Unbounded 'while improving' on a shared GPU is not a policy."""
        v = should_stop([1.0, 0.9, 0.8], policy(max_epochs=3))
        assert v.stop is True and "max_epochs" in v.reason

    def test_the_floor_outranks_patience(self):
        """Early QLoRA noise can look like a plateau for an epoch or two."""
        v = should_stop([1.0, 1.1, 1.2], policy(min_epochs=5, patience=1))
        assert v.stop is False and "min_epochs" in v.reason

    def test_the_ceiling_outranks_the_floor(self):
        """Contradictory bounds must resolve to the safe one, not to an error."""
        v = should_stop([1.0, 1.0], ContinuationPolicy(min_epochs=2, max_epochs=2,
                                                       patience=5))
        assert v.stop is True


class TestMinDelta:
    def test_an_improvement_in_the_fifth_decimal_does_not_count(self):
        """Without min_delta a run never stops on its own."""
        history = [1.0, 0.999999, 0.999998, 0.999997]
        assert should_stop(history, policy(patience=2, min_delta=1e-3)).stop is True

    def test_the_same_history_never_stops_without_it(self):
        history = [1.0, 0.999999, 0.999998, 0.999997]
        assert should_stop(history, policy(patience=2, min_delta=0.0)).stop is False


class TestDirection:
    def test_a_loss_improves_downward(self):
        assert should_stop([0.5, 0.4], policy()).best_epoch == 2

    def test_an_accuracy_improves_upward(self):
        v = should_stop([0.70, 0.82, 0.81], policy(greater_is_better=True))
        assert v.best_epoch == 2 and v.best_value == 0.82

    def test_reading_an_accuracy_as_a_loss_inverts_the_verdict(self):
        """The direction is the whole meaning of the number; a wrong default here
        would stop every improving run and continue every stalled one."""
        rising = [0.70, 0.75, 0.80]
        assert should_stop(rising, policy(patience=2)).stop is True          # as loss
        assert should_stop(rising, policy(patience=2, greater_is_better=True)).stop is False


class TestEdges:
    def test_no_history_is_not_a_decision(self):
        assert should_stop([], policy()).stop is False

    def test_a_single_evaluation_continues(self):
        assert should_stop([1.0], policy()).stop is False

    @pytest.mark.parametrize("kw,match", [
        (dict(min_epochs=5, max_epochs=3), "below min_epochs"),
        (dict(patience=0), "at least 1"),
        (dict(min_delta=-1.0), "not be negative"),
    ])
    def test_an_impossible_policy_is_refused_at_construction(self, kw, match):
        with pytest.raises(ValueError, match=match):
            ContinuationPolicy(**{**dict(min_epochs=1, max_epochs=10), **kw})

    def test_the_verdict_is_readable_because_it_is_logged(self):
        text = str(should_stop([1.0, 0.9, 0.95, 0.96], policy(patience=2)))
        assert text.startswith("stop:") and "best epoch 2" in text
