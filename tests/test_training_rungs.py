"""S9: successive-halving ladder and promotion (#91)."""

import pytest

from atr_serving.training.rungs import RungError, plan_rungs, promote


def test_the_ladder_from_the_plan():
    rungs = plan_rungs(45, eta=3, base_epochs=3, pages=[2500, 5000, 24744])
    assert [(r.configs, r.epochs, r.pages) for r in rungs] == [
        (45, 3, 2500),
        (15, 9, 5000),
        (5, 27, 24744),
        (1, 81, 24744),
    ]


def test_pages_are_padded_with_the_last_value():
    """A short pages list means "and the full shard from there on" — the final
    rung must never train on less data than the winner will be judged on."""
    rungs = plan_rungs(9, eta=3, base_epochs=2, pages=[1000])
    assert [r.pages for r in rungs] == [1000, 1000, 1000]


def test_ladder_terminates_at_one_config():
    assert plan_rungs(1)[0].configs == 1
    assert len(plan_rungs(1)) == 1


def test_max_rungs_truncates():
    assert len(plan_rungs(81, eta=3, max_rungs=2)) == 2


def test_rejects_incoherent_plans():
    with pytest.raises(RungError):
        plan_rungs(0)
    with pytest.raises(RungError):
        plan_rungs(9, eta=1)
    with pytest.raises(RungError):
        plan_rungs(9, base_epochs=0)


def test_promotes_the_top_third_by_accuracy():
    scores = {f"c{i}": i / 10 for i in range(9)}
    result = promote(scores, eta=3)
    assert result.promoted == ["c8", "c7", "c6"]
    assert len(result.eliminated) == 6


def test_ties_break_by_config_id_so_reruns_agree():
    first = promote({"b": 0.8, "a": 0.8, "c": 0.1}, eta=3, keep=1)
    second = promote({"c": 0.1, "a": 0.8, "b": 0.8}, eta=3, keep=1)
    assert first.promoted == second.promoted == ["a"]


def test_a_config_without_a_score_never_promotes_and_is_reported():
    """A crashed run has no evidence behind it. Promoting it would spend the next
    rung's budget on nothing, and dropping it silently would make a sweep that
    lost a third of its candidates to a bug look like one that worked."""
    result = promote({"good": 0.9, "crashed": None, "poor": 0.1}, eta=3)
    assert result.promoted == ["good"]
    assert result.unscored == ["crashed"]
    assert "crashed" not in result.eliminated


def test_keep_is_a_fraction_of_everything_that_entered_the_rung():
    """Six configs, three of which crashed: the field still narrows to two rather
    than promoting every survivor."""
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "x": None, "y": None, "z": None}
    result = promote(scores, eta=3)
    assert result.promoted == ["a", "b"]
    assert result.eliminated == ["c"]


def test_keep_override_and_empty_input():
    assert promote({"a": 1.0, "b": 0.5}, keep=2).promoted == ["a", "b"]
    with pytest.raises(RungError):
        promote({})
    with pytest.raises(RungError):
        promote({"a": 1.0}, eta=1)
