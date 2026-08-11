"""The step-count guard (#72).

The fixtures are the real runs, from docs/TRAINING_PLAN.md §9a: the guard has to
flag the configuration that produced CER 0.98 and pass the one that worked.
"""

from __future__ import annotations

import pytest

from atr_serving.training.convergence import (
    FLOOR_FROM_SCRATCH,
    check_convergence,
    floor_for,
    plan_steps,
)


# ── the arithmetic ──────────────────────────────────────────────────────────
def test_the_thun_run_that_failed():
    """2,087 transcribed lines, 189 held out for eval → 1,898 training, batch 256."""
    budget = plan_steps(1898, 256, 50)
    assert budget.steps_per_epoch == 8      # ceil(1898/256) = 8
    assert budget.total_steps == 400        # 8 batches/epoch x 50


def test_gradient_accumulation_counts_as_one_step():
    """The runbook's own OOM remedy is batch 64 x accumulate 4. That is the same
    effective batch as 256 and must be judged identically — otherwise the
    documented workaround walks straight past the guard."""
    assert plan_steps(1898, 64 * 4, 50).total_steps == plan_steps(1898, 256, 50).total_steps


def test_a_partial_batch_still_costs_a_step():
    assert plan_steps(10, 256, 1).steps_per_epoch == 1


# ── the floors ──────────────────────────────────────────────────────────────
def test_from_scratch_needs_far_more_than_a_fine_tune():
    assert floor_for("kraken", from_scratch=True) > floor_for("kraken", from_scratch=False)


def test_a_qlora_adapter_is_judged_on_its_own_scale():
    """qwen3vl-thun-smoke improved CER 1.837 -> 0.466 in 38 steps. A floor in the
    hundreds would refuse work that demonstrably functions."""
    assert floor_for("vllm", from_scratch=False) < 38


# ── the verdicts, against the recorded runs ─────────────────────────────────
def test_the_run_that_produced_cer_098_is_refused():
    verdict = check_convergence("kraken", from_scratch=True, train_lines=1898,
                                effective_batch=256, epochs=50)
    assert verdict.ok is False
    assert "1,898 training lines" in verdict.reason
    assert "8 step(s) per epoch" in verdict.reason
    assert str(FLOOR_FROM_SCRATCH) in verdict.reason.replace(",", "")


def test_the_refusal_names_all_three_ways_out():
    verdict = check_convergence("kraken", from_scratch=True, train_lines=1898,
                                effective_batch=256, epochs=50)
    assert "base_model" in verdict.reason          # fine-tune instead
    assert "lower batch_size to ~" in verdict.reason
    assert "raise epochs" in verdict.reason
    assert '"force": true' in verdict.reason       # and how to override it


def test_the_suggested_batch_actually_clears_the_floor():
    """An actionable remedy has to be one that works, not merely a smaller number."""
    verdict = check_convergence("kraken", from_scratch=True, train_lines=1898,
                                effective_batch=256, epochs=50)
    suggested = int(verdict.reason.split("lower batch_size to ~")[1].split(" ")[0])
    assert plan_steps(1898, suggested, 50).total_steps >= verdict.floor


def test_the_vlm_smoke_run_is_allowed():
    """38 steps, and it worked — the guard must not refuse the one run that has
    produced a real improvement on this box."""
    verdict = check_convergence("vllm", from_scratch=False, train_lines=783,
                                effective_batch=16, epochs=1)
    assert verdict.ok is True


def test_a_well_configured_finetune_passes():
    """The runbook's corrected example: Thun at batch 16, fine-tuned."""
    verdict = check_convergence("kraken", from_scratch=False, train_lines=1898,
                                effective_batch=16, epochs=50)
    assert verdict.ok is True
    assert verdict.budget.total_steps > 5000


def test_the_same_config_from_scratch_is_still_refused_at_a_bigger_batch():
    verdict = check_convergence("kraken", from_scratch=True, train_lines=1898,
                                effective_batch=64, epochs=50)
    assert verdict.ok is False          # 1,500 steps, still under the 2,000 floor


def test_a_large_corpus_passes_at_the_documented_defaults():
    """The defaults are correct for the corpus they were written against; the
    guard must not condemn them."""
    verdict = check_convergence("kraken", from_scratch=True, train_lines=18_000_000,
                                effective_batch=256, epochs=50)
    assert verdict.ok is True


# ── absence is not a verdict ────────────────────────────────────────────────
@pytest.mark.parametrize("lines", [None, 0])
def test_no_line_count_means_no_judgement(lines):
    """Refusing on a missing number would block a run for a reason about us
    rather than about the configuration."""
    assert check_convergence("kraken", True, lines, 256, 50) is None
