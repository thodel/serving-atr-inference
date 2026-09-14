"""How much of the card may this model take?

Three launches of `qwen3vl-german-xix-v1` failed on asterAIx on 2026-09-14, and
all three failed on the same question answered by a constant — 0.70 — that knows
neither the model nor the card:

1. `Free memory on device (18.94/44.45 GiB) < desired GPU memory utilization
   (0.7, 31.11 GiB)` — refused before loading anything.
2. set to 0.35 by hand, an orphaned training process meanwhile gone: loaded the
   weights, then `2.25 GiB KV cache is needed, 2.22 GiB is available` and died.
   One percent short, after a minute of loading.
3. 0.45, chosen by arithmetic on the back of the second message.

The numbers below are those runs. `plan_gpu_budget` is pure, so they are
regression tests rather than a paragraph in a runbook.
"""

import math

import pytest

from atr_serving.manager import (
    KV_HEADROOM,
    MAX_UTILISATION,
    MIN_HEADROOM,
    Budget,
    ManagerError,
    gpu_budget,
    plan_gpu_budget,
)
from atr_serving.registry import ModelSpec


#: `nvidia-smi --query-gpu=memory.total` on asterAIx GPU 1 (L40S).
TOTAL = 45516

#: What `config/models.yaml` claims for the three German-XIX models.
QWEN3VL_MB = 12000


# ── the three failures ───────────────────────────────────────────────────────

def test_the_first_failure_would_have_been_sized_to_fit():
    """19 394 MiB free. 0.70 asked for 31 GB of a card that had 19, and vLLM said
    so without loading anything — the one honest failure of the three."""
    budget = plan_gpu_budget(QWEN3VL_MB, 19394, TOTAL)
    assert budget is not None
    assert budget.utilisation == 0.38
    # what actually has to hold: the ask must fit in what the card reported free
    assert budget.utilisation * TOTAL <= 19394


def test_the_second_failure_the_one_percent_short_one():
    """31 047 MiB free after the orphan died. A hand-picked 0.35 left a KV cache
    2.22 GiB against the 2.25 needed; 0.42 is not a nicer guess, it is 12 000 MiB
    of weights times the KV headroom."""
    budget = plan_gpu_budget(QWEN3VL_MB, 31047, TOTAL)
    assert budget is not None
    assert budget.utilisation == 0.42
    assert budget.utilisation > 0.35, "must beat the value that was 1% short"


def test_an_8b_model_on_a_free_card_lands_where_a_human_put_it():
    """The check that this is sizing and not just shrinking: for the case 0.70 was
    chosen for — an 18 GB model on an empty card — the arithmetic agrees with the
    person who chose it."""
    assert plan_gpu_budget(18000, 44000, TOTAL).utilisation == 0.63


# ── the shape of the arithmetic ──────────────────────────────────────────────

def test_a_free_card_gives_the_model_what_it_asks_for():
    """Plenty free: the size comes from the model, and the card does not enter."""
    budget = plan_gpu_budget(12000, 45000, TOTAL)
    assert budget.utilisation == pytest.approx(12000 * KV_HEADROOM / TOTAL, abs=0.01)
    assert "KV cache" in budget.reason


def test_a_crowded_card_gives_the_model_what_is_left():
    """Less free than the model wants but more than it needs: take the rest, and
    say in the reason that this is a ceiling rather than a request."""
    budget = plan_gpu_budget(12000, 18000, TOTAL)
    assert budget.utilisation * TOTAL <= 18000 - 2048
    assert "all that is free" in budget.reason


def test_the_reserve_is_actually_left_free():
    """vLLM checks `free >= util * total` once, at startup. Everything that grows
    afterwards — this process's own CUDA context, the small engines on the same card —
    has to come out of memory nobody promised to vLLM."""
    budget = plan_gpu_budget(12000, 20000, TOTAL, reserve_mb=4096)
    assert budget.utilisation * TOTAL <= 20000 - 4096


def test_a_card_with_no_room_is_refused_rather_than_launched():
    """The 8.3 GB lesson: vLLM discovers this after loading the weights, a minute
    in, and its message names neither the model nor what holds the memory."""
    assert plan_gpu_budget(12000, 8000, TOTAL) is None


def test_exactly_the_minimum_is_granted_not_refused():
    free = math.ceil(12000 * MIN_HEADROOM) + 2048
    budget = plan_gpu_budget(12000, free, TOTAL)
    assert budget is not None and budget.utilisation > 0


def test_under_the_minimum_is_refused():
    free = math.floor(12000 * MIN_HEADROOM) + 2048 - 1
    assert plan_gpu_budget(12000, free, TOTAL) is None


def test_utilisation_is_floored_never_rounded_up():
    """0.4251 rounded is 0.43, which asks for memory that was measured as absent.
    The direction of the error matters more than its size."""
    budget = plan_gpu_budget(12000, 45000, TOTAL)
    assert budget.utilisation * TOTAL <= 45000 - 2048
    assert budget.utilisation == int(budget.utilisation * 100) / 100


def test_a_model_that_would_take_the_whole_card_stops_short_of_it():
    """A utilisation of 1.0 leaves the driver's own allocations nothing. Reachable
    only with the reserve turned off, which is why the ceiling is a second check
    and not a consequence of the reserve."""
    assert plan_gpu_budget(30000, TOTAL, TOTAL, reserve_mb=0).utilisation == MAX_UTILISATION


# ── what it does not know ────────────────────────────────────────────────────

@pytest.mark.parametrize("vram_mb,total_mb", [(0, TOTAL), (12000, 0), (0, 0)])
def test_without_both_numbers_it_hands_back_the_configured_constant(vram_mb, total_mb):
    """Most of the registry predates vram_mb and carries 0. Refusing to launch
    those would turn a sizing improvement into an outage."""
    budget = plan_gpu_budget(vram_mb, 30000, total_mb, fallback=0.70)
    assert budget == Budget(0.70, budget.reason)
    assert "0.7" in budget.reason


# ── against the live card ────────────────────────────────────────────────────

class _Settings:
    vllm_gpu_memory_utilization = 0.70
    vllm_autosize = True
    vllm_vram_headroom = KV_HEADROOM
    vllm_vram_reserve_mb = 2048


def _spec(vram_mb: int = QWEN3VL_MB) -> ModelSpec:
    return ModelSpec(id="qwen3vl-german-xix-v1", engine="vllm",
                     hf_repo="dh-unibe/qwen3vl-german-xix-v1", vram_mb=vram_mb)


def test_gpu_budget_sizes_from_the_card(monkeypatch):
    monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory",
                        lambda index: (31047, TOTAL))
    assert gpu_budget(_spec(), 1, _Settings()).utilisation == 0.42


def test_gpu_budget_asks_about_the_gpu_it_was_given(monkeypatch):
    """CUDA_VISIBLE_DEVICES renames the card to 0 for the child, but the memory
    that has to hold the model is the memory of the physical card."""
    asked: list[int] = []
    monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory",
                        lambda index: asked.append(index) or (44000, TOTAL))
    gpu_budget(_spec(), 1, _Settings())
    assert asked == [1]


def test_an_unreadable_card_falls_back_instead_of_refusing(monkeypatch):
    """No nvidia-smi (a dev box, a container without the driver) must behave
    exactly as this launcher did before autosizing existed."""
    monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory", lambda index: None)
    budget = gpu_budget(_spec(), 1, _Settings())
    assert budget.utilisation == 0.70 and "unreadable" in budget.reason


def test_autosizing_can_be_switched_off(monkeypatch):
    """An escape hatch that needs no code change at 3 a.m."""
    monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory",
                        lambda index: (31047, TOTAL))
    settings = _Settings()
    settings.vllm_autosize = False
    assert gpu_budget(_spec(), 1, settings).utilisation == 0.70


def test_a_card_that_cannot_hold_the_model_raises_and_names_the_numbers(monkeypatch):
    monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory",
                        lambda index: (8000, TOTAL))
    with pytest.raises(ManagerError) as err:
        gpu_budget(_spec(), 1, _Settings())
    message = str(err.value)
    assert "qwen3vl-german-xix-v1" in message
    assert "8000" in message and str(TOTAL) in message
    assert "/gpu" in message, "the operator needs to be told where to look"
