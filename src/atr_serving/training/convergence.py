"""The step-count guard (#72): refuse a run that cannot converge.

`kraken-thun-missiven-v1` was submitted with the documented defaults, ran to
completion, and reported **CER 0.9838** — honestly. The arithmetic that explains
it was available the moment ``prepare`` finished:

    2,087 transcribed lines, 189 of them the held-out eval projects → **1,898
    training lines**. At batch 256 that is 8 batches per epoch (7 full + 1
    partial), so **400 optimizer steps** over 50 epochs.

400 optimizer steps for a 15.2 M-parameter network starting from random weights,
with ``1cycle`` ramping and annealing the learning rate across all of them. An
unconverged CTC network has not learned blank-dominance and emits a character at
nearly every timestep — which *is* an insertion-dominated CER, and which then
cost two days to diagnose (11,191 insertions against 2 deletions).

Every other guard in this subsystem protects a resource: the disk, the GPU, the
filesystem, the honesty of the report. This one protects the **experiment**.

**Why it refuses rather than warns.** A warning in a log nobody reads is what the
previous state already amounted to — the run completed and reported its number.
A configuration that cannot learn should not hold the GPU for three hours, so it
is refused where a full disk is refused.

**Why the floors differ per engine.** A QLoRA fine-tune of an 8 B VLM improved CER
from 1.837 to 0.466 in **38 steps**; a from-scratch CTC network needs thousands.
One number cannot serve both, and the case that actually failed — training from
scratch — gets its own, much higher floor.

The floors are judgement calls, not measurements, and are written here rather
than buried so that raising one is a visible decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "StepBudget",
    "ConvergenceVerdict",
    "FLOOR_FROM_SCRATCH",
    "FLOOR_FINETUNE",
    "FLOOR_BY_ENGINE",
    "plan_steps",
    "floor_for",
    "check_convergence",
]

#: A network starting from random weights has to learn the alphabet, the blank
#: symbol and the script. 400 steps produced CER 0.98; a Thun fine-tune at batch
#: 16 would be ~3,600. This sits well below anything reasonable and well above the
#: run that failed.
FLOOR_FROM_SCRATCH = 2_000
#: Starting from trained weights, far fewer steps are needed — the model already
#: knows what characters look like.
FLOOR_FINETUNE = 500
#: Per-engine overrides for the fine-tune floor. ``vllm`` is a QLoRA adapter over a
#: pretrained VLM: the one real run converged usefully in 38 steps, so a floor in
#: the hundreds would refuse work that demonstrably functions.
FLOOR_BY_ENGINE: dict[str, int] = {"vllm": 25}


@dataclass(frozen=True)
class StepBudget:
    """How many optimizer steps a configuration will actually take."""

    train_lines: int
    effective_batch: int
    epochs: int

    @property
    def steps_per_epoch(self) -> int:
        return max(1, math.ceil(self.train_lines / max(1, self.effective_batch)))

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * max(1, self.epochs)

    def batch_for(self, floor: int) -> int:
        """The largest effective batch that would clear ``floor`` at these epochs.

        Used to make the error actionable: "lower batch_size to ~16" beats "too
        few steps".
        """
        wanted_steps_per_epoch = math.ceil(floor / max(1, self.epochs))
        return max(1, self.train_lines // max(1, wanted_steps_per_epoch))


@dataclass(frozen=True)
class ConvergenceVerdict:
    ok: bool
    budget: StepBudget
    floor: int
    reason: str = ""


def plan_steps(train_lines: int, effective_batch: int, epochs: int) -> StepBudget:
    return StepBudget(train_lines=train_lines, effective_batch=effective_batch, epochs=epochs)


def floor_for(engine: str, from_scratch: bool) -> int:
    """Minimum optimizer steps this kind of run needs to be worth starting."""
    if from_scratch:
        return FLOOR_FROM_SCRATCH
    return FLOOR_BY_ENGINE.get(engine, FLOOR_FINETUNE)


def check_convergence(
    engine: str,
    from_scratch: bool,
    train_lines: int | None,
    effective_batch: int,
    epochs: int,
) -> ConvergenceVerdict | None:
    """Judge a configuration. ``None`` when there is nothing to judge.

    A missing line count is not a failure: it means ``prepare`` reported no
    number, and refusing on an absence would block runs for a reason that is
    about us rather than about the configuration.
    """
    if not train_lines or train_lines < 1:
        return None

    budget = plan_steps(train_lines, effective_batch, epochs)
    floor = floor_for(engine, from_scratch)
    if budget.total_steps >= floor:
        return ConvergenceVerdict(True, budget, floor)

    suggested_batch = budget.batch_for(floor)
    needed_epochs = math.ceil(floor / budget.steps_per_epoch)
    start = "from scratch" if from_scratch else f"as a {engine} fine-tune"
    remedy = (
        "fine-tune from a base model instead (`base_model` + `resize: \"union\"`)"
        if from_scratch else "train for longer"
    )
    return ConvergenceVerdict(
        False, budget, floor,
        reason=(
            f"{budget.train_lines:,} training lines at effective batch "
            f"{budget.effective_batch} is {budget.steps_per_epoch} step(s) per epoch; "
            f"over {budget.epochs} epochs that is {budget.total_steps:,} optimizer "
            f"steps, {start}, against a floor of {floor:,}. A run this short does not "
            f"converge — it was 400 steps that produced CER 0.98 on the Thun set "
            f"(see docs/TRAINING_PLAN.md §9a). Either {remedy}, lower batch_size to "
            f"~{suggested_batch} (≈{plan_steps(budget.train_lines, suggested_batch, budget.epochs).total_steps:,} "
            f"steps), or raise epochs to ~{needed_epochs}. Submit with "
            f'"force": true to run it anyway.'
        ),
    )
