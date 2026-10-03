"""Values this gateway shares with the trainer on asteraix, and nothing else.

They are not settings: nobody sets them per box, and a wrong one is not a
configuration mistake but a protocol mismatch. Until #207 they were imported
from ``atr_serving.training.contracts`` and ``atr_serving.training.promote``,
which was convenient exactly as long as the training package sat in this tree —
and would have broken the gateway the day it was deleted
(``docs/SPLIT_PLAN.md`` E3: a thin seam, no shared Python dependency).

So they are stated here, with the value written out, and
``tests/test_seam_values.py`` pins each literal: a change on either side has to
be deliberate on both. The table in
``docs/INFRASTRUCTURE.md#shared-values`` carries the same two rows, and names
what a disagreement does.
"""

from __future__ import annotations

#: Pixels one image may carry into a VLM, per the level it was trained for.
#:
#: **Serving replays the scale training used.** A processor divides this by the
#: area of one merged patch to get visual tokens, and that area is
#: model-specific: 32² for Qwen3-VL (patch 16 x merge 2). The figures carry the
#: intended token counts — 256 for a line, 2048 for a page — against that grid,
#: because that is the family both halves train and serve.
#:
#: A model given an image at a scale it never trained on does not fail, it
#: answers briefly: ``vllm serve`` is launched with no processor kwargs, so
#: without this a full archival scan arrives at Qwen3-VL's own default of 16384
#: tokens, eight times the training scale (``pipeline.visual_budget``).
#:
#: Must equal ``atr_training.contracts.VLM_PIXEL_BUDGET``. On the training side
#: the runtime re-derives the cap from the processor's own patch_size and
#: merge_size and reports it, so a base with another grid cannot quietly train
#: at a different budget; this side has no processor to ask and trusts the value.
VLM_PIXEL_BUDGET: dict[str, int] = {"line": 256 * 32 * 32, "page": 2048 * 32 * 32}

#: The header the trainer's promotion gate sends, value ``"1"``.
#:
#: The model under test is registered ``enabled: false``, and this gateway
#: refuses a disabled id to every caller — so without a way to ask for exactly
#: that one, the gate could not pass at all: every kraken job's gate got
#: ``404 unknown model`` (#138 review). Honoured only with the shared registry
#: on, and only for a trained registration without a ``disabled_reason``
#: (``api/routes.py`` ``_awaiting_the_gate``).
#:
#: Must equal ``atr_training.promote.PROMOTION_GATE_HEADER``. A disagreement is
#: quiet: every job completes and every trained model stays disabled.
PROMOTION_GATE_HEADER = "X-ATR-Promotion-Gate"
