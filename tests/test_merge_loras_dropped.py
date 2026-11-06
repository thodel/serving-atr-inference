"""Base tensors the save drops go back in (training-atr-models#83).

06.11.2026, idhefix: gemma4-e4b-german-xix-v1 merged with DONE and vLLM 0.29
refused it — k_proj, v_proj and k_norm of the KV-shared layers 24–41 were not in
the checkpoint, because transformers 5 never builds them. The planner is pure;
the file part needs torch and runs on the box.
"""

from __future__ import annotations

from scripts.merge_loras import dropped_base_tensors

SHARED = [f"model.language_model.layers.{i}.self_attn.{m}.weight"
          for i in range(24, 42) for m in ("k_proj", "v_proj", "k_norm")]
KEPT = [f"model.language_model.layers.{i}.self_attn.{m}.weight"
        for i in range(42) for m in ("q_proj", "o_proj", "q_norm")]


def test_the_incident_restores_the_54_shared_kv_tensors():
    plan = dropped_base_tensors(KEPT + SHARED, KEPT)
    assert len(plan) == 54
    assert plan == {k: k for k in SHARED}


def test_nothing_dropped_is_nothing_to_restore():
    assert dropped_base_tensors(KEPT, KEPT) == {}


def test_a_renamed_base_is_compared_under_its_merged_names():
    mapping = {r"^model\.vision_encoder\.": "model.vision_tower."}
    base = ["model.vision_encoder.ln_pre.weight", "model.vision_encoder.patch_conv.weight"]
    saved = ["model.vision_tower.ln_pre.weight"]
    assert dropped_base_tensors(base, saved, mapping) == {
        "model.vision_tower.patch_conv.weight": "model.vision_encoder.patch_conv.weight"}
