"""A base whose checkpoint names its weights differently is renamed, not merged blind (#226).

``lightonai/LightOnOCR-2-1B-base`` says ``architectures: [LightOnOCRForConditionalGeneration]``
but ``model_type: mistral3``. transformers builds a Mistral3 and looks for
``model.vision_tower.*`` / ``model.multi_modal_projector.*``; the checkpoint has the
same 218 + 4 tensors as ``model.vision_encoder.*`` / ``model.vision_projection.*``
(its safetensors header, read 17.10.2026). Without a renaming the whole vision side
was initialised at random (#220) — and since #223 refused.

No torch: the libraries are stand-ins, the names are the real ones.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

from atr_serving.registry import ModelSpec
from scripts import merge_loras
from scripts.merge_loras import (
    KEY_MAPPINGS,
    MergeRefused,
    adapter_key_mapping,
    key_mapping_for,
)

LIGHTONOCR_CFG = {"architectures": ["LightOnOCRForConditionalGeneration"], "model_type": "mistral3"}

#: What transformers 4.57.6 reported missing on 14.10. and 17.10.2026 (the names a
#: Mistral3 expects), in full.
VISION = [f"model.vision_tower.transformer.layers.{i}.{name}"
          for i in range(24)
          for name in ("attention.k_proj.weight", "attention.o_proj.weight",
                       "attention.q_proj.weight", "attention.v_proj.weight",
                       "attention_norm.weight", "feed_forward.down_proj.weight",
                       "feed_forward.gate_proj.weight", "feed_forward.up_proj.weight",
                       "ffn_norm.weight")]
EXPECTED = ["model.multi_modal_projector.linear_1.weight",
            "model.multi_modal_projector.linear_2.weight",
            "model.multi_modal_projector.norm.weight",
            "model.multi_modal_projector.patch_merger.merging_layer.weight",
            "model.vision_tower.ln_pre.weight", "model.vision_tower.patch_conv.weight",
            *VISION]

#: transformers 4.57.6, models/mistral3/modeling_mistral3.py, the conditional-generation class.
MISTRAL3_CONVERSION = {
    "^language_model.model": "model.language_model",
    "^vision_tower": "model.vision_tower",
    "^multi_modal_projector": "model.multi_modal_projector",
    "^language_model.lm_head": "lm_head",
}


def _checkpoint_name(expected: str) -> str:
    """The name the LightOnOCR checkpoint uses for a tensor Mistral3 calls ``expected``."""
    return (expected.replace("model.vision_tower.", "model.vision_encoder.", 1)
                    .replace("model.multi_modal_projector.", "model.vision_projection.", 1))


def _apply(mapping: dict[str, str], key: str) -> str:
    """transformers' and peft's loop: patterns in order, the first that matches wins."""
    for pattern, replacement in mapping.items():
        new, n = re.subn(pattern, replacement, key)
        if n:
            return new
    return key


# ── the table ────────────────────────────────────────────────────────────────

def test_the_mapping_is_chosen_by_the_configs_architectures():
    assert key_mapping_for(LIGHTONOCR_CFG) == KEY_MAPPINGS["LightOnOCRForConditionalGeneration"]
    assert key_mapping_for({"architectures": ["Qwen3VLForConditionalGeneration"]}) is None
    assert key_mapping_for({"model_type": "mistral3"}) is None, "model_type alone is not enough"
    assert key_mapping_for({}) is None
    assert key_mapping_for(LIGHTONOCR_CFG) is not KEY_MAPPINGS[
        "LightOnOCRForConditionalGeneration"], "a copy, so a caller cannot edit the table"


def test_the_real_checkpoint_names_become_exactly_the_expected_ones():
    """222 missing = 222 unused, name for name: the incident, as a mapping."""
    mapping = key_mapping_for(LIGHTONOCR_CFG)
    checkpoint = [_checkpoint_name(k) for k in EXPECTED]
    assert len(set(checkpoint)) == 222 and not set(checkpoint) & set(EXPECTED)
    assert sorted(_apply(mapping, k) for k in checkpoint) == sorted(EXPECTED)


def test_the_language_model_keys_are_left_alone():
    mapping = key_mapping_for(LIGHTONOCR_CFG)
    for key in ("model.language_model.layers.0.mlp.gate_proj.weight",
                "model.language_model.embed_tokens.weight", "lm_head.weight"):
        assert _apply(mapping, key) == key


def test_the_adapter_mapping_keeps_the_classs_own_after_the_rename():
    """peft uses the class's conversion mapping when given none; given one, it uses
    only that. So the class's is appended, and the base's renames come first."""
    mapping = key_mapping_for(LIGHTONOCR_CFG)
    base = types.SimpleNamespace(_checkpoint_conversion_mapping=MISTRAL3_CONVERSION)
    merged = adapter_key_mapping(mapping, base)
    assert list(merged)[:2] == list(mapping)
    assert {k: merged[k] for k in MISTRAL3_CONVERSION} == MISTRAL3_CONVERSION
    # peft strips "base_model.model." before matching; a vision LoRA weight lands
    # where the renamed base has its module.
    key = "model.vision_encoder.transformer.layers.0.feed_forward.gate_proj.lora_A.weight"
    assert _apply(merged, key) == (
        "model.vision_tower.transformer.layers.0.feed_forward.gate_proj.lora_A.weight")
    assert adapter_key_mapping(mapping, object()) == mapping, "a class without one"


# ── merge_one with stand-ins ─────────────────────────────────────────────────

class _Model:
    _checkpoint_conversion_mapping = MISTRAL3_CONVERSION

    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save_pretrained(self, out_dir, **_kw) -> None:
        self.saved.append(Path(out_dir))


@pytest.fixture
def libs(monkeypatch):
    """A transformers that loads blind unless the renaming is passed, as on idhefix."""
    state = {"base_kw": None, "peft_kw": None, "model": _Model(), "processor": _Model(),
             "blind_even_with_mapping": False, "cfg": LIGHTONOCR_CFG}

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"

    peft = types.ModuleType("peft")

    class _Peft:
        def __init__(self, base):
            self.base = base

        def merge_and_unload(self):
            return self.base

    class PeftModel:
        @staticmethod
        def from_pretrained(base, adapter, **kw):
            state["peft_kw"] = kw
            return _Peft(base)

    peft.PeftModel = PeftModel

    transformers = types.ModuleType("transformers")

    class AutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(name, **kw):
            state["base_kw"] = kw
            renamed = "key_mapping" in kw and not state["blind_even_with_mapping"]
            missing = [] if renamed else EXPECTED
            unused = [] if renamed else [_checkpoint_name(k) for k in EXPECTED]
            return state["model"], {"missing_keys": missing, "unexpected_keys": unused}

    class AutoProcessor:
        @staticmethod
        def from_pretrained(name, **kw):
            return state["processor"]

    transformers.AutoModelForImageTextToText = AutoModelForImageTextToText
    transformers.AutoProcessor = AutoProcessor

    for name, module in (("torch", torch), ("peft", peft), ("transformers", transformers)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(merge_loras, "_base_config", lambda _name: state["cfg"])
    return state


def _spec(tmp_path: Path, base: str = "lightonai/LightOnOCR-2-1B-base") -> ModelSpec:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_version": "0.18.1"}),
                                                 encoding="utf-8")
    return ModelSpec(id="lightonocr-catmus-caroline", engine="vllm", local_path=str(adapter),
                     base_model=base)


def test_lightonocr_is_renamed_on_both_loads_and_merged(tmp_path: Path, libs, capsys):
    out = tmp_path / "merged" / "lightonocr-catmus-caroline"

    merge_loras.merge_one(_spec(tmp_path), out)

    assert libs["base_kw"]["key_mapping"] == KEY_MAPPINGS["LightOnOCRForConditionalGeneration"]
    assert libs["base_kw"]["output_loading_info"] is True
    peft_mapping = libs["peft_kw"]["key_mapping"]
    assert list(peft_mapping)[:2] == list(KEY_MAPPINGS["LightOnOCRForConditionalGeneration"])
    assert "^vision_tower" in peft_mapping, "the class's own mapping is kept for the adapter"
    assert libs["model"].saved == [out] and libs["processor"].saved == [out]
    printed = capsys.readouterr().out
    assert "renaming checkpoint keys: ^model\\.vision_encoder\\. -> model.vision_tower." in printed
    assert "DONE" in printed


def test_a_base_that_needs_no_renaming_is_loaded_as_before(tmp_path: Path, libs):
    libs["cfg"] = {"architectures": ["Qwen3VLForConditionalGeneration"]}
    libs["model"] = _Model()

    # This stand-in loads blind without a mapping, which a real Qwen3-VL does not:
    # the point is only that no key_mapping reaches either loader.
    with pytest.raises(MergeRefused):
        merge_loras.merge_one(_spec(tmp_path, base="Qwen/Qwen3-VL-4B-Instruct"),
                              tmp_path / "merged" / "x")
    assert "key_mapping" not in libs["base_kw"]


def test_an_unreadable_config_means_no_renaming(tmp_path: Path, libs, monkeypatch):
    def unreadable(_name):
        raise OSError("offline")

    monkeypatch.setattr(merge_loras, "_base_config", unreadable)
    with pytest.raises(MergeRefused):
        merge_loras.merge_one(_spec(tmp_path), tmp_path / "merged" / "x")
    assert "key_mapping" not in libs["base_kw"]


def test_a_renaming_that_is_not_enough_is_still_refused(tmp_path: Path, libs):
    """The table is a claim; the blind check is what holds it to account."""
    libs["blind_even_with_mapping"] = True
    out = tmp_path / "merged" / "lightonocr-catmus-caroline"

    with pytest.raises(MergeRefused, match="KEY_MAPPINGS"):
        merge_loras.merge_one(_spec(tmp_path), out)

    assert "key_mapping" in libs["base_kw"]
    assert libs["peft_kw"] is None, "the adapter is never loaded onto a blind base"
    assert not out.exists() and libs["model"].saved == []
