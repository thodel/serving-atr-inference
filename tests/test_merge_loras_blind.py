"""A base that loads blind is refused, not merged (#220).

14.10.2026, idhefix, ``.venvs/vlm-train``: ``lightonai/LightOnOCR-2-1B-base`` loaded
as ``Mistral3ForConditionalGeneration`` with the whole vision side "newly
initialized", peft warned that the adapter's vision LoRA had nothing to attach
to, and the merge said DONE. The curated entry is enabled and pinned.

No torch here: ``merge_one`` is driven with stand-ins for the three libraries it
imports, which is enough to see what it does with the loading report.
"""

from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path

import pytest

from atr_serving.registry import ModelSpec
from scripts import merge_loras
from scripts.merge_loras import MergeRefused, adapter_keys_summary, loading_failure

VISION = [f"model.vision_tower.transformer.layers.{i}.{name}"
          for i in range(24)
          for name in ("attention.k_proj.weight", "attention.o_proj.weight",
                       "attention.q_proj.weight", "attention.v_proj.weight",
                       "attention_norm.weight", "feed_forward.down_proj.weight",
                       "feed_forward.gate_proj.weight", "feed_forward.up_proj.weight",
                       "ffn_norm.weight")]
MISSING = ["model.multi_modal_projector.linear_1.weight",
           "model.multi_modal_projector.linear_2.weight",
           "model.multi_modal_projector.norm.weight",
           "model.multi_modal_projector.patch_merger.merging_layer.weight",
           "model.vision_tower.ln_pre.weight", "model.vision_tower.patch_conv.weight",
           *VISION]
UNEXPECTED = ["vision_encoder.ln_pre.weight", "vision_encoder.patch_conv.weight"]

PEFT_WARNING = (
    "Found missing adapter keys while loading the checkpoint: ["
    + ", ".join(repr(f"base_model.model.model.vision_tower.transformer.layers.{i}."
                     f"feed_forward.{m}.lora_{ab}.default.weight")
                for i in range(24) for m in ("gate_proj", "up_proj", "down_proj")
                for ab in ("A", "B"))
    + "]."
)


# ── the two pure parts ───────────────────────────────────────────────────────

def test_a_clean_load_is_not_a_failure():
    assert loading_failure([], []) is None
    assert loading_failure([], ["lm_head.weight"]) is None, \
        "unused checkpoint tensors alone do not make the model blind"


def test_the_incident_is_a_refusal_that_names_the_blocks():
    why = loading_failure(MISSING, UNEXPECTED)
    assert why is not None
    assert f"{len(MISSING)} weights" in why and "newly initialised" in why
    assert "multi_modal_projector, vision_tower" in why, "the blocks, not 222 names"
    assert "model.multi_modal_projector.linear_1.weight" in why
    assert f"{len(UNEXPECTED)} checkpoint tensors went unused" in why
    assert "vision_encoder.ln_pre.weight" in why
    assert "nothing was written" in why and "disabled_reason" in why


def test_the_peft_warning_becomes_a_count_and_the_blocks():
    line = adapter_keys_summary(PEFT_WARNING)
    assert line.startswith("peft found 144 adapter keys")
    assert "blocks: vision_tower" in line
    assert "no-ops" in line
    assert len(line) < 400, "one line, not the 144-name list again"


def test_a_peft_warning_that_carries_no_list_is_passed_through_shortened():
    line = adapter_keys_summary("Found missing adapter keys while loading: see log " * 30)
    assert line.startswith("peft: ") and len(line) <= 310


# ── merge_one with stand-ins ─────────────────────────────────────────────────

class _Model:
    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save_pretrained(self, out_dir, **_kw) -> None:
        self.saved.append(Path(out_dir))


class _Peft:
    def __init__(self, base, warn: str | None) -> None:
        self.base, self.warn = base, warn

    def merge_and_unload(self):
        return self.base


@pytest.fixture
def libs(monkeypatch):
    """torch, peft and transformers as merge_one sees them, scripted per test."""
    state = {"info": {"missing_keys": [], "unexpected_keys": []}, "peft_warning": None,
             "model": _Model(), "processor": _Model()}

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"

    peft = types.ModuleType("peft")

    class PeftModel:
        @staticmethod
        def from_pretrained(base, adapter):
            if state["peft_warning"]:
                warnings.warn(state["peft_warning"], UserWarning, stacklevel=1)
            return _Peft(base, state["peft_warning"])

    peft.PeftModel = PeftModel

    transformers = types.ModuleType("transformers")

    class AutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(name, **kw):
            assert kw.get("output_loading_info") is True, "the report has to be asked for"
            return state["model"], state["info"]

    class AutoProcessor:
        @staticmethod
        def from_pretrained(name, **kw):
            return state["processor"]

    transformers.AutoModelForImageTextToText = AutoModelForImageTextToText
    transformers.AutoProcessor = AutoProcessor

    for name, module in (("torch", torch), ("peft", peft), ("transformers", transformers)):
        monkeypatch.setitem(sys.modules, name, module)
    return state


def _spec(tmp_path: Path) -> ModelSpec:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_version": "0.21.0"}),
                                                 encoding="utf-8")
    return ModelSpec(id="lightonocr-catmus-caroline", engine="vllm", local_path=str(adapter),
                     base_model="lightonai/LightOnOCR-2-1B-base")


def test_a_blind_base_is_refused_before_anything_is_written(tmp_path: Path, libs):
    libs["info"] = {"missing_keys": MISSING, "unexpected_keys": UNEXPECTED}
    out = tmp_path / "merged" / "lightonocr-catmus-caroline"

    with pytest.raises(MergeRefused, match="loaded blind"):
        merge_loras.merge_one(_spec(tmp_path), out)

    assert not out.exists(), "no directory: merged_state must not find it complete"
    assert libs["model"].saved == [] and libs["processor"].saved == []


def test_a_clean_base_is_merged_as_before(tmp_path: Path, libs, capsys):
    out = tmp_path / "merged" / "lightonocr-catmus-caroline"

    merge_loras.merge_one(_spec(tmp_path), out)

    assert libs["model"].saved == [out] and libs["processor"].saved == [out]
    assert "DONE" in capsys.readouterr().out


def test_the_peft_warning_is_one_line_on_stderr_and_not_fatal(tmp_path: Path, libs, capsys):
    libs["peft_warning"] = PEFT_WARNING
    out = tmp_path / "merged" / "lightonocr-catmus-caroline"

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # nothing may escape as a bare warning
        merge_loras.merge_one(_spec(tmp_path), out)

    captured = capsys.readouterr()
    assert "[lightonocr-catmus-caroline] WARNING: peft found 144 adapter keys" in captured.err
    assert libs["model"].saved == [out]


def test_main_prints_the_refusal_readably_and_counts_it(tmp_path: Path, libs, monkeypatch,
                                                        capsys):
    """``FAILED: {exc!r}`` is right for a traceback-shaped surprise; a refusal is
    a sentence for an operator and is printed as one."""
    from atr_serving.config import Settings
    from atr_serving.registry import Registry

    libs["info"] = {"missing_keys": MISSING, "unexpected_keys": []}
    spec = _spec(tmp_path)
    settings = Settings(api_key="k" * 32, vllm_merged_dir=tmp_path / "merged")
    monkeypatch.setattr(merge_loras, "get_settings", lambda: settings)
    monkeypatch.setattr(merge_loras, "registered_models", lambda _s: Registry([spec]))
    monkeypatch.setattr(merge_loras, "preflight", lambda _s: [])
    monkeypatch.setattr(sys, "argv", ["merge_loras.py", "--only", spec.id])

    assert merge_loras.main() == 1

    err = capsys.readouterr().err
    assert "FAILED: the base loaded blind" in err and "MergeRefused(" not in err
    assert "1 failed: lightonocr-catmus-caroline" in err
