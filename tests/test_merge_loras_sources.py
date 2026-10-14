"""What merge_loras merges, and what it refuses before loading anything (#219).

On 14.10.2026 the first full run after #217 merged four ids the gateway never
reads: leftovers of the retired overlay (#143), each with a ``local_path`` that
held no adapter. The preflight could not read their config ("Repo id must be in
the form …" — the hub was asked about a directory) and said "not checked"; each
then loaded the 8B base, the first after a 17 GB download, to fail on the file
that was never there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from atr_serving.config import Settings
from atr_serving.registry import ModelSpec
from scripts import merge_loras
from scripts.merge_loras import Blocker, local_adapter_blocker, peft_blocker, preflight

CURATED = {"models": [{
    "id": "qwen3vl-8b-hebrew", "engine": "vllm", "hf_repo": "wjbmattingly/x",
    "base_model": "Qwen/Qwen3-VL-8B-Instruct", "vram_mb": 18000,
}]}
GHOST = {"models": [{
    "id": "qwen3vl-thun-smoke", "engine": "vllm", "enabled": False,
    "local_path": "/mnt/share/trained/qwen3vl-thun-smoke",
    "base_model": "Qwen/Qwen3-VL-8B-Instruct",
}]}
TRAINED = {
    "id": "vlm-xix-noise-s42", "engine": "vllm", "enabled": False,
    "local_path": "/mnt/share/trained/vlm-xix-noise-s42",
    "base_model": "Qwen/Qwen3-VL-4B-Instruct",
}


@pytest.fixture
def checkout(tmp_path: Path) -> dict:
    config = tmp_path / "config"
    config.mkdir()
    (config / "models.yaml").write_text(yaml.safe_dump(CURATED), encoding="utf-8")
    overlay = config / "models.local.yaml"
    overlay.write_text(yaml.safe_dump(GHOST), encoding="utf-8")
    share = tmp_path / "registry"
    (share / "trained").mkdir(parents=True)
    (share / "trained" / "vlm-xix-noise-s42.yaml").write_text(
        yaml.safe_dump(TRAINED), encoding="utf-8")
    return {"models": config / "models.yaml", "overlay": overlay, "share": share}


def _ids(settings: Settings) -> list[str]:
    return sorted(s.id for s in merge_loras.registered_models(settings).by_engine("vllm")
                  if s.base_model)


# ── the sources ──────────────────────────────────────────────────────────────

def test_the_overlay_next_to_the_registry_is_not_read(checkout):
    """The gateway stopped reading it in #143; a merge for an id the gateway
    never serves has no consumer, however complete the directory it leaves."""
    settings = Settings(api_key="k" * 32, models_config=checkout["models"],
                        registry_root=checkout["share"])
    assert checkout["overlay"].is_file(), "the leftover file is right there"
    assert _ids(settings) == ["qwen3vl-8b-hebrew", "vlm-xix-noise-s42"]


def test_the_overlay_is_read_when_the_setting_names_it(checkout):
    """Exactly the gateway's rule: ``ATR_MODELS_OVERLAY`` brings it back."""
    settings = Settings(api_key="k" * 32, models_config=checkout["models"],
                        registry_root=checkout["share"], models_overlay=checkout["overlay"])
    assert _ids(settings) == ["qwen3vl-8b-hebrew", "qwen3vl-thun-smoke", "vlm-xix-noise-s42"]


def test_without_a_share_the_tracked_registry_alone_is_merged(checkout):
    settings = Settings(api_key="k" * 32, models_config=checkout["models"])
    assert _ids(settings) == ["qwen3vl-8b-hebrew"]


# ── a local adapter that is not there ────────────────────────────────────────

def test_a_local_adapter_without_its_config_is_fatal_and_names_the_path(tmp_path: Path):
    empty = tmp_path / "qwen3vl-thun-smoke"
    empty.mkdir()
    b = local_adapter_blocker(str(empty))
    assert b is not None and b.fatal
    assert str(empty) in b.text and "no adapter_config.json" in b.text

    gone = tmp_path / "never-written"
    b = local_adapter_blocker(str(gone))
    assert b is not None and b.fatal
    assert str(gone) in b.text and "does not exist" in b.text


def test_a_local_adapter_with_its_config_and_a_hub_id_pass(tmp_path: Path):
    adapter = tmp_path / "vlm-xix-noise-s42"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_version": "0.21.0"}', encoding="utf-8")
    assert local_adapter_blocker(str(adapter)) is None
    assert local_adapter_blocker("wjbmattingly/Qwen3-VL-8B-hebrew-3-epochs") is None
    assert local_adapter_blocker("relative/path") is None, "the registry refuses those first"


def test_the_preflight_stops_there_and_reads_nothing_else(tmp_path: Path, monkeypatch):
    """Before #219 the same spec cost a base download and a base load. Now the
    one blocker comes back and neither config is even asked for."""
    spec = ModelSpec(id="qwen3vl-thun-smoke", engine="vllm", enabled=False,
                     local_path=str(tmp_path / "qwen3vl-thun-smoke"),
                     base_model="Qwen/Qwen3-VL-8B-Instruct")

    def never(*_a, **_k):
        raise AssertionError("the preflight read a config after the adapter was missing")

    monkeypatch.setattr(merge_loras, "_adapter_config", never)
    monkeypatch.setattr(merge_loras, "_base_config", never)
    monkeypatch.setattr(merge_loras, "_supported_archs", never)

    blockers = preflight(spec)

    assert len(blockers) == 1 and blockers[0].fatal
    assert isinstance(blockers[0], Blocker) and "does not exist" in blockers[0].text


# ── the peft hint ────────────────────────────────────────────────────────────

def test_the_peft_hint_no_longer_calls_idhefix_the_trainer():
    """Training moved to asteraix on 16.09.2026; on 14.10. idhefix's vlm-train had
    peft 0.20.0 against adapters written by 0.21.0, and the hint said the venv
    "trained it". The block was right, the advice was not."""
    b = peft_blocker("0.21.0", "0.20.0")
    assert b is not None and b.fatal
    assert "trained it" not in b.text
    assert "asteraix" in b.text and "vlm-train" in b.text
    assert "peft>=0.21.0" in b.text, "the pip line to run, not a paraphrase of it"
