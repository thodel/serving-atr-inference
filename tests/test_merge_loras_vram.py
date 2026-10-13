"""The merge records what it weighs (#216).

A VLM adapter is registered without ``vram_mb`` on purpose (training-atr-models#225):
what is served is base plus adapter, and only the merge on idhefix produces that.
Until #216 ``scripts/merge_loras.py`` printed DONE and dropped the number, so every
merged VLM launched on the 0.70 fallback and ``/models`` answered null for it.

No torch, no GPU: a "merged" directory here is a few files of known size.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from atr_serving.config import Settings
from atr_serving.registry import ModelSpec, Registry
from atr_serving.shared_registry import read_trained
from scripts import merge_loras
from scripts.merge_loras import record_vram_mb, weights_mib

MIB = 2**20
MODEL = "qwen3vl-test-v1"

HEADER = (
    "# One trained model (training-atr-models#14), read by the gateway from the\n"
    "# shared registry (serving-atr-inference#138). The file name is the id.\n"
    "# Written by the trainer on asteraix at 2026-10-12T09:00:00Z.\n"
)
REGISTRATION = {
    "id": MODEL, "engine": "vllm", "local_path": "/mnt/share/adapters/" + MODEL,
    "base_model": "Qwen/Qwen3-VL-8B-Instruct", "enabled": False, "task": "htr",
    "languages": ["de"], "centuries": [19],
}


def _merged(out_dir: Path, *sizes: int, suffix: str = ".safetensors") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, size in enumerate(sizes):
        (out_dir / f"model-{i:05d}-of-{len(sizes):05d}{suffix}").write_bytes(b"\0" * size)
    (out_dir / "config.json").write_text("{}", encoding="utf-8")
    (out_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    return out_dir


@pytest.fixture
def share(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    (root / "trained").mkdir(parents=True)
    (root / "trained" / f"{MODEL}.yaml").write_text(
        HEADER + yaml.safe_dump(REGISTRATION, sort_keys=False), encoding="utf-8")
    return root


def _registered(share: Path) -> ModelSpec:
    (spec,) = read_trained(share)
    return spec


# ── sizing ───────────────────────────────────────────────────────────────────

def test_the_weights_are_summed_and_rounded_up(tmp_path: Path):
    out = _merged(tmp_path / "m", 3 * MIB, MIB + 1)
    assert weights_mib(out) == 5          # 4 MiB and one byte -> 5, never 4


def test_bin_weights_count_and_the_index_and_config_do_not(tmp_path: Path):
    out = _merged(tmp_path / "m", 2 * MIB, suffix=".bin")
    (out / "config.json").write_text("{" + " " * MIB + "}", encoding="utf-8")
    assert weights_mib(out) == 2


def test_nothing_to_size_is_none_not_zero(tmp_path: Path):
    out = tmp_path / "m"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")
    assert weights_mib(out) is None
    assert weights_mib(tmp_path / "absent") is None


# ── recording ────────────────────────────────────────────────────────────────

def test_the_size_lands_in_the_registration_on_the_share(tmp_path: Path, share: Path):
    out = _merged(tmp_path / "m", 7 * MIB, MIB)
    line = record_vram_mb(_registered(share), out, share)

    assert "8 MiB" in line and "written" in line
    spec = _registered(share)
    assert spec.vram_mb == 8
    text = (share / "trained" / f"{MODEL}.yaml").read_text(encoding="utf-8")
    assert text.startswith(HEADER), "the trainer's header lines are kept"
    assert spec.model_dump(exclude={"vram_mb"}) == ModelSpec(**REGISTRATION).model_dump(
        exclude={"vram_mb"}), "nothing else in the registration changed"
    assert [p.name for p in (share / "trained").iterdir()] == [f"{MODEL}.yaml"], \
        "no tmp file is left behind"


def test_the_write_is_a_tmp_file_renamed_not_an_edit_in_place(tmp_path: Path, share: Path,
                                                             monkeypatch):
    out = _merged(tmp_path / "m", MIB)
    target = share / "trained" / f"{MODEL}.yaml"
    seen = []
    real = os.replace

    def spy(src, dst):
        seen.append((Path(src).name, Path(dst)))
        real(src, dst)

    monkeypatch.setattr(merge_loras.os, "replace", spy)
    record_vram_mb(_registered(share), out, share)
    assert len(seen) == 1
    name, dst = seen[0]
    assert dst == target
    assert name.startswith(".") and name.endswith(".tmp"), \
        "a dotted tmp name, which the gateway's listing ignores"


def test_a_measured_registration_is_left_alone(tmp_path: Path, share: Path):
    target = share / "trained" / f"{MODEL}.yaml"
    measured = dict(REGISTRATION, vram_mb=4, vram_measured={
        "total_mib": 9000, "weights_mib": 4, "host": "idhefix",
        "measured_at": "2026-10-01T10:00:00Z"})
    target.write_text(HEADER + yaml.safe_dump(measured, sort_keys=False), encoding="utf-8")
    before = target.read_bytes()

    out = _merged(tmp_path / "m", 9 * MIB)
    line = record_vram_mb(_registered(share), out, share)

    assert "measure_vram" in line
    assert target.read_bytes() == before


def test_a_larger_registered_value_stays_and_a_smaller_one_is_raised(tmp_path: Path,
                                                                     share: Path):
    target = share / "trained" / f"{MODEL}.yaml"
    out = _merged(tmp_path / "m", 5 * MIB)

    target.write_text(HEADER + yaml.safe_dump(dict(REGISTRATION, vram_mb=9), sort_keys=False),
                      encoding="utf-8")
    before = target.read_bytes()
    assert "stays" in record_vram_mb(_registered(share), out, share)
    assert target.read_bytes() == before

    target.write_text(HEADER + yaml.safe_dump(dict(REGISTRATION, vram_mb=2), sort_keys=False),
                      encoding="utf-8")
    line = record_vram_mb(_registered(share), out, share)
    assert "was 2" in line
    assert _registered(share).vram_mb == 5


def test_nothing_to_size_writes_nothing_and_says_so(tmp_path: Path, share: Path):
    target = share / "trained" / f"{MODEL}.yaml"
    before = target.read_bytes()
    out = tmp_path / "m"
    out.mkdir()

    line = record_vram_mb(_registered(share), out, share)

    assert "nothing to size" in line
    assert target.read_bytes() == before
    assert _registered(share).vram_mb is None


def test_a_registration_not_on_the_share_gets_the_line_to_add(tmp_path: Path):
    out = _merged(tmp_path / "m", 3 * MIB)
    curated = ModelSpec(id="qwen3vl-8b-hebrew", engine="vllm", hf_repo="x/y",
                        base_model="Qwen/Qwen3-VL-8B-Instruct")
    line = record_vram_mb(curated, out, None)
    assert "`vram_mb: 3`" in line and "config/models.yaml" in line

    # A share that has no registration of this id: the same answer, no file made.
    root = tmp_path / "registry"
    (root / "trained").mkdir(parents=True)
    line = record_vram_mb(curated, out, root)
    assert "`vram_mb: 3`" in line
    assert list((root / "trained").iterdir()) == []


def test_a_write_that_fails_is_said_not_raised(tmp_path: Path, share: Path, monkeypatch):
    out = _merged(tmp_path / "m", 3 * MIB)
    target = share / "trained" / f"{MODEL}.yaml"
    before = target.read_bytes()

    def refuse(src, dst):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(merge_loras.os, "replace", refuse)
    line = record_vram_mb(_registered(share), out, share)

    assert "could not write" in line and "PermissionError" in line
    assert "`vram_mb: 3`" in line and "by hand" in line
    assert target.read_bytes() == before
    assert [p.name for p in (share / "trained").iterdir()] == [f"{MODEL}.yaml"], \
        "the tmp file is removed after the failed rename"


def test_a_registration_named_for_another_id_is_not_rewritten(tmp_path: Path, share: Path):
    """The file name is the key (#138); a mismatch is the trainer's rule, kept here."""
    target = share / "trained" / f"{MODEL}.yaml"
    target.write_text(HEADER + yaml.safe_dump(dict(REGISTRATION, id="other"), sort_keys=False),
                      encoding="utf-8")
    before = target.read_bytes()
    out = _merged(tmp_path / "m", MIB)
    spec = ModelSpec(**REGISTRATION)

    line = record_vram_mb(spec, out, share)

    assert "could not write" in line and "must be named" in line
    assert target.read_bytes() == before


# ── through main ─────────────────────────────────────────────────────────────

def test_an_already_merged_model_gets_its_size_on_the_skip(tmp_path: Path, share: Path,
                                                            monkeypatch, capsys):
    """A model merged before #216 is only ever seen again as "already merged"."""
    merged_dir = tmp_path / "merged"
    out = _merged(merged_dir / MODEL, 6 * MIB)
    for name in ("tokenizer.json", "preprocessor_config.json"):
        (out / name).write_text("{}", encoding="utf-8")
    settings = Settings(api_key="k" * 32, registry_root=share, vllm_merged_dir=merged_dir)
    monkeypatch.setattr(merge_loras, "get_settings", lambda: settings)
    monkeypatch.setattr(merge_loras, "registered_models",
                        lambda _settings: Registry([_registered(share)]))
    monkeypatch.setattr(sys, "argv", ["merge_loras.py", "--only", MODEL])

    assert merge_loras.main() == 0

    printed = capsys.readouterr().out
    assert "already merged" in printed
    assert "6 MiB" in printed and "written" in printed
    assert _registered(share).vram_mb == 6
