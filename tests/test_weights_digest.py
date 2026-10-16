"""#200: the weights get an identity, and an unmeasured one cannot pose as it.

`/models` hands out every `ModelSpec` field and none of them named a version of
the weights. `zenodo_id` carries one; `hf_repo` is a name whose `main` moves and
`local_path` is a path. So for every model this project trained itself — the
ones rewritten by every sweep and every finetune — a published reading could say
nothing checkable about the weights behind it.

Two halves are tested here, and the second is the one that keeps the first
worth having:

* the digest is a **measurement**: over the files this host loads, reproducible
  across machines, and changed by anything that rewrites the checkpoint;
* a digest that nobody measured **does not load**. It would read exactly like
  one somebody did, and `#200` says plainly that this is worse than its absence.
  So `WeightsDigest` recomputes itself from its own file list and refuses a
  value that does not follow, and it requires its provenance.

Offline throughout: files in `tmp_path`, hashed for real. No model, no card, no
network — the measurement is file bytes, which is the whole point.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from atr_serving.kraken_loader import resolve_weights
from atr_serving.registry import ModelSpec
from atr_serving.weights import (
    WeightFile,
    WeightsDigest,
    WeightsUnreadable,
    combined_digest,
    digest_weights,
    weight_files,
)

#: The pinned vector below lives in both repos: the trainer measures at its
#: promotion gate and cannot import this package (its `test_isolation.py`
#: forbids it), so the two implementations are held to one answer the way the
#: trainer contract fixtures are (#13, #250).
CONTRACT = Path(__file__).parent / "fixtures" / "weights_digest" / "contract.json"


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ── the measurement ─────────────────────────────────────────────────────────

def test_a_single_file_is_hashed_by_its_basename(tmp_path):
    """Relative, not absolute: the share is mounted at different paths on the
    trainer and on the gateway, and identical bytes must give one answer."""
    weights = _write(tmp_path / "model_best.mlmodel", b"kraken weights")

    digest = digest_weights(weights)

    assert [f.name for f in digest.files] == ["model_best.mlmodel"]
    assert digest.files[0].size_bytes == len(b"kraken weights")
    assert digest.total_bytes == len(b"kraken weights")


def test_the_same_bytes_under_another_mountpoint_give_the_same_digest(tmp_path):
    """The case that makes relative names load-bearing — asteraix and idhefix
    mount the research share at different paths."""
    here = _write(tmp_path / "asteraix" / "trained" / "m" / "w.safetensors", b"same")
    there = _write(tmp_path / "idhefix" / "mnt" / "m" / "w.safetensors", b"same")

    assert digest_weights(here).sha256 == digest_weights(there).sha256


def test_one_changed_byte_changes_the_digest(tmp_path):
    """The finetune rewritten in place, which is the incident in #200."""
    weights = _write(tmp_path / "m" / "model.safetensors", b"epoch 11")
    before = digest_weights(weights).sha256

    _write(tmp_path / "m" / "model.safetensors", b"epoch 12")

    assert digest_weights(weights).sha256 != before


def test_every_shard_of_a_vlm_checkpoint_is_hashed(tmp_path):
    directory = tmp_path / "qwen3vl-medieval-v3"
    for index in range(3):
        _write(directory / f"model-0000{index}-of-00003.safetensors", f"shard{index}".encode())
    _write(directory / "config.json", b'{"irrelevant": true}')

    digest = digest_weights(directory)

    assert len(digest.files) == 3, "the config is not weights"
    assert all(f.name.endswith(".safetensors") for f in digest.files)


def test_the_answer_does_not_depend_on_the_order_the_files_came_in():
    """Over sorted lines, not over concatenated bytes: a filesystem that lists a
    directory in another order must not produce another version."""
    a = WeightFile(name="a.safetensors", sha256="a" * 64, size_bytes=1)
    b = WeightFile(name="b.safetensors", sha256="b" * 64, size_bytes=2)

    assert combined_digest([a, b]) == combined_digest([b, a])


def test_a_reshard_changes_the_digest(tmp_path):
    """Deliberate, and the conservative direction: the same parameters written
    as two shards instead of one is a rewritten checkpoint until somebody proves
    otherwise — and #200 exists because two runs were indistinguishable after a
    rewrite."""
    one = tmp_path / "one"
    _write(one / "model.safetensors", b"AAAABBBB")
    two = tmp_path / "two"
    _write(two / "model-00001-of-00002.safetensors", b"AAAA")
    _write(two / "model-00002-of-00002.safetensors", b"BBBB")

    assert digest_weights(one).sha256 != digest_weights(two).sha256


def test_a_rename_changes_the_digest(tmp_path):
    """The names are in the hashed lines. A checkpoint served under a different
    filename is a different thing to load, even byte for byte."""
    first = tmp_path / "a"
    _write(first / "model.safetensors", b"same bytes")
    second = tmp_path / "b"
    _write(second / "weights.safetensors", b"same bytes")

    assert digest_weights(first).sha256 != digest_weights(second).sha256


def test_the_digest_covers_the_file_the_loader_would_serve(tmp_path):
    """The coupling, enforced rather than duplicated.

    `kraken_loader.resolve_weights` prefers CoreML when a directory holds both,
    because `load_any` cannot serve `.safetensors` in kraken 7.0.2. A digest
    covering the neighbour would version a file nobody can run.
    """
    directory = tmp_path / "kraken-thun-missiven-v1"
    _write(directory / "best.mlmodel", b"coreml")
    _write(directory / "best.safetensors", b"torch")

    digest = digest_weights(directory)
    served = resolve_weights(directory)

    assert [f.name for f in digest.files] == [served.name] == ["best.mlmodel"]


def test_the_provenance_travels_with_the_number(tmp_path):
    """A digest is about bytes on one machine at one time; after a re-merge it
    describes a checkpoint that is no longer there."""
    weights = _write(tmp_path / "m.mlmodel", b"x")

    digest = digest_weights(weights, note="after the 2026-10-16 re-merge")

    assert digest.host and digest.measured_at
    assert digest.note == "after the 2026-10-16 re-merge"
    assert digest.source == str(weights)


# ── the refusals ────────────────────────────────────────────────────────────

def test_a_path_that_is_not_there_is_refused_not_hashed(tmp_path):
    """`sha256("")` is a perfectly good hash of nothing, and it would travel as
    a version. The absence is a finding instead."""
    with pytest.raises(WeightsUnreadable) as exc:
        digest_weights(tmp_path / "never-trained")
    assert "measurement" in str(exc.value)


def test_a_directory_without_weights_is_refused(tmp_path):
    directory = tmp_path / "m"
    _write(directory / "metadata.json", b"{}")
    _write(directory / "train.log", b"...")

    with pytest.raises(WeightsUnreadable):
        weight_files(directory)


def test_a_digest_that_does_not_follow_from_its_files_refuses_to_load():
    """The mechanism behind "worse than its absence".

    A hand-written value in a registry file reads exactly like a measured one.
    This one cannot: the model recomputes the combined digest from the file list
    it carries, so the only values that load are the ones that follow from it.
    """
    files = [WeightFile(name="w.safetensors", sha256="c" * 64, size_bytes=4)]

    with pytest.raises(ValidationError) as exc:
        WeightsDigest(sha256="d" * 64, files=files, host="asteraix",
                      measured_at="2026-10-16T08:00:00+00:00")

    assert "does not follow" in str(exc.value)
    assert "measure_weights" in str(exc.value), "the refusal says how to fix it"


def test_a_digest_over_no_files_refuses_to_load():
    with pytest.raises(ValidationError, match="not a measurement"):
        WeightsDigest(sha256="e" * 64, files=[], host="h", measured_at="t")


@pytest.mark.parametrize("missing", ["host", "measured_at"])
def test_a_digest_without_its_provenance_refuses_to_load(missing):
    """Same rule as `vram_measured` (#130): a number whose circumstances are
    unknown is not a measurement, it is a value."""
    files = [WeightFile(name="w.safetensors", sha256="c" * 64, size_bytes=4)]
    payload = {"sha256": combined_digest(files), "files": files,
               "host": "asteraix", "measured_at": "2026-10-16T08:00:00+00:00"}
    payload.pop(missing)

    with pytest.raises(ValidationError):
        WeightsDigest(**payload)


def test_a_model_spec_rejects_a_dishonest_digest(tmp_path):
    """End to end through the registry, because that is the file a person edits."""
    with pytest.raises(ValidationError, match="does not follow"):
        ModelSpec(id="m", engine="kraken", local_path=str(tmp_path / "m.mlmodel"),
                  weights_digest={"sha256": "f" * 64,
                                  "files": [{"name": "w.mlmodel", "sha256": "a" * 64,
                                             "size_bytes": 1}],
                                  "host": "h", "measured_at": "t"})


# ── what /models says ───────────────────────────────────────────────────────

def test_models_carries_the_digest_and_defaults_to_absent(tmp_path):
    """Absent means nobody measured these weights — not that they have no
    version, and not a digest over an empty file."""
    from atr_serving.api.schemas import ModelInfo

    assert "weights_digest" in ModelInfo.model_fields
    unmeasured = ModelSpec(id="m", engine="kraken", local_path=str(tmp_path / "m.mlmodel"))
    assert unmeasured.weights_digest is None
    assert ModelInfo(**unmeasured.model_dump(), resident=False).weights_digest is None


def test_a_measured_model_reaches_a_caller_with_the_number(tmp_path):
    from atr_serving.api.schemas import ModelInfo

    weights = _write(tmp_path / "m.mlmodel", b"weights")
    spec = ModelSpec(id="m", engine="kraken", local_path=str(weights),
                     weights_digest=digest_weights(weights))

    info = ModelInfo(**spec.model_dump(), resident=True)

    assert info.weights_digest.sha256 == spec.weights_digest.sha256
    assert len(info.weights_digest.sha256) == 64


# ── the contract the other half is held to ──────────────────────────────────

def test_the_pinned_vector_still_produces_the_pinned_digest(tmp_path):
    """One known file set → one known digest, in a fixture both repos hold.

    The measurement belongs at the trainer's promotion gate, which has the
    weights path at the moment it proves the model servable — and that repo
    cannot import this package (`test_isolation.py` there forbids it). So the
    two implementations are pinned to one answer, the way `/host` and `/gpu` are
    (#13, #250). If this test fails, the digest's definition changed and every
    recorded value in both repos now means something else.
    """
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    directory = tmp_path / "pinned"
    for name, content in contract["files"].items():
        _write(directory / name, content.encode("utf-8"))

    digest = digest_weights(directory, host="contract", now=None)

    assert digest.sha256 == contract["sha256"], (
        "the definition of the digest moved — see the fixture's own note")
    assert [f.name for f in digest.files] == contract["hashed"]
