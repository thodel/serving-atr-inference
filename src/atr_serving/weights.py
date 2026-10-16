"""What the weights on this box actually are, as a number (#200).

``/models`` hands out every field of a ``ModelSpec``, and until now none of them
named a *version* of the weights:

===============  =================================================
``zenodo_id``    carries a version — Zenodo mints a record per version
``hf_repo``      does not — a repo name whose ``main`` moves
``local_path``   does not — a path
===============  =================================================

Which means that for **every model this project trained itself**, a published
reading could say nothing checkable about which weights produced it: those are
registered by path, and they are exactly the ones whose contents change most
often — every sweep, every rung, every finetune. agentic_historian#482 asks the
gateway for a revision and gets ``revision_kind: unversioned_path``.

A digest is the version those weights can have. Not a name somebody assigned:
the bytes the gateway loads, hashed.

What is hashed, and why that is a decision
------------------------------------------
A single file is itself. For a directory the rule follows the **loader**: if it
holds a CoreML file, that is what ``kraken_loader.resolve_weights`` serves and
the digest covers that file alone — hashing a ``.safetensors`` beside it would
version a file nobody can serve in kraken 7.0.2. Otherwise every weight-bearing
file is hashed, which is what a sharded VLM checkpoint is.

The combined digest is taken over the per-file lines ``<name> <sha256> <size>``,
sorted by name — not over the concatenated bytes. So a **re-shard changes the
digest** even when the parameters are identical. That is deliberate and it is
the conservative direction: a re-written checkpoint should read as a different
checkpoint until somebody proves otherwise, and #200 exists because two runs
over one page were indistinguishable after a finetune was rewritten.

A digest in a file that nobody measured
---------------------------------------
...is worse than no digest, because it reads like a measurement. Two things stop
that: :class:`WeightsDigest` requires its provenance (``host``, ``measured_at``)
like :class:`~atr_serving.registry.VramMeasurement` does, and it **recomputes
the combined digest from its own file list** on load. A hand-written value that
does not follow from the files it claims to describe does not validate, so the
registry refuses it instead of serving it.
"""

from __future__ import annotations

import hashlib
import socket
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "WEIGHT_SUFFIXES",
    "COREML_SUFFIX",
    "WeightFile",
    "WeightsDigest",
    "WeightsUnreadable",
    "combined_digest",
    "digest_weights",
    "weight_files",
]

#: Files whose bytes are the model. ``.mlmodel`` is kraken's CoreML, the rest is
#: what a VLM or TrOCR checkpoint is written as.
WEIGHT_SUFFIXES = (".mlmodel", ".safetensors", ".bin", ".pt", ".pth", ".ckpt")

#: Preferred alone when a directory holds it — see the module note on the loader.
COREML_SUFFIX = ".mlmodel"

#: Read in blocks rather than whole: a merged 27B checkpoint is tens of GB, and
#: a measurement that needs the file in memory is one nobody runs.
_BLOCK = 1024 * 1024


class WeightsUnreadable(FileNotFoundError):
    """The reference names no weights this host can hash."""


class WeightFile(BaseModel):
    """One hashed file, named relative to the model's directory.

    Relative, so the digest of the same checkpoint is the same on the trainer
    and on the gateway — the share is mounted at different paths on each, and an
    absolute name would make the two disagree about identical bytes.
    """

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class WeightsDigest(BaseModel):
    """The weights' identity, with what it was taken over and by whom.

    ``sha256`` is the value a caller cites. ``files`` is what makes it checkable
    without the weights at hand, and the validator below uses it: a digest that
    does not follow from its own file list never loads.
    """

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: Every file hashed, in the order they were hashed (sorted by name).
    files: list[WeightFile]
    #: Where and when. A digest is about bytes on one machine at one time; after
    #: a re-merge it describes a checkpoint that is no longer there.
    host: str
    measured_at: str
    #: What the caller resolved — the path or directory as configured. Kept for
    #: the person reading the record, not for comparison: it differs between the
    #: trainer and the gateway for the same weights.
    source: str | None = None
    #: Free text: a re-merge, a conversion, anything that explains why a digest
    #: changed without the model changing name.
    note: str | None = None

    @model_validator(mode="after")
    def _digest_follows_from_its_files(self) -> "WeightsDigest":
        if not self.files:
            raise ValueError(
                "weights_digest lists no files: a digest over nothing is not a "
                "measurement of anything"
            )
        expected = combined_digest(self.files)
        if self.sha256 != expected:
            raise ValueError(
                f"weights_digest {self.sha256[:12]}… does not follow from the "
                f"{len(self.files)} file(s) it lists (they give {expected[:12]}…). "
                "A digest nobody measured reads like one somebody did, which is "
                "worse than none: re-measure with scripts/measure_weights.py."
            )
        return self

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.files)

    def short(self) -> str:
        """The form a log line or a provenance string carries."""
        return self.sha256[:12]


def combined_digest(files: list[WeightFile]) -> str:
    """The digest over *files* — name, hash and size, one line each, sorted.

    Over the lines rather than the bytes, so the answer does not depend on the
    order the files were read in, and so a rename or a re-shard is visible. See
    the module note: that is the conservative direction on purpose.
    """
    lines = sorted(f"{entry.name} {entry.sha256} {entry.size_bytes}" for entry in files)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def weight_files(path: str | Path) -> list[Path]:
    """The files that *are* this model, in the order they will be hashed.

    Raises :class:`WeightsUnreadable` rather than returning an empty list: "no
    weights here" is a finding, and a digest over zero files would be a valid
    hash of nothing.
    """
    root = Path(path).expanduser()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise WeightsUnreadable(
            f"{root} does not exist on {socket.gethostname()}: a digest is a "
            "measurement on the file this host loads, and there is none here"
        )
    # CoreML alone when present — the loader serves that and nothing else, so a
    # digest covering its neighbours would version files nobody can run.
    coreml = sorted(p for p in root.rglob(f"*{COREML_SUFFIX}") if p.is_file())
    if coreml:
        return coreml
    found = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix in WEIGHT_SUFFIXES)
    if not found:
        raise WeightsUnreadable(
            f"{root} exists but holds no {', '.join(WEIGHT_SUFFIXES)} file — "
            "nothing here is weights"
        )
    return found


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def digest_weights(path: str | Path, *, note: str | None = None,
                   host: str | None = None, now: datetime | None = None) -> WeightsDigest:
    """Hash the weights at *path* and say what was hashed, where and when.

    The one place a :class:`WeightsDigest` is produced. Everything else reads
    one: a value that appears in a registry file without having come from here
    cannot validate, which is the point of the check in
    :meth:`WeightsDigest._digest_follows_from_its_files`.
    """
    paths = weight_files(path)
    root = Path(path).expanduser()
    base = root if root.is_dir() else root.parent
    entries = [
        WeightFile(name=str(each.relative_to(base)), sha256=_file_digest(each),
                   size_bytes=each.stat().st_size)
        for each in paths
    ]
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    return WeightsDigest(
        sha256=combined_digest(entries), files=entries,
        host=host or socket.gethostname(), measured_at=stamp,
        source=str(root), note=note,
    )
