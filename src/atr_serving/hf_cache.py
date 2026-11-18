"""Which commit of a Hugging Face repo a vLLM model was loaded from (#200).

``hf_repo`` names a repository, and a repository's ``main`` moves. A reading
cited as "repo X" says nothing checkable about which weights produced it. The
commit does, and it is on this box: vLLM is started by the gateway, as the same
user and with the gateway's environment, and it loads through the Hugging Face
hub cache. That cache records what a ref resolved to as a plain file:

    <hub cache>/models--<org>--<name>/refs/<ref>        → the commit sha
    <hub cache>/models--<org>--<name>/snapshots/<sha>/  → the files loaded

So the revision is read there, with no new dependency and no network call
(decided 18.11.2026 on #200, over adding ``huggingface_hub`` to the gateway or
having the engines report it). Read **right after a launch became healthy**, it
is the commit vLLM resolved and loaded — ``snapshot_download`` writes the ref
before vLLM reads the snapshot. Read for a model that is not resident, it is
only what the cache would load now, and :attr:`HfRevision.loaded` says which of
the two it is.

Nothing is guessed. A ref file that is missing, that does not hold a 40-digit
sha, or whose snapshot is not on disk gives ``None``, never the configured ref
dressed up as a commit. And the field is runtime state, not registry config: a
hand-written commit in ``models.yaml`` would read like a measurement nobody made.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

__all__ = [
    "HfRevision",
    "hub_cache_dir",
    "read_revision",
    "requested_ref",
]

_SHA = re.compile(r"[0-9a-f]{40}")


class HfRevision(BaseModel):
    """The commit of ``hf_repo`` that the hub cache resolved ``ref`` to."""

    sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    #: The ref that was asked for: ``main`` unless the model's ``vllm_args``
    #: name a ``--revision``.
    ref: str
    #: True: recorded when this gateway launched the model, so it is the commit
    #: the running instance loaded. False: the cache's current answer for a model
    #: that is not resident — what a launch now would load if vLLM finds no newer
    #: commit on the hub.
    loaded: bool
    read_at: str


def hub_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    """The hub cache a process with ``env`` uses, by huggingface_hub's own order:
    ``HF_HUB_CACHE``, the older ``HUGGINGFACE_HUB_CACHE``, ``HF_HOME/hub``, then
    ``XDG_CACHE_HOME`` or ``~/.cache``, plus ``huggingface/hub``."""
    env = os.environ if env is None else env
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(key):
            return Path(env[key]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    base = env.get("XDG_CACHE_HOME") or str(Path(env.get("HOME") or Path.home()) / ".cache")
    return Path(base).expanduser() / "huggingface" / "hub"


def requested_ref(vllm_args: Sequence[str]) -> str:
    """The ref vLLM is asked to load: ``--revision X`` or ``--revision=X``, else ``main``."""
    args = list(vllm_args or [])
    for i, arg in enumerate(args):
        if arg == "--revision" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--revision="):
            return arg.split("=", 1)[1]
    return "main"


def read_revision(repo: str, ref: str, *, loaded: bool, cache: Path | None = None,
                  env: Mapping[str, str] | None = None) -> HfRevision | None:
    """What the cache resolved ``ref`` of ``repo`` to, or None when it cannot say.

    A ``ref`` that is itself a commit sha is its own answer, provided its
    snapshot is on disk. Anything unreadable is None — never an exception on the
    ``/models`` path, and never a sha that is not one.
    """
    if not repo or "/" not in repo or not ref:
        return None
    root = (cache if cache is not None else hub_cache_dir(env)) / (
        "models--" + repo.replace("/", "--"))
    try:
        if _SHA.fullmatch(ref):
            sha = ref
        else:
            if "/" in ref or ref in (".", ".."):
                return None              # a ref is a file name under refs/, not a path
            sha = (root / "refs" / ref).read_text(encoding="utf-8").strip()
        if not _SHA.fullmatch(sha) or not (root / "snapshots" / sha).is_dir():
            return None
    except OSError:
        return None
    return HfRevision(sha=sha, ref=ref, loaded=loaded,
                      read_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
