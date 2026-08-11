"""Resolving and loading kraken recognition weights (#36, #32).

Two engine services load kraken nets — ``kraken_svc`` and ``party_svc`` — so the
logic lives here once.

**What `rpred` requires, measured on the box (kraken 7.0.2, 2026-08-10):**

    rpred.rpred(network: 'TorchSeqRecognizer', im, bounds, ...)

    kraken.lib.models.load_any(path, device=...) -> TorchSeqRecognizer   ✔ usable
    kraken.models.load_models(path)             -> list[BaseModel]       ✘ not

``BaseModel`` has no ``.to``, no ``.eval``, no ``predict``: it is a
serialization/registry layer, not a runnable recognizer, and ``rpred`` will not
take one. #36 was written on the premise that ``kraken.models.load_models`` is
"the working entry point" for serving; that is true for *deserializing* a file
and false for *recognizing* with it. Loading through it broke both engines —
party with ``TypeError: load_models() got an unexpected keyword argument
'device'``, and kraken silently, because it loads lazily and nothing had called
``/recognize`` yet.

So recognition goes through ``load_any``, as it did before.

**What that leaves unsolved, honestly.** ``load_any`` is CoreML-only in 7.0.2, so
a ``.safetensors`` recognition model still cannot be served this way. That is
survivable: every kraken model in ``config/models.yaml`` is a Zenodo ``.mlmodel``,
and the trainer defaults to ``weights_format: coreml`` for exactly this reason.
It is a real limit, and the error below names it rather than letting a
``KrakenInvalidModelException`` stand alone.

**#32 is a different problem than it looked.** ``atr-party`` fails on its
``model.safetensors``, but not because the format cannot be read —
``load_models`` parses it fine and then reports ``PartyModel is not in model
registry``. That class ships with the standalone ``party`` package, which
``engines/party_svc/requirements.txt`` does not install. A loader change was never
going to fix it.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["RECOGNITION_SUFFIXES", "WeightsNotFound", "resolve_weights", "load_recognition_model"]

#: Formats a caller might hand us. ``.mlmodel`` is CoreML — the only one
#: ``load_any`` can actually serve in 7.0.2; ``.safetensors`` is what
#: ``ketos train`` writes unless told otherwise, and is accepted here only so the
#: error can say something useful about it.
RECOGNITION_SUFFIXES = (".mlmodel", ".safetensors")


class WeightsNotFound(FileNotFoundError):
    """A local reference exists but holds nothing this kraken can serve."""


def resolve_weights(ref: str | Path) -> Path | None:
    """A model reference → a local weights file, or ``None`` if it is not local.

    ``None`` is the signal to fall back to a remote fetch (``htrmopo`` by DOI),
    so a caller can pass any reference and let this decide. A *directory* is
    accepted because that is the shape the trainer registers: one directory per
    model holding the weights and a ``metadata.json``.

    CoreML is preferred over safetensors when a directory holds both, because
    CoreML is the one that can be served (see the module docstring).
    """
    if not ref:
        return None
    path = Path(str(ref)).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        return None  # not a local reference at all — probably a DOI or a hub id
    for suffix in RECOGNITION_SUFFIXES:
        found = sorted(path.glob(f"*{suffix}"))
        if found:
            return found[0]
    raise WeightsNotFound(
        f"{path} exists but holds no {' or '.join(RECOGNITION_SUFFIXES)} file. A "
        "registered model directory should contain the weights the register stage "
        "copied there."
    )


def load_recognition_model(path: str | Path, device: str = "cpu", load_any=None):
    """Load kraken recognition weights into something ``rpred`` accepts.

    ``load_any`` is injectable so the dispatch is testable without kraken.
    """
    path = str(path)
    if path.endswith(".safetensors"):
        # Better than the bare KrakenInvalidModelException this used to raise:
        # that named neither the format nor the way out.
        raise WeightsNotFound(
            f"{path} is safetensors, which kraken 7.0.2 cannot serve: rpred needs a "
            "TorchSeqRecognizer, and only kraken.lib.models.load_any produces one — "
            "CoreML only. For a model this box trained: retrain with "
            "--weights-format coreml (the trainer's default), or convert the "
            "checkpoint with `ketos convert`. For a third-party model (party's "
            "Zenodo release), neither applies — its class has to be registered by "
            "the package that defines it, which is #32."
        )
    if load_any is None:
        from kraken.lib.models import load_any  # noqa: PLC0415 — engine venvs only
    return load_any(path, device=device)
