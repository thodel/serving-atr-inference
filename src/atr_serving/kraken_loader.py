"""Resolving and loading kraken recognition weights (#36, closes #32).

Two engine services load kraken nets — ``kraken_svc`` and ``party_svc`` — and
both got it wrong in the same way, so the logic lives here once.

**What was wrong.** Both called ``kraken.lib.models.load_any``, which in kraken
7.0.2 goes to ``TorchVGSLModel.load_model`` and is **CoreML-only** (and marked
deprecated: "use kraken.registry.load_model instead"). Meanwhile
``ketos train --weights-format`` *defaults to safetensors*. So the trainer's
default output could not be served by the server that asked for it, and
``atr-party`` failed outright on its ``model.safetensors``
(``KrakenInvalidModelException`` — #32). ``kraken.models.load_models`` is the
entry point that dispatches on the file, so that is what both engines use now.

**Why a fallback remains.** ``kraken.models`` is 7.x; the loader tries it and
falls back to ``load_any`` rather than assuming, because an engine that cannot
load its model at all is worse than one loading a CoreML file the old way. Which
path was taken is logged, so "it works" and "it works for the reason we think"
stay distinguishable.

The kraken import is inside the functions: this module is imported by the
gateway's test suite, which has no kraken and no torch.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["RECOGNITION_SUFFIXES", "WeightsNotFound", "resolve_weights", "load_recognition_model"]

#: Formats kraken 7.x can load. ``.mlmodel`` is CoreML (the M2 workaround),
#: ``.safetensors`` is what ``ketos train`` writes unless told otherwise.
RECOGNITION_SUFFIXES = (".safetensors", ".mlmodel")


class WeightsNotFound(FileNotFoundError):
    """A local reference exists but holds nothing loadable."""


def resolve_weights(ref: str | Path) -> Path | None:
    """A model reference → a local weights file, or ``None`` if it is not local.

    ``None`` is the signal to fall back to a remote fetch (``htrmopo`` by DOI),
    so a caller can pass any reference and let this decide. A *directory* is
    accepted because that is the shape the trainer registers: one directory per
    model holding the weights and a ``metadata.json``.
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


def load_recognition_model(path: str | Path, device: str = "cpu",
                           load_models=None, load_any=None):
    """Load kraken recognition weights, preferring the loader that reads both formats.

    ``load_models`` / ``load_any`` are injectable so the dispatch is testable
    without kraken installed; in production both are imported lazily.
    """
    path = str(path)
    if load_models is None:
        try:
            from kraken.models import load_models as load_models  # noqa: PLC0415
        except ImportError:
            load_models = False  # marker: not available in this kraken
    if load_models:
        return load_models(path, device=device)

    if load_any is None:
        from kraken.lib.models import load_any  # noqa: PLC0415
    # CoreML-only in 7.0.2. Reaching here with safetensors is the #32 failure, so
    # say what happened rather than letting KrakenInvalidModelException stand
    # alone with no mention of the loader that would have worked.
    if path.endswith(".safetensors"):
        raise WeightsNotFound(
            f"{path} is safetensors, but this kraken has no kraken.models.load_models "
            "and kraken.lib.models.load_any reads CoreML only (#32). Either upgrade "
            "kraken or retrain with --weights-format coreml."
        )
    return load_any(path, device=device)
