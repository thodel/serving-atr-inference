"""The promotion gate (#36): advertise a trained model only once it has served.

Registering a model writes a row in a YAML file. That is not evidence of
anything — and confidently advertising models the host could not run is this
repo's most repeated failure (#21, #30, #31, #32). So a trained model is
registered ``enabled: false``, and only a **real transcription through the real
engine** flips it: the trainer posts one held-out page to the gateway's ``/ocr``
with the new model id, and non-empty text is the gate.

Three properties worth stating, because each is a decision:

* **The page comes from the run's own validation split.** Any page would prove
  the engine can load the weights, but a held-out page also exercises the
  material the model was scored on, and it is already on disk.
* **A failed gate does not fail the job.** The model trained, it scored, it is
  registered — it simply is not servable yet, which is a fact about the serving
  side. `docs/TRAINING_PLAN.md` §5 originally said a job completes only with a
  passing smoke recognition; that reads well until the VLM backend, whose
  adapters *cannot* be served until ``scripts/merge_loras.py`` bakes them in.
  Failing those jobs would call a good training run a failure. The job completes,
  ``promoted`` is false, the reason is on the record, and ``/models`` stays quiet.
* **Empty text is a failure, not a pass.** A 200 with ``""`` is exactly what #21
  was about: the gateway used to answer that way for a model it could not run.

The HTTP call is injectable, so the whole gate is testable without a gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from atr_serving.training.manifests import read_manifest

__all__ = ["PromotionResult", "Recognizer", "held_out_page", "promote", "http_recognizer"]


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    reason: str
    #: What came back, trimmed — enough to see *what* was transcribed, in the log
    #: and on the job record, without pasting a page of text into json.
    sample: str | None = None


class Recognizer(Protocol):
    """Posts one image to the gateway and returns the transcription."""

    def __call__(self, model_id: str, image: Path) -> str: ...


def held_out_page(data_dir: Path, manifest_name: str = "pages_val.lst") -> Path | None:
    """The first validation page's image, or None when there is nothing to send.

    The manifests list PageXML paths; the materialized image sits beside each one
    with the same stem (``prepare`` writes ``<stem>.xml`` and ``<stem>.jpg``).
    """
    manifest = Path(data_dir) / manifest_name
    if not manifest.exists():
        return None
    for entry in read_manifest(manifest):
        image = Path(entry).with_suffix(".jpg")
        if image.exists():
            return image
    return None


def promote(model_id: str, page: Path | None, recognize: Recognizer) -> PromotionResult:
    """Run the gate. Never raises: every outcome is a reportable result."""
    if page is None:
        return PromotionResult(False, "no held-out page was available to test with")
    try:
        text = recognize(model_id, page)
    except Exception as exc:  # noqa: BLE001 — a failed gate is a result, not a crash
        return PromotionResult(False, f"{type(exc).__name__}: {exc}")

    if not (text or "").strip():
        # #21: a 200 with empty text is precisely how the gateway used to answer
        # for a model it could not actually run.
        return PromotionResult(False, f"the engine returned no text for {page.name}")
    return PromotionResult(True, f"transcribed {page.name} through the gateway",
                           sample=text.strip()[:200])


def http_recognizer(gateway_url: str, api_key: str, timeout: float = 120.0) -> Recognizer:
    """A :class:`Recognizer` that posts to the gateway's ``/ocr``.

    Deliberately goes through the gateway rather than straight to the engine: the
    question the gate asks is "can this box *serve* it", and the gateway is what
    clients talk to. A model that only works when addressed directly is not
    promoted, because that is not how anyone will call it.
    """

    def recognize(model_id: str, image: Path) -> str:
        import httpx  # trainer venv only

        with image.open("rb") as fh:
            response = httpx.post(
                f"{gateway_url.rstrip('/')}/ocr",
                headers={"X-API-Key": api_key},
                files={"image": (image.name, fh, "image/jpeg")},
                data={"model": model_id},
                timeout=timeout,
            )
        response.raise_for_status()
        return str(response.json().get("text") or "")

    return recognize
