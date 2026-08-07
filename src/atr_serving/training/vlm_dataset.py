"""Materialized pages → VLM training samples. Pure: stdlib + the contracts.

The VLM backend's ``compile`` stage is the analogue of ``ketos compile``. Where
kraken turns ``pages/*.{jpg,xml}`` into an ``.arrow`` dataset of normalized line
crops, this turns the same pages into a **JSONL of samples**, one object per
line:

.. code-block:: json

    {"image": "pages/000012_x.jpg", "text": "Item ontfaen van Janne",
     "source_type": "line", "bbox": [10, 24, 812, 96], "page": "000012_x.xml"}

Everything here is decisions — which lines become samples, what text they carry,
which crop rectangle — and none of it is pixels. The cropping itself needs PIL
and lives in the engine (``vlm_train_svc.runner``), so this module is importable
and unit-testable in the repo venv, the same rule the rest of
:mod:`atr_serving.training` follows.

``source_type`` travels with each sample because the collator budgets visual
tokens by it (:data:`~atr_serving.training.contracts.VLM_PIXEL_BUDGET`) — the
same field, spelled the same way, as in ``lassberg/vlm_training``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from atr_serving.training.contracts import VLM_PIXEL_BUDGET
from atr_serving.training.pagexml import line_boxes, line_texts

__all__ = [
    "VlmDatasetError",
    "Sample",
    "DEFAULT_LINE_PAD",
    "MIN_CROP_PX",
    "MIN_TEXT_LEN",
    "page_sample",
    "line_samples",
    "samples_for",
    "write_jsonl",
    "read_jsonl",
    "chat_example",
]


class VlmDatasetError(ValueError):
    """Raised when a page cannot produce a usable training sample."""


#: Padding added around a line polygon before cropping (see ``TextLineBox.padded``).
DEFAULT_LINE_PAD = 8
#: A crop smaller than this in either dimension carries no legible glyph — the
#: page's coordinates are wrong, or the polygon is a stray click. Dropped rather
#: than trained on.
MIN_CROP_PX = 8
#: Shorter transcriptions are usually a stray mark or an editorial dash. Kept low
#: because single-character lines (a folio number, an ``&``) are real.
MIN_TEXT_LEN = 1


@dataclass(frozen=True)
class Sample:
    """One training example: an image (or a crop of one) and its transcription."""

    image: str
    text: str
    source_type: str
    #: ``[left, top, right, bottom]`` to crop from ``image``; None = the whole file.
    bbox: list[int] | None = None
    #: The PageXML this came from — kept so a sample is traceable to its page, and
    #: so the train/val split can be verified to be page-disjoint after the fact.
    page: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: dict) -> "Sample":
        try:
            return cls(
                image=raw["image"],
                text=raw["text"],
                source_type=raw.get("source_type", "line"),
                bbox=list(raw["bbox"]) if raw.get("bbox") else None,
                page=raw.get("page"),
            )
        except KeyError as exc:
            raise VlmDatasetError(f"sample is missing {exc.args[0]!r}: {raw!r}") from None


def _image_for(xml_path: Path) -> Path:
    """The JPEG written beside this PageXML by the prepare stage.

    prepare writes ``<stem>.jpg`` + ``<stem>.xml`` as siblings and rewrites
    ``@imageFilename`` accordingly, so the sibling is the authority — reading the
    attribute back would only re-derive what we just wrote.
    """
    image = xml_path.with_suffix(".jpg")
    if not image.exists():
        raise VlmDatasetError(
            f"{xml_path.name} has no sibling {image.name}; the prepare stage writes "
            "the two together, so one without the other means the page directory "
            "was modified after prepare ran"
        )
    return image


def page_sample(xml_path: str | Path, root: str | Path | None = None) -> Sample | None:
    """One sample per page: the whole scan, and every line joined by newlines.

    Returns None for a page with no transcription — the same rule prepare applies,
    re-checked here because ``compile`` may run over a page directory prepare did
    not write (a re-run, a hand-assembled set).
    """
    xml_path = Path(xml_path)
    text = "\n".join(t.strip() for t in line_texts(xml_path.read_text(encoding="utf-8")) if t.strip())
    if len(text) < MIN_TEXT_LEN:
        return None
    image = _image_for(xml_path)
    return Sample(
        image=_relative(image, root),
        text=text,
        source_type="page",
        page=_relative(xml_path, root),
    )


def line_samples(
    xml_path: str | Path,
    root: str | Path | None = None,
    pad: int = DEFAULT_LINE_PAD,
    page_size: tuple[int, int] | None = None,
) -> list[Sample]:
    """One sample per transcribed ``TextLine``, with the box to crop.

    ``page_size`` (width, height) clamps the padded box to the page. It is
    optional because reading it costs an image open; the runner passes it, tests
    do not. Without it a box may extend past the edge, which PIL handles by
    padding with black — survivable, but a real crop is better.
    """
    xml_path = Path(xml_path)
    image = _image_for(xml_path)
    width, height = page_size if page_size else (None, None)

    out: list[Sample] = []
    for box in line_boxes(xml_path.read_text(encoding="utf-8")):
        padded = box.padded(pad, width, height)
        if padded.width < MIN_CROP_PX or padded.height < MIN_CROP_PX:
            continue
        if len(box.text) < MIN_TEXT_LEN:
            continue
        out.append(Sample(
            image=_relative(image, root),
            text=box.text,
            source_type="line",
            bbox=[padded.left, padded.top, padded.right, padded.bottom],
            page=_relative(xml_path, root),
        ))
    return out


def samples_for(
    xml_paths: Iterable[str | Path],
    granularity: str,
    root: str | Path | None = None,
    pad: int = DEFAULT_LINE_PAD,
    page_sizes: dict[str, tuple[int, int]] | None = None,
) -> list[Sample]:
    """Build every sample for a set of pages at the requested granularity.

    ``page_sizes`` maps an image path (as written into the sample) to its
    ``(width, height)``; missing entries simply skip the clamp.
    """
    if granularity not in VLM_PIXEL_BUDGET:
        raise VlmDatasetError(
            f"granularity {granularity!r} is not one of {sorted(VLM_PIXEL_BUDGET)}"
        )
    out: list[Sample] = []
    for xml_path in xml_paths:
        if granularity == "page":
            sample = page_sample(xml_path, root)
            if sample is not None:
                out.append(sample)
        else:
            size = None
            if page_sizes:
                size = page_sizes.get(_relative(Path(xml_path).with_suffix(".jpg"), root))
            out.extend(line_samples(xml_path, root, pad, size))
    return out


def _relative(path: Path, root: str | Path | None) -> str:
    """Path relative to ``root`` when it is under it, else absolute.

    Samples are written relative to the job directory so a job can be moved (or
    read from the gateway host) without every path going stale.
    """
    path = Path(path)
    if root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def write_jsonl(path: str | Path, samples: Iterable[Sample]) -> int:
    """Write samples one JSON object per line. Returns how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(sample.to_json() + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[Sample]:
    """Stream samples back. Blank lines are skipped; a malformed line raises."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VlmDatasetError(f"{path}:{number} is not JSON: {exc}") from exc
            yield Sample.from_dict(raw)


def chat_example(prompt: str, text: str | None = None) -> list[dict]:
    """The chat turns for one sample, in the shape ``apply_chat_template`` wants.

    Built here rather than in the training script so the *exact* conversation the
    model is tuned on is defined once, unit-tested, and reused verbatim at
    evaluation time (where ``text`` is None — the assistant turn is what the model
    must produce). A prompt that drifts between training and inference is a silent
    distribution shift, which is why the trained ModelSpec also carries it.
    """
    messages: list[dict] = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    if text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return messages
