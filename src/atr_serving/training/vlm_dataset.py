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
from atr_serving.training.pagexml import (
    MAX_LINE_ASPECT,
    is_plausible_line,
    line_boxes,
    line_texts,
)

__all__ = [
    "VlmDatasetError",
    "VisualBudgetError",
    "AppliedBudget",
    "FALLBACK_CELL_PX",
    "apply_visual_budget",
    "Sample",
    "DEFAULT_LINE_PAD",
    "MIN_CROP_PX",
    "MAX_LINE_ASPECT",
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
        # And the ceiling, which nothing enforced: a box 60x wider than it is tall
        # is a segmentation error far more often than a line, and it sets the
        # memory ceiling for every batch it lands in (#90).
        if not is_plausible_line(padded):
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


@dataclass(frozen=True)
class LengthFilter:
    """What :func:`drop_long_samples` kept, and what it found."""

    kept: list["Sample"]
    dropped: int
    max_chars: int

    def __str__(self) -> str:
        if not self.dropped:
            return f"no sample over the cap (longest {self.max_chars} chars)"
        return (f"dropped {self.dropped} sample(s) over the cap; the longest was "
                f"{self.max_chars} chars")


def drop_long_samples(samples: Iterable["Sample"], max_chars: int) -> LengthFilter:
    """Remove samples whose transcription cannot be afforded (#110).

    Cross-entropy upcasts the logits to fp32, so one sample costs
    ``tokens × vocab × 4`` bytes in a **single** allocation — 8.16 GiB for the
    14,411-token page that killed `20260908T101611Z-qwen3vl-german-pages-v1` in
    its eval loop, eleven hours in. There is nothing to be done about it at train
    time: truncating a multimodal sequence severs the image tokens from the
    placeholders that index them and produces an invalid sample rather than a
    shorter one (#86), and a batch of one cannot drop its only member.

    So it is done here, where a page can simply not become a sample, and where the
    cost is visible before any GPU time is spent. The same shape of fix as
    ``drop_wide_lines`` (#90), for the same shape of problem: the median is fine
    and the tail is fatal.

    ``max_chars`` counts characters rather than tokens because tokenizing the
    corpus would mean loading the processor into the supervising service, which
    imports no engine on purpose. At roughly 2 characters per token in this
    orthography the estimate is conservative in the safe direction.
    """
    kept: list[Sample] = []
    dropped = 0
    longest = 0
    for sample in samples:
        length = len(sample.text)
        longest = max(longest, length)
        if length > max_chars:
            dropped += 1
            continue
        kept.append(sample)
    return LengthFilter(kept=kept, dropped=dropped, max_chars=longest)


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


#: Passed to every ``apply_chat_template`` call, in training and in evaluation.
#:
#: The Qwen3.5 templates read ``enable_thinking``, and **they disagree about the
#: default**: 0.8B and 2B treat an unset variable as *off* (an empty
#: ``<think></think>`` block), while 4B treats it as *on* and opens ``<think>`` so
#: the model reasons before answering. Left unset, the 4B would generate a
#: reasoning trace in front of every line transcription — which is not a slightly
#: worse CER, it is a meaningless one — while the smaller two were correct only
#: by accident of their default. One transcription of one line needs no
#: reasoning, so it is pinned off everywhere.
#:
#: Harmless for Qwen3-VL, whose template never reads the variable; Jinja ignores
#: an extra name.
CHAT_TEMPLATE_KWARGS: dict = {"enable_thinking": False}


def chat_example(prompt: str, text: str | None = None,
                 system: str | None = None) -> list[dict]:
    """The chat turns for one sample, in the shape ``apply_chat_template`` wants.

    Built here rather than in the training script so the *exact* conversation the
    model is tuned on is defined once, unit-tested, and reused verbatim at
    evaluation time (where ``text`` is None — the assistant turn is what the model
    must produce). A prompt that drifts between training and inference is a silent
    distribution shift, which is why the trained ModelSpec also carries it.

    ``system`` is for models trained with an instruction in the system turn and
    nothing but the image in the user turn — CHURRO's template is exactly that
    (docs/CHURRO_PLAN.md §1.1). An empty ``prompt`` then means *no* text part in
    the user turn, not an empty one: a stray empty string is still a token
    sequence the model never saw in training.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    content: list[dict] = [{"type": "image"}]
    if prompt:
        content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})
    if text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return messages


# ── the visual-token budget (#86) ───────────────────────────────────────────
#: Area of one merged patch when a processor will not say: Qwen3-VL's 16 x 2.
FALLBACK_CELL_PX = 32


class VisualBudgetError(VlmDatasetError):
    """The visual-token budget could not be applied to this processor."""


@dataclass(frozen=True)
class AppliedBudget:
    """What ``apply_visual_budget`` actually set, so a caller can print it."""

    knob: str
    max_pixels: int
    cell_px: int
    visual_tokens: int
    #: True when the grid was read off the processor rather than assumed.
    grid_known: bool

    def __str__(self) -> str:
        grid = f"{self.cell_px}px cell" + ("" if self.grid_known else ", ASSUMED")
        return (f"{self.knob}={self.max_pixels} -> ~{self.visual_tokens} visual "
                f"tokens ({grid})")


def apply_visual_budget(processor, max_pixels: int) -> AppliedBudget:
    """Bound visual tokens per image, whatever this processor calls the knob.

    Passing ``max_pixels=`` to ``AutoProcessor.from_pretrained`` is a **Qwen2-VL**
    idiom. Qwen3-VL's image processor is a ``Qwen2VLImageProcessorFast`` configured
    through ``size={"longest_edge", "shortest_edge"}`` — areas in pixels — and it
    accepts the kwarg without applying it. So the budget looked set and was not:
    against an intended ~256 tokens a line crop produced 600, the sequence budget
    truncated it, and truncation severed the image tokens from the placeholders that
    index them. The job died at step 2 of 774 (#86).

    The knob is therefore written **directly onto the image processor** and read
    back. The read-back proves the attribute exists and now holds this value; it
    cannot prove the processor honours it, which would need a real image. That is
    still the difference between a budget that is wrong and one that is absent, and
    absent was the bug.

    The returned token figure is derived from the processor's own
    ``patch_size``/``merge_size`` rather than a constant, because that grid is what
    made :data:`~atr_serving.training.contracts.VLM_PIXEL_BUDGET` wrong for this
    model in the first place.
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise VisualBudgetError(
            "processor exposes no image_processor, so visual tokens cannot be "
            "bounded — refusing to train at the model's default, which for "
            "Qwen3-VL is 16384 tokens per image"
        )

    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and "longest_edge" in size:
        size["longest_edge"] = max_pixels
        image_processor.size = size
        knob, read_back = "size.longest_edge", image_processor.size.get("longest_edge")
    elif size is not None and hasattr(size, "longest_edge"):
        # transformers 5.x. ``size`` stopped being a plain dict and became a
        # ``SizeDict`` object that does not answer to mapping access, so the
        # branch above stops matching even though the knob is still there and
        # still called longest_edge. The guard caught it rather than training at
        # the default — 16,384 tokens an image against an intended 256 — which is
        # the whole reason this function refuses instead of proceeding (#86).
        size.longest_edge = max_pixels
        image_processor.size = size
        knob = "size.longest_edge"
        read_back = getattr(getattr(image_processor, "size", None), "longest_edge", None)
    elif getattr(image_processor, "max_pixels", None) is not None:
        image_processor.max_pixels = max_pixels
        knob, read_back = "max_pixels", image_processor.max_pixels
    else:
        raise VisualBudgetError(
            f"{type(image_processor).__name__} has neither size['longest_edge'] nor "
            "max_pixels; there is no knob here to bound visual tokens with"
        )

    if read_back != max_pixels:
        raise VisualBudgetError(
            f"set {knob}={max_pixels} but it reads back as {read_back!r} — the budget "
            "did not take, and training would run at the model's default"
        )

    patch = getattr(image_processor, "patch_size", None)
    merge = getattr(image_processor, "merge_size", None)
    grid_known = bool(patch) and bool(merge)
    cell = int(patch) * int(merge) if grid_known else FALLBACK_CELL_PX
    return AppliedBudget(knob=knob, max_pixels=max_pixels, cell_px=cell,
                         visual_tokens=max_pixels // (cell * cell),
                         grid_known=grid_known)
