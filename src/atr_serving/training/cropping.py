"""Line cropping: cut transcribed TextLines out of their page scans.

``vlm_dataset.Sample`` carries a bbox (computed by :func:`line_boxes`) and a path
to the page scan. This module owns the pixel half of that contract: open one page
at a time (samples arrive grouped by page — caching all decoded scans is
gigabytes of RAM), clamp the bbox to the page, crop, and write a JPEG.

PIL is a **gateway dependency** (``pyproject.toml``), so this can live in the
shared training core and is unit-testable in the repo venv.

Public API
----------
``write_crops(samples, root, dest) -> list[Sample]``
    Cut each sample's bbox out of its page and return repointed samples.
    Samples without a bbox pass through unchanged (page-granularity samples).
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from atr_serving.training.vlm_dataset import Sample

__all__ = ["write_crops"]


def write_crops(
    samples: list[Sample],
    root: Path | str,
    dest: Path | str,
) -> list[Sample]:
    """Crop each sample's bbox from its page scan and write a JPEG.

    The source image path (``sample.image``) is resolved relative to ``root``.
    The crop is written under ``dest`` with a zero-padded index filename so
    training pipelines can enumerate them deterministically.

    The bbox is **clamped to the page** before cropping. Without clamping, PIL
    pads an out-of-bounds box with black, and a padded band of black is a worse
    training signal than a slightly tighter crop. Transkribus polygons routinely
    overrun the page edge once padding is added; the same clamp is tested in
    the VLM pipeline suite.

    Parameters
    ----------
    samples
        Samples as returned by :func:`vlm_dataset.samples_for`. Samples with
        ``bbox=None`` (page-granularity) are returned unchanged.
    root
        Job root: the directory under which ``sample.image`` is relative.
    dest
        Directory to write crops into. Created if it does not exist.

    Returns
    -------
    list[Sample]
        Samples with the same text/source_type/page, but ``image`` repointed to
        the crop path (relative to ``root``) and ``bbox=None``.
    """
    from PIL import Image  # gateway dep; safe to import here

    root = Path(root)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    out: list[Sample] = []
    open_path: str | None = None
    page: "Image.Image | None" = None

    for index, sample in enumerate(samples):
        if sample.bbox is None:
            # page-granularity: train on the whole scan, no crop needed
            out.append(sample)
            continue

        if sample.image != open_path:
            # One page held open at a time. Samples arrive grouped by page, so
            # this is one decode per page; caching them all would be gigabytes.
            if page is not None:
                page.close()
            page = Image.open(root / sample.image).convert("RGB")
            open_path = sample.image

        # Clamp to page bounds — load-bearing: PIL would otherwise pad with
        # black, and Transkribus polygons frequently overrun the edge.
        left, top, right, bottom = sample.bbox
        box = (
            max(left, 0),
            max(top, 0),
            min(right, page.width),
            min(bottom, page.height),
        )

        crop_path = dest / f"{index:07d}.jpg"
        page.crop(box).save(crop_path, format="JPEG", quality=95)
        logger.debug("cropped {} -> {}", sample.image, crop_path.name)

        out.append(
            Sample(
                image=str(crop_path.relative_to(root)),
                text=sample.text,
                source_type=sample.source_type,
                bbox=None,
                page=sample.page,
            )
        )

    if page is not None:
        page.close()

    return out