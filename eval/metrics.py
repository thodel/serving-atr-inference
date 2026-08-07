"""Ground-truth loading for the eval harness; CER/WER come from the trainer's.

The metrics themselves live in :mod:`atr_serving.training.textmetrics` and are
re-exported here. One implementation, so a CER printed by an eval run and a CER
recorded on a training job are the same number computed the same way — the whole
point of `eval/run_eval.py` comparing a freshly trained model against the served
ones. ``score_pairs`` there additionally aggregates corpus-level, which is what
``ketos test`` reports.

Ground truth is read from ``.txt`` or PAGE-XML.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# eval/ is run from the repo root, where src/ is not automatically importable
# unless the gateway venv has the package installed (it does, `pip install -e .`).
# The fallback keeps `python eval/run_eval.py` working from a bare checkout.
try:
    from atr_serving.training.textmetrics import cer, levenshtein, score_pairs, wer
except ModuleNotFoundError:  # pragma: no cover - only without an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from atr_serving.training.textmetrics import cer, levenshtein, score_pairs, wer

#: Kept under its historical private name — eval code imported it directly.
_levenshtein = levenshtein

__all__ = ["cer", "wer", "levenshtein", "score_pairs",
           "parse_page_xml", "load_ground_truth", "find_ground_truth"]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_page_xml(path: Path) -> str:
    """Extract line text from a PAGE-XML file, one TextLine per line, in document order."""
    root = ET.parse(path).getroot()
    lines: list[str] = []
    for el in root.iter():
        if _local(el.tag) != "TextLine":
            continue
        for sub in el.iter():
            if _local(sub.tag) == "Unicode" and sub.text:
                lines.append(sub.text)
                break
    return "\n".join(lines)


def load_ground_truth(path: Path) -> str:
    if path.suffix.lower() == ".xml":
        return parse_page_xml(path)
    return path.read_text(encoding="utf-8").strip()


def find_ground_truth(image_path: Path, gt_dir: Path | None) -> Path | None:
    """Locate ground truth for an image: <stem>.txt / .gt.txt / .xml in gt_dir
    (default: alongside the image)."""
    base = gt_dir if gt_dir is not None else image_path.parent
    for name in (f"{image_path.stem}.txt", f"{image_path.stem}.gt.txt", f"{image_path.stem}.xml"):
        cand = base / name
        if cand.is_file():
            return cand
    return None
