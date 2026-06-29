"""CER / WER and ground-truth loading for the eval harness.

No external dependencies — a plain Levenshtein over characters (CER) or
whitespace tokens (WER). Ground truth is read from ``.txt`` or PAGE-XML.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence


def _levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two sequences (O(len(a)*len(b)) time, O(len(b)) space)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    """Character error rate = edits / len(ref). Empty ref → 0.0 if pred empty else 1.0."""
    if not ref:
        return 0.0 if not pred else 1.0
    return _levenshtein(pred, ref) / len(ref)


def wer(pred: str, ref: str) -> float:
    """Word error rate over whitespace-split tokens."""
    ref_tokens = ref.split()
    if not ref_tokens:
        return 0.0 if not pred.split() else 1.0
    return _levenshtein(pred.split(), ref_tokens) / len(ref_tokens)


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
