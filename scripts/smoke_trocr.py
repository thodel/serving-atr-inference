#!/usr/bin/env python3
"""Ein TrOCR-Trainingslauf in 30 Sekunden, auf der CPU (#99, T3).

    .venvs/trocr-train/bin/python scripts/smoke_trocr.py

Baut vier synthetische Zeilenbilder, schreibt das JSONL, das ``compile`` sonst
schreibt, und laesst ``train_trocr`` eine Epoche darauf laufen — inklusive
Evaluation mit ``predict_with_generate`` und Checkpoint. Kein GPU, kein
Datensatz, keine Vorbereitung.

**Warum es das gibt.** ``train_trocr.py`` war vollstaendig implementiert, hatte
Tests, war in ``BACKENDS`` registriert — und war nie gelaufen. Der erste echte
Aufruf brauchte sieben Korrekturen hintereinander, jede davon erst sichtbar,
nachdem die vorige behoben war: ``length_penalty`` in den TrainingArguments,
``tokenizer=`` im Trainer, ``model=None`` nie zugewiesen, ``resize_token_embeddings``
auf dem Verbundmodell, ``length_penalty`` bei Greedy, die weggeworfenen
Dataset-Spalten, und ``input_ids`` statt ``labels`` im Collator. Unit-Tests mit
Fakes finden davon keinen einzigen, weil alle sieben an der echten
transformers-API haengen.

Nichts davon braucht eine GPU oder mehr als vier Bilder. Deshalb steht das hier.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "microsoft/trocr-base-handwritten"
LINES = ["hallo welt", "zweite zeile", "dritte zeile", "vierte"]


def build_dataset(work: Path) -> None:
    from PIL import Image, ImageDraw

    rows = []
    for index, text in enumerate(LINES):
        image = Image.new("RGB", (384, 48), "white")
        ImageDraw.Draw(image).text((4, 14), text, fill="black")
        image.save(work / f"{index}.png")
        rows.append({"image": f"{index}.png", "text": text})
    # Train und val identisch: geprueft wird der Pfad, nicht die Generalisierung.
    for name in ("train", "val"):
        (work / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="trocr-smoke-") as tmp:
        work = Path(tmp)
        build_dataset(work)
        cmd = [
            sys.executable, "-m", "trocr_train_svc.train_trocr",
            "--base-model", BASE_MODEL,
            "--train-manifest", str(work / "train.jsonl"),
            "--val-manifest", str(work / "val.jsonl"),
            "--output-dir", str(work / "out"),
            "--epochs", "1", "--batch-size", "2", "--accumulate-grad-batches", "1",
            "--precision", "fp32", "--device", "cpu", "--workers", "0",
            "--max-new-tokens", "16", "--beam-size", "1",
            "--no-gradient-checkpointing",
        ]
        env = {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'engines'}",
               "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
               "HF_HOME": str(Path.home() / "atr-cache" / "hf")}
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print("\nSMOKE FEHLGESCHLAGEN — siehe Traceback oben", file=sys.stderr)
            return result.returncode
        saved = work / "out" / "training_summary.json"
        print(f"\nSMOKE OK — {saved.name}: {saved.read_text().strip()[:120]}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
