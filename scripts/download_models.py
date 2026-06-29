#!/usr/bin/env python3
"""Prefetch all model weights named in config/models.yaml.

Run AFTER the venvs exist (it shells out to each engine venv so downloads land
with the right libraries and in HF_HOME):

    python scripts/download_models.py            # all
    python scripts/download_models.py --engine vllm trocr
    python scripts/download_models.py --dry-run

- vllm / trocr models  -> huggingface_hub.snapshot_download(hf_repo)
- kraken / party models -> `kraken get <zenodo_id>` (needs the kraken venv)

Honors HF_HOME from the environment (set it in .env — see docs/asteraix-environment.md;
the box's root partition is ~80% full).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_serving.registry import load_registry  # noqa: E402

VENVS = ROOT / ".venvs"


def hf_snapshot(repo: str, venv: str) -> list[str]:
    py = VENVS / venv / "bin" / "python"
    code = (
        "from huggingface_hub import snapshot_download;"
        f"snapshot_download({repo!r});"
        f"print('ok {repo}')"
    )
    return [str(py), "-c", code]


def kraken_get(zenodo_id: str) -> list[str]:
    return [str(VENVS / "kraken" / "bin" / "kraken"), "get", zenodo_id]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", nargs="*", choices=["vllm", "trocr", "kraken", "party"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reg = load_registry(ROOT / "config" / "models.yaml")
    specs = reg.all()
    if args.engine:
        specs = [s for s in specs if s.engine in args.engine]

    failures: list[str] = []
    for spec in specs:
        if spec.engine in {"vllm", "trocr"} and spec.hf_repo:
            venv = "vllm" if spec.engine == "vllm" else "trocr"
            cmd = hf_snapshot(spec.hf_repo, venv)
            label = f"[{spec.engine}] {spec.hf_repo}"
        elif spec.engine in {"kraken", "party"} and spec.zenodo_id:
            cmd = kraken_get(spec.zenodo_id)
            label = f"[{spec.engine}] {spec.zenodo_id}"
        else:
            print(f"SKIP {spec.id}: no source")
            continue

        print(f"==> {label}")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue
        if not Path(cmd[0]).exists():
            print(f"    SKIP — venv missing ({cmd[0]}); run make_venvs.sh first")
            failures.append(spec.id)
            continue
        if subprocess.run(cmd).returncode != 0:
            failures.append(spec.id)

    if failures:
        print(f"\n{len(failures)} failed/skipped: {', '.join(failures)}")
        return 1
    print("\nAll requested models prefetched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
