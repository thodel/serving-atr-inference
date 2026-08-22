#!/usr/bin/env python3
"""Plan a combined training corpus from the datasets on the HuggingFace Hub (#87).

Reads every dataset card of an org, scores each against a target profile
(period, language, script class), removes projects that two datasets both
publish, caps any dataset that would dominate, and writes a ready job request.

Needs ``huggingface_hub`` — the gateway venv deliberately does not have it, and
the dh-unibe datasets are gated, so run it with a trainer venv that is logged in:

    .venvs/kraken-train/bin/python scripts/plan_corpus.py --org dh-unibe
    .venvs/kraken-train/bin/python scripts/plan_corpus.py --json /tmp/corpus.json \
        --engine vllm --model-id qwen3vl-medieval-german-v1

    --period 1300 1600     target span (default: 14th-16th century)
    --max-share 0.45       cap on any one dataset's share of the corpus
    --max-pages N          overall page budget
    --exclude-project P    held-out material; repeatable
    --cache FILE           write/reuse the fetched catalogue, to iterate offline

Every number it prints is an estimate from card metadata. Only ``prepare`` knows
the real page and line counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_serving.training.corpus_plan import (  # noqa: E402
    Candidate,
    CorpusPlanError,
    Target,
    job_request,
    parse_card,
    plan_corpus,
    score_candidate,
)

def fetch_catalogue(org: str) -> list[dict]:
    """Every dataset of ``org``, with its card text, page count and size."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    rows = []
    for info in api.list_datasets(author=org):
        try:
            path = hf_hub_download(info.id, "README.md", repo_type="dataset")
            card = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:                       # gated, missing, offline
            print(f"  skip {info.id}: {exc}", file=sys.stderr)
            continue
        detail = api.dataset_info(info.id, files_metadata=False)
        card_data = getattr(detail, "cardData", None) or {}
        dataset_info = card_data.get("dataset_info") or {}
        if isinstance(dataset_info, list):
            dataset_info = dataset_info[0] if dataset_info else {}
        splits = dataset_info.get("splits") or []
        projects = [line[2:].strip() for line in card.splitlines()
                    if line.startswith("- ") and len(line) > 2]
        rows.append({
            "repo": info.id,
            "card": card,
            "pages": sum(s.get("num_examples") or 0 for s in splits),
            "gb": round(sum(s.get("num_bytes") or 0 for s in splits) / 1024**3, 1),
            "projects": projects,
        })
    return rows


def to_candidates(rows: list[dict]) -> list[Candidate]:
    return [parse_card(r["repo"], r["card"], r["pages"], r["gb"], r["projects"])
            for r in rows if r["pages"]]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--org", default="dh-unibe")
    p.add_argument("--period", nargs=2, type=int, default=[1300, 1600],
                   metavar=("FROM", "TO"))
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--max-share", type=float, default=0.45)
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--exclude-project", action="append", default=[])
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None, help="write a job request here")
    p.add_argument("--engine", default="vllm", choices=["kraken", "vllm", "trocr"])
    p.add_argument("--model-id", default="corpus-v1")
    p.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--eval-repo", default="", help="must be a repo the plan selects")
    p.add_argument("--eval-project", action="append", default=[])
    args = p.parse_args(argv)

    if args.cache and args.cache.is_file():
        rows = json.loads(args.cache.read_text(encoding="utf-8"))
        print(f"catalogue: {len(rows)} datasets (cached)")
    else:
        print(f"fetching {args.org} …", file=sys.stderr)
        rows = fetch_catalogue(args.org)
        if args.cache:
            args.cache.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"catalogue: {len(rows)} datasets")

    target = Target(period=(args.period[0], args.period[1]), threshold=args.threshold)
    candidates = to_candidates(rows)

    print(f"\nscored against {target.period}, threshold {target.threshold:.2f}\n")
    print(f"{'dataset':<50}{'score':>7}{'per':>6}{'lang':>6}{'scr':>6}{'pages':>9}")
    print("-" * 84)
    for s in sorted((score_candidate(c, target) for c in candidates),
                    key=lambda s: -s.score):
        c = s.candidate
        print(f"{c.repo.split('/')[-1][:49]:<50}{s.score:>7.2f}{s.period:>6.2f}"
              f"{s.language:>6.2f}{s.script:>6.2f}{c.pages:>9}")

    eval_projects = list(args.eval_project)
    try:
        plan = plan_corpus(candidates, target, max_share=args.max_share,
                           max_pages=args.max_pages,
                           exclude_projects=args.exclude_project + eval_projects)
    except CorpusPlanError as exc:
        print(f"\nno plan: {exc}", file=sys.stderr)
        return 1

    print(f"\n── corpus ─────────────────────────────────────────────────────")
    for s in plan.selections:
        share = s.pages / plan.pages if plan.pages else 0
        extra = f"  (-{len(s.dropped_duplicates)} dup)" if s.dropped_duplicates else ""
        print(f"{s.repo.split('/')[-1][:46]:<48}{s.pages:>8}p{share:>7.0%}{extra}")
    print(f"{'total':<48}{plan.pages:>8}p")
    print(f"\n~{plan.estimated_lines:,} lines estimated, largest share "
          f"{plan.largest_share:.0%}")
    for note in plan.notes:
        print(f"  · {note}")
    if plan.rejected:
        print("\nrejected:")
        for repo, why in plan.rejected:
            print(f"  · {repo.split('/')[-1][:44]:<46} {why[:70]}")

    if not eval_projects:
        biggest = max(plan.selections, key=lambda s: s.pages)
        suggestion = list(biggest.projects[-2:]) or ["<a project of this repo>"]
        print("\nNO EVALUATION SET. Chunked prepare needs explicit eval_projects — a "
              "validation set cannot come from splitting a stream that is discarded "
              "as it is read — and a corpus this size needs chunking.")
        print(f"  suggestion: --eval-repo {biggest.repo} " +
              " ".join(f"--eval-project {p}" for p in suggestion))

    if args.json:
        try:
            body = job_request(plan, engine=args.engine, model_id=args.model_id,
                               base_model=args.base_model,
                               eval_repo=args.eval_repo, eval_projects=eval_projects)
        except CorpusPlanError as exc:
            print(f"\ncannot build a request: {exc}", file=sys.stderr)
            return 1
        args.json.write_text(json.dumps(body, indent=2, ensure_ascii=False))
        print(f"\njob request → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
