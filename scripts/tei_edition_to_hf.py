#!/usr/bin/env python3
"""Turn a TEI edition plus a IIIF image server into a HuggingFace page dataset (#91).

Written for the St. Gallen missives, but nothing here is specific to them beyond
the defaults: a TEI edition on GitHub whose ``<pb facs="…">`` names an image, and
a IIIF server whose identifier *is* that name.

    TEI:   <pb facs="StadtASG_Missive_1_1.JPG"/><lb n="1"/>Min undertaenig …
    IIIF:  https://…/iiif/2/sg-missiven!StadtASG_Missive_1_1.JPG/full/1500,/0/default.jpg

The output is the layout every dh-unibe dataset uses —
``data/train/<project>/*.parquet`` with ``image`` / ``xml_content`` / ``filename``
/ ``project_name`` — so the training pipeline reads it with no changes at all.

**Page granularity only.** An edition has no coordinates, so there are no line
crops and the emitted PageXML carries none. Train with
``"granularity": "page"`` and ``"engine": "vllm"``; kraken is a line-level CTC
engine and cannot use this. Asking for line granularity yields nothing, which is
the honest outcome rather than a silent fallback.

    .venvs/kraken-train/bin/python scripts/tei_edition_to_hf.py --limit 50 --dry-run
    .venvs/kraken-train/bin/python scripts/tei_edition_to_hf.py --limit 50
    .venvs/kraken-train/bin/python scripts/tei_edition_to_hf.py          # all of it

Repos are created **private**. Making an edition's images public is a licensing
decision this script will not take for you: the TEI is CC-BY-SA-4.0, the images
carry no statement, and the two are not the same question.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atr_serving.training.tei import TeiError, page_texts, to_pagexml  # noqa: E402

GH_API = "https://api.github.com/repos/{repo}/contents/{path}"
IIIF = "{base}/{prefix}{facs}/full/{size}/0/default.jpg"

DEFAULT_TEI_REPO = "Briefverkehr-der-Stadt-St-Gallen/sg-missiven-data"
DEFAULT_IIIF = "https://media.sources-online.org/cantaloupe/iiif/2"
DEFAULT_PREFIX = "sg-missiven!"
DEFAULT_TARGET = "dh-unibe/image-text_sg-missiven"
DEFAULT_PROJECT = "sg-missiven"

#: Long edge in pixels requested from IIIF. The originals are 3000 px; a VLM at
#: page granularity gets 2048 visual tokens, so 1500 is already more detail than
#: the budget can carry, and it halves both transfer and decode time.
DEFAULT_SIZE = "1500,"

#: Seconds between IIIF requests. This is somebody's public image server, and the
#: full run is ~1,640 fetches; the HuggingFace lesson of #89 was that a pipeline
#: which cannot survive its own traffic is the pipeline's problem, not the host's.
DEFAULT_DELAY = 0.25


def _get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    """GET with backoff on 429/5xx. Anything else is raised immediately."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "serving-atr/tei2hf"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
            wait = min(2 ** attempt, 30)
            print(f"    HTTP {exc.code}, waiting {wait}s ({attempt}/{retries})", flush=True)
            time.sleep(wait)
            last = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
            last = exc
    raise RuntimeError(f"unreachable: {last}")


def list_tei(repo: str, path: str, token: str | None) -> list[str]:
    """Every ``.xml`` under ``path``, paginated."""
    names: list[str] = []
    page = 1
    while True:
        url = GH_API.format(repo=repo, path=path) + f"?per_page=100&page={page}"
        request = urllib.request.Request(url, headers={
            "User-Agent": "serving-atr/tei2hf",
            "Accept": "application/vnd.github+json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        })
        with urllib.request.urlopen(request, timeout=60) as response:
            batch = json.load(response)
        if not batch:
            break
        names += [e["name"] for e in batch if e["type"] == "file" and e["name"].endswith(".xml")]
        if len(batch) < 100:
            break
        page += 1
    return sorted(names)


def fetch_tei(repo: str, path: str, name: str, token: str | None) -> str:
    url = GH_API.format(repo=repo, path=f"{path}/{name}")
    request = urllib.request.Request(url, headers={
        "User-Agent": "serving-atr/tei2hf",
        "Accept": "application/vnd.github+json",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return base64.b64decode(payload["content"]).decode("utf-8", "replace")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tei-dir", type=Path, default=None,
                   help="read TEI from a local checkout instead of the GitHub API "
                        "— no rate limit, no token, and the obvious choice when "
                        "the repo is already cloned")
    p.add_argument("--tei-repo", default=DEFAULT_TEI_REPO,
                   help="used only when --tei-dir is absent")
    p.add_argument("--tei-path", default="data")
    p.add_argument("--iiif-base", default=DEFAULT_IIIF)
    p.add_argument("--iiif-prefix", default=DEFAULT_PREFIX,
                   help="prepended to the facs name to form the IIIF identifier")
    p.add_argument("--size", default=DEFAULT_SIZE, help="IIIF size parameter")
    p.add_argument("--target", default=DEFAULT_TARGET, help="dataset repo to create")
    p.add_argument("--project", default=DEFAULT_PROJECT,
                   help="project directory inside the dataset")
    p.add_argument("--limit", type=int, default=None, help="only this many editions")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    p.add_argument("--out", type=Path, default=Path("/tmp/tei_hf"),
                   help="where the parquet is written before upload")
    p.add_argument("--dry-run", action="store_true",
                   help="convert the TEI and report; fetch no images, upload nothing")
    p.add_argument("--check-images", type=int, default=0, metavar="N",
                   help="with --dry-run: probe N pages against IIIF to confirm the "
                        "identifiers resolve, without downloading the corpus")
    p.add_argument("--public", action="store_true",
                   help="create a PUBLIC repo (default: private)")
    args = p.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    if args.tei_dir:
        source = args.tei_dir if args.tei_dir.name == args.tei_path else args.tei_dir / args.tei_path
        if not source.is_dir():
            print(f"no such directory: {source}", file=sys.stderr)
            return 1
        names = sorted(f.name for f in source.glob("*.xml"))
        print(f"reading TEI from {source}")
        read_tei = lambda n: (source / n).read_text(encoding="utf-8", errors="replace")
    else:
        print(f"listing {args.tei_repo}/{args.tei_path} …", flush=True)
        names = list_tei(args.tei_repo, args.tei_path, token)
        read_tei = lambda n: fetch_tei(args.tei_repo, args.tei_path, n, token)
    if args.limit:
        names = names[: args.limit]
    print(f"{len(names)} TEI file(s)")

    rows: list[dict] = []
    stats = {"editions": 0, "pages": 0, "lines": 0, "chars": 0,
             "no_image": 0, "unparseable": 0}
    missing: list[str] = []

    for i, name in enumerate(names, start=1):
        try:
            pages = page_texts(read_tei(name))
        except (TeiError, OSError, urllib.error.HTTPError) as exc:
            stats["unparseable"] += 1
            print(f"  [{i}/{len(names)}] {name}: SKIP ({exc})", flush=True)
            continue
        stats["editions"] += 1

        for page in pages:
            if args.dry_run:
                # A dry run that downloads 1,640 images is not a dry run. The TEI
                # side is what it checks; image availability is sampled separately
                # with --check-images.
                image = b""
            else:
                url = IIIF.format(base=args.iiif_base.rstrip("/"),
                                  prefix=args.iiif_prefix, facs=page.image, size=args.size)
                try:
                    image = _get(urllib.parse.quote(url, safe=":/?&=,!"))
                except Exception as exc:  # noqa: BLE001 — one missing scan is not fatal
                    stats["no_image"] += 1
                    missing.append(page.image)
                    print(f"  [{i}/{len(names)}] {page.image}: no image ({exc})", flush=True)
                    continue
            rows.append({
                "image": {"bytes": image, "path": page.image},
                "xml_content": to_pagexml(page),
                "filename": page.image,
                "project_name": args.project,
            })
            stats["pages"] += 1
            stats["lines"] += len(page.lines)
            stats["chars"] += len(page.text)
            if not args.dry_run:
                time.sleep(args.delay)

        if i % 10 == 0 or i == len(names):
            print(f"  [{i}/{len(names)}] {stats['pages']} pages, "
                  f"{stats['lines']} lines so far", flush=True)

    print("\n── converted ──────────────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<12} {v}")
    if stats["pages"]:
        print(f"  {'lines/page':<12} {stats['lines'] / stats['pages']:.1f}")
    if missing:
        print(f"  images missing: {missing[:5]}{' …' if len(missing) > 5 else ''}")
    if not rows:
        print("\nnothing to upload", file=sys.stderr)
        return 1

    if args.dry_run:
        if args.check_images:
            import random

            random.seed(0)
            probe = random.sample(rows, min(args.check_images, len(rows)))
            found = 0
            for row in probe:
                url = IIIF.format(base=args.iiif_base.rstrip("/"),
                                  prefix=args.iiif_prefix,
                                  facs=row["filename"], size=args.size)
                try:
                    _get(urllib.parse.quote(url.replace("/full/" + args.size + "/0/default.jpg",
                                                        "/info.json"), safe=":/?&=,!"),
                         timeout=20, retries=2)
                    found += 1
                except Exception:  # noqa: BLE001
                    print(f"  not on IIIF: {row['filename']}")
            print(f"\nIIIF probe: {found}/{len(probe)} sampled pages resolve")
        print("\n--dry-run: no images fetched, nothing written, nothing uploaded")
        print("first page's PageXML:\n")
        print(rows[0]["xml_content"][:600])
        return 0

    from datasets import Dataset, Features, Image, Value

    features = Features({
        "image": Image(decode=False),      # the pipeline requires the original bytes
        "xml_content": Value("string"),
        "filename": Value("string"),
        "project_name": Value("string"),
    })
    ds = Dataset.from_list(rows, features=features)

    out = args.out / "data" / "train" / args.project
    out.mkdir(parents=True, exist_ok=True)
    shard = out / "0000.parquet"
    ds.to_parquet(str(shard))
    print(f"\nwrote {shard} ({shard.stat().st_size / 1e6:.1f} MB)")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.target, repo_type="dataset",
                    private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.target, repo_type="dataset",
                      folder_path=str(args.out),
                      commit_message=f"{stats['pages']} pages from {args.tei_repo}")
    visibility = "PUBLIC" if args.public else "private"
    print(f"uploaded to {args.target} ({visibility})")
    print(f"\nTrain with:  \"hf_repo\": \"{args.target}\", "
          f"\"train_projects\": [\"{args.project}\"], \"granularity\": \"page\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
