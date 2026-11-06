"""Where a model's held-out pages live, and the selection with its proof (#181, part 1).

The corpus is built in tmp_path the way a run on the training host leaves it:
the pages in the artefact entry (where a run that adopted its corpus stores
them, as pages-v5 on asteraix) and the train/val/reserved lists in the job
directory on the share. The fixtures here are shared with
test_eval_pages_archive.py (part 2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atr_serving import eval_pages as ep  # noqa: E402

JOB_ID = "20260916T041001Z-qwen3vl-german-pages-v5"
DIGEST = "ab12cd34" * 15
MODEL = "qwen3vl-german-pages-v5-merged"


def page_xml(stem: str, lineless: bool = False, ns: bool = False) -> str:
    if lineless:
        return (
            f'<Page imageFilename="{stem}.jpg">'
            '<TextRegion id="r1">'
            '<TextLine id="l1"><Coords points="10,50 300,50 300,80 10,80"/></TextLine>'
            "</TextRegion></Page>"
        )
    if ns:
        open_tag, close_tag = 'Page xmlns="urn:test-page"', "Page"
    else:
        open_tag = close_tag = "Page"
    return f"""<{open_tag} imageFilename="{stem}.jpg">
  <TextRegion id="r1">
    <TextLine id="l1">
      <Coords points="10,100 300,100 300,130 10,130"/>
      <TextEquiv><Unicode>Erste Zeile {stem}</Unicode></TextEquiv>
    </TextLine>
    <TextLine id="l2">
      <Baseline points="5,200 250,200"/>
      <TextEquiv><Unicode>Zweite Zeile {stem}</Unicode></TextEquiv>
    </TextLine>
  </TextRegion>
</{close_tag}>"""


def make_corpus(root: Path, *, reserved: bool = True, lineless: bool = False,
                groups: int = 3, per_group: int = 6):
    """trained/ + jobs/ + artefacts/ for one job; returns paths and the page names."""
    trained = root / "trained"
    jobs = root / "jobs"
    artefacts = root / "artefacts"
    (trained / MODEL).mkdir(parents=True)
    (trained / MODEL / "metadata.json").write_text(
        json.dumps({"model_id": MODEL, "job_id": JOB_ID}), encoding="utf-8"
    )
    job_dir = jobs / JOB_ID
    (job_dir / "data").mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps({"request": {"model_id": "pages-v5"},
                    "progress": {"artefact": DIGEST}}),
        encoding="utf-8",
    )
    pages = [f"g{gi}_{i:05d}" for gi in range(groups) for i in range(per_group)]
    train = [p for p in pages if p.endswith("00000")]
    val = [p for p in pages if p.endswith("00001")]
    rest = [p for p in pages if p not in train and p not in val]
    resv = rest[:7] if reserved else []

    def lst(name, names):
        (job_dir / "data" / name).write_text(
            "".join(f"/ubelix/runs/{n}.xml\n" for n in names), encoding="utf-8"
        )

    lst("pages_train.lst", train)
    lst("pages_val.lst", val)
    if reserved:
        lst("pages_reserved.lst", resv)

    pages_dir = artefacts / DIGEST / "data" / "pages"
    pages_dir.mkdir(parents=True)
    for p in pages:
        Image.new("RGB", (320, 200), "white").save(pages_dir / f"{p}.jpg", format="JPEG")
        (pages_dir / f"{p}.xml").write_text(page_xml(p, lineless=lineless), encoding="utf-8")
    return trained, jobs, artefacts, {"pages": pages, "train": train, "val": val, "reserved": resv}


def _sets(info):
    return {p + ".xml" for p in info["train"]}, \
        {p + ".xml" for p in info["val"]}, \
        {p + ".xml" for p in info["reserved"]}


# --- resolving ----------------------------------------------------------------

def test_resolve_model_job(tmp_path):
    trained, _, _ = make_corpus(tmp_path)[:3]
    assert ep.resolve_model_job(MODEL, trained) == (MODEL, JOB_ID)
    assert ep.resolve_model_job(JOB_ID, trained) == (None, JOB_ID)
    with pytest.raises(ep.EvalPagesError, match="does not exist"):
        ep.resolve_model_job("no-such-model", trained)


def test_resolve_corpus_pages_artefact_lists_job(tmp_path):
    trained, jobs, artefacts, _ = make_corpus(tmp_path)
    corpus = ep.resolve_corpus(JOB_ID, jobs, artefacts)
    assert corpus.pages_from == "artefact"  # pages in the artefact entry
    assert corpus.artefact == DIGEST
    assert corpus.requested_model_id == "pages-v5"
    assert str(corpus.train_lst).startswith(str(jobs))  # lists in the job dir
    assert corpus.reserved_lst is not None


def test_resolve_corpus_requires_the_lists(tmp_path):
    trained, jobs, artefacts, _ = make_corpus(tmp_path)
    (jobs / JOB_ID / "data" / "pages_train.lst").unlink()
    with pytest.raises(ep.EvalPagesError, match="proof"):
        ep.resolve_corpus(JOB_ID, jobs, artefacts)
    with pytest.raises(ep.EvalPagesError, match="no job"):
        ep.resolve_corpus("20200101T000000Z-x", jobs, artefacts)


# --- selection ----------------------------------------------------------------

def test_select_pages_prefers_reserved_and_proves(tmp_path):
    _, _, _, info = make_corpus(tmp_path)
    train, val, reserved = _sets(info)
    pages, proof = ep.select_pages(info["pages"], train, val, reserved, 5, 42)
    assert len(pages) == 5
    assert set(pages) <= {s[:-4] for s in reserved}  # all from the held-out pool
    assert proof["train_overlap"] == [] and proof["val_overlap"] == []
    assert sorted(p + ".xml" for p in pages) == proof["reserved_in_selection"]


def test_select_pages_fills_from_pool_and_spreads(tmp_path):
    _, _, _, info = make_corpus(tmp_path)
    train, val, reserved = _sets(info)
    pages, proof = ep.select_pages(info["pages"], train, val, reserved, 10, 7)
    assert len(pages) == 10
    assert {p.split("_")[0] for p in pages} == {"g0", "g1", "g2"}
    assert proof["train_overlap"] == [] and proof["val_overlap"] == []
    assert len(proof["reserved_in_selection"]) == 7  # the whole reserved pool
    assert set(pages) - {s[:-4] for s in reserved}  # ...filled from the rest


def test_select_pages_reproducible_and_rejects_too_few(tmp_path):
    _, _, _, info = make_corpus(tmp_path)
    train, val, reserved = _sets(info)
    stems = info["pages"]
    a, _ = ep.select_pages(stems, train, val, reserved, 8, 42)
    b, _ = ep.select_pages(stems, train, val, reserved, 8, 42)
    assert a == b
    with pytest.raises(ep.EvalPagesError, match="memory"):
        ep.select_pages(stems, train, val, reserved, len(stems) + 1, 42)
