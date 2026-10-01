"""The kraken registry, held against the recorded mismatch set (#101).

28 of 43 kraken entries name something their DOI does not contain. #101 asked
for two things and deliberately did neither of two others:

    1. a report-only `scripts/audit_registry.py` that resolves every DOI and
       prints that table, so the state is checkable at any time;
    2. a test pinning the current mismatch set, so it can shrink but never grow.

Point 1 exists: `scripts/audit_registry.py` and `config/registry_mismatches.json`
were written with the issue. Point 2 did not, and could not, because resolving a
DOI needs zenodo.org — which means `--check` cannot run in CI, and nothing at
all pinned the state. This file is point 2, built from the parts that need no
network, because they are facts about the registry file rather than about Zenodo:

* five DOIs claimed by two ids each, all ten servable;
* every id in the baseline still present, still carrying a DOI;
* and `classify()`, the heuristic the whole table rests on, held against the
  rows of the issue itself.

**Nothing here corrects a mapping.** #101's reason stands: the name says what
somebody wanted, the DOI says what they got, and neither says which is the
mistake. Three of the mismatches resolve to models trained on the Inzigkofen
manuscripts, which this project benchmarks against — a wrong guess would turn
memorisation into a published recognition number (#100). The test's job is to
make a change visible, not to make one.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atr_serving.registry import load_registry  # noqa: E402
from atr_serving.registry_audit import (  # noqa: E402
    BASELINE,
    duplicate_dois,
    load_baseline,
    missing_from_registry,
    normalize_zenodo_id,
)
from scripts.audit_registry import classify, offline, record_id  # noqa: E402

#: The three entries that make this more than untidy naming: their DOIs resolve
#: to models trained on the Inzigkofen manuscripts, the corpus this project
#: benchmarks recognition against (#100).
INZIGKOFEN = ("kraken-medieval_generic_a", "kraken-medieval_generic_c",
              "kraken-medieval_generic_d")


@pytest.fixture(scope="module")
def kraken():
    registry = load_registry(ROOT / "config" / "models.yaml")
    return [s for s in registry.all() if s.engine == "kraken" and s.zenodo_id]


@pytest.fixture(scope="module")
def baseline():
    return load_baseline()


# ── the baseline is in the repository, not in an issue comment ──────────────
def test_the_recorded_mismatch_set_lives_in_the_repository(baseline):
    """The state has to be checkable offline, and an issue is not a file a test
    can read."""
    assert BASELINE.is_file()
    assert BASELINE.name == "registry_mismatches.json"
    assert len(baseline) == 28


def test_the_baseline_is_sorted_and_free_of_duplicates(baseline):
    """It is compared as a set and printed as a list; "whichever order the audit
    happened to resolve them in" is neither."""
    assert baseline == sorted(set(baseline))


def test_the_inzigkofen_entries_are_still_flagged(baseline):
    """The three that would turn memorisation into a recognition number. If one
    leaves the baseline, it was corrected or the baseline was edited, and from
    here those look alike."""
    for model_id in INZIGKOFEN:
        assert model_id in baseline


def test_every_recorded_mismatch_is_still_in_the_registry_with_a_doi(baseline, kraken):
    """The pin. An id that vanished, or lost its DOI, is one way to fix a
    mismatch and one way to hide one — so it fails here and a person decides."""
    gone = missing_from_registry(baseline, kraken)

    assert gone == [], (
        "ids recorded as mismatched that the registry no longer has. If that "
        "was a correction, re-record it: scripts/audit_registry.py "
        f"--write-baseline. {gone}")


def test_the_mismatch_set_can_shrink_but_not_grow(baseline, kraken):
    """The "never grow" half, as a statement about counts: 28 of 43. A new
    kraken entry nobody resolved is exactly what produced this issue — the block
    was filled in without per-entry checking — so the ratio is pinned too."""
    assert len(baseline) <= len(kraken)
    assert (len(kraken), len(baseline)) == (43, 28)


def test_a_removed_id_is_reported(kraken):
    assert missing_from_registry(["kraken-not-in-the-registry"], kraken) == \
        ["kraken-not-in-the-registry"]


def test_an_id_that_lost_its_doi_is_reported(kraken):
    """Dropping the DOI makes the entry unauditable rather than correct."""
    class _NoDoi:
        id = INZIGKOFEN[0]
        zenodo_id = None

    others = [s for s in kraken if s.id != INZIGKOFEN[0]]

    assert missing_from_registry([INZIGKOFEN[0]], [*others, _NoDoi()]) == [INZIGKOFEN[0]]


# ── the duplicates, which need no network ───────────────────────────────────
def test_five_dois_are_claimed_by_two_ids_each(kraken):
    """A fact about the file. #101's prose says four and its own table lists
    five, which is why this is a test and not a sentence."""
    duplicates = duplicate_dois(kraken)

    assert len(duplicates) == 5
    assert all(len(ids) == 2 for ids in duplicates.values())


def test_the_registry_serves_fewer_models_than_it_lists(kraken):
    unique = {normalize_zenodo_id(str(s.zenodo_id)) for s in kraken}

    assert (len(kraken), len(unique)) == (43, 38)


def test_both_halves_of_every_duplicate_are_servable(kraken):
    """Why it matters rather than being untidy: a caller can pick either id and
    cannot tell they are one model."""
    by_id = {s.id: s for s in kraken}

    for ids in duplicate_dois(kraken).values():
        assert all(by_id[i].enabled for i in ids), ids


def test_each_duplicate_pair_is_an_underscore_and_a_hyphen_of_one_name(kraken):
    """The shape that says these are a naming slip rather than two deliberate
    aliases — and the thing a de-duplication would key on."""
    for ids in duplicate_dois(kraken).values():
        assert len({i.replace("-", "_") for i in ids}) == 1, ids


def test_the_duplicates_are_the_pairs_the_issue_named(kraken):
    """Shrinking is welcome; a new pair is not."""
    pairs = {tuple(ids) for ids in duplicate_dois(kraken).values()}

    assert ("kraken-catmus-medieval", "kraken-catmus_medieval") in pairs
    assert ("kraken-printed-french", "kraken-printed_french") in pairs


def test_an_entry_without_a_doi_is_not_counted(kraken):
    """local_path models — the ones trained here — have no Zenodo record and are
    not this audit's business."""
    class _NoDoi:
        id = "kraken-thun-kurrent-v2"
        zenodo_id = None

    assert duplicate_dois([*kraken, _NoDoi()]) == duplicate_dois(kraken)


def test_duplicates_come_back_in_a_stable_order(kraken):
    """Compared against a recorded set, so file order is not an answer."""
    duplicates = duplicate_dois(kraken)

    assert list(duplicates) == sorted(duplicates)
    assert all(ids == sorted(ids) for ids in duplicates.values())


# ── classify(), held against the rows of the issue ──────────────────────────
@pytest.mark.parametrize("model_id,title", [
    ("kraken-catmus_caroline",
     "Medieval Hebrew manuscripts in Sephardi bookhand version 1.0"),
    ("kraken-bohemian_19th", "A generalized model for English printed text"),
    ("kraken-early_modern_german", "CATMuS Medieval"),
    ("kraken-medieval_15_16", "HTR model for (Japanese) Kuzushiji"),
    ("kraken-medieval_generic_e",
     "Fanny loves Wilhelm. HTR Model trained on letters and notes by Fanny "
     "Mendelssohn (Fanny Hensel)"),
    ("kraken-czech_historic",
     "Printed Arabic-Script Base Model Trained on the OpenITI Corpus"),
    ("kraken-late_medieval_german",
     "Bifrost: A Handwritten Text Recognition Model for Old Norse"),
])
def test_the_rows_of_the_issue_still_classify_as_mismatches(model_id, title):
    """The heuristic behind the whole table, which nothing pinned. Loosen it and
    these rows quietly stop being findings."""
    assert classify(model_id, title) == "mismatch"


@pytest.mark.parametrize("model_id,title", [
    ("kraken-catmus_medieval", "CATMuS Medieval"),
    ("kraken-catmus-medieval", "CATMuS Medieval"),
    ("kraken-printed_french", "LECTAUREP Contemporary French Model"),
])
def test_a_title_that_contains_the_name_is_a_match(model_id, title):
    assert classify(model_id, title) == "match"


def test_a_shared_family_is_related_rather_than_a_mismatch():
    """Deliberately generous: a Persian model under an Arabic id is a defensible
    grouping. A Japanese one under a medieval-Latin id is not, which is the
    distinction the family table buys."""
    assert classify("kraken-arabic_manuscripts",
                    "Printed Ottoman Base Model Trained on the OpenITI Corpus") \
        == "related"


def test_an_unresolved_record_is_not_reported_as_a_mismatch():
    """A network failure is not a finding about the registry, and counting it as
    one would make the mismatch set grow whenever Zenodo is slow."""
    assert classify("kraken-catmus_medieval", "<unresolved: URLError>") == "unresolved"


def test_version_suffixes_do_not_make_a_mismatch():
    """`v2`, `base`, `extended` and the generic letters carry no claim about the
    model's content, so they must not count against a title."""
    assert classify("kraken-catmus_medieval_v2", "CATMuS Medieval") == "match"


# ── one answer to "which record is that" ────────────────────────────────────
@pytest.mark.parametrize("given,expected", [
    ("10.5281/zenodo.15366732", "15366732"),
    ("zenodo.15366732", "15366732"),
    ("15366732", "15366732"),
    ("https://zenodo.org/record/15366732/", "15366732"),
    ("  10.5281/zenodo.7516057  ", "7516057"),
])
def test_a_zenodo_reference_normalises_to_its_record_id(given, expected):
    assert normalize_zenodo_id(given) == expected
    assert record_id(given) == expected


def test_the_normaliser_is_not_a_second_copy():
    """`scripts/discover_models.py` had it as a private helper. Two copies would
    be two answers to "which record is that", and the audit and the discovery
    have to agree about the same DOI."""
    script = (ROOT / "scripts" / "discover_models.py").read_text(encoding="utf-8")

    assert "def _normalize_zenodo_id" not in script
    assert "from atr_serving.registry_audit import normalize_zenodo_id" in script


def test_the_audit_and_the_discovery_use_the_same_function():
    """Identity, not equality: the point is that there is one of it."""
    from scripts.discover_models import normalize_zenodo_id as discovery

    assert discovery is normalize_zenodo_id


# ── the script ──────────────────────────────────────────────────────────────
def test_the_offline_run_needs_no_network(capsys):
    assert offline("kraken") == 0

    printed = capsys.readouterr().out
    assert "43 entries for kraken, 43 with a DOI, 38 distinct records" in printed
    assert "28 ids recorded as mismatched, all still in the registry" in printed


def test_the_offline_run_names_every_duplicate_pair(capsys):
    offline("kraken")

    printed = capsys.readouterr().out
    for model_id in ("kraken-printed-french", "kraken-catmus_medieval",
                     "kraken-openiti-arabic"):
        assert model_id in printed


def test_a_baseline_that_no_longer_fits_the_registry_fails_the_offline_run(
        monkeypatch, capsys):
    """Through the script, not just the function: `--offline` is what CI runs,
    so its exit code is the thing that has to be non-zero."""
    import scripts.audit_registry as script

    monkeypatch.setattr(script, "load_baseline",
                        lambda *a, **k: ["kraken-corrected-by-someone"])

    assert script.offline("kraken") == 1
    printed = capsys.readouterr().out
    assert "kraken-corrected-by-someone" in printed
    assert "--write-baseline" in printed


def test_the_offline_run_is_reachable_from_the_command_line():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_registry.py"),
         "--offline", "--engine", "kraken"],
        capture_output=True, text=True, timeout=120, cwd=ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "38 distinct records" in result.stdout


def test_the_script_never_writes_the_registry():
    """Report-only is the whole posture. A script that could edit `models.yaml`
    would eventually be asked to guess which half of a mismatch is wrong."""
    source = (ROOT / "scripts" / "audit_registry.py").read_text(encoding="utf-8")
    writes = [line.strip() for line in source.splitlines()
              if "write_text" in line or "safe_dump" in line]

    assert writes == ["BASELINE.write_text(json.dumps({\"mismatched_ids\": current}, "
                      "indent=1) + \"\\n\","], writes


def test_the_baseline_file_is_the_shape_the_script_writes():
    """`--write-baseline` and `--check` are the only writer and reader, and a
    hand-edit that drifts from their format breaks both silently."""
    raw = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert set(raw) == {"mismatched_ids"}
    assert all(isinstance(i, str) for i in raw["mismatched_ids"])
