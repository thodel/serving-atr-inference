"""The kraken registry, held against the recorded mismatch set (#101).

**The mapping has since been corrected (PR #198): every entry is named after the
Zenodo record its DOI loads, the baseline is empty, and the duplicates are gone.**
What follows therefore pins the corrected state — the same tests, read the other
way round: a new mismatch, a new duplicate pair, or a re-appearing DOI without
weights all fail here.

28 of 43 kraken entries used to name something their DOI does not contain. #101
asked for two things and deliberately did neither of two others:

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

**How the correction was decided**, since this file used to say it could not be:
the DOI is what the engine downloads and loads, so the record is the only
verifiable side, and most of the wished-for names (`medieval_generic_a`…`_e`,
`printed_urdu_wide`, `czech_historic`) name no published model at all. The three
Inzigkofen models are now named after their records and carry the ground truth
they were trained on, which is what keeps #100 — memorisation reported as a
recognition number — visible at the point of choosing a model.
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

#: The three entries that make this more than untidy naming: they are trained on
#: the Inzigkofen manuscripts, the corpus this project benchmarks recognition
#: against (#100). Named `medieval_generic_a/_c/_d` until PR #198.
INZIGKOFEN = ("kraken-bastarda_inzigkofen", "kraken-cursive_inzigkofen",
              "kraken-textualis_inzigkofen")
#: The DOI that publishes a metadata JSON and no .mlmodel, removed in PR #198.
WITHOUT_WEIGHTS = "18732245"


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
    assert len(baseline) == 0, "PR #198 corrected every mismatch; see the diff"


def test_the_baseline_is_sorted_and_free_of_duplicates(baseline):
    """It is compared as a set and printed as a list; "whichever order the audit
    happened to resolve them in" is neither."""
    assert baseline == sorted(set(baseline))


def test_the_inzigkofen_entries_say_what_they_were_trained_on(kraken):
    """The three that would turn memorisation into a recognition number. They are
    no longer hidden behind a generic name, and each one records the ground truth
    its record names — which is what a caller needs before scoring on Inzigkofen
    (#100)."""
    by_id = {s.id: s for s in kraken}

    for model_id in INZIGKOFEN:
        assert model_id in by_id, f"{model_id} left the registry"
        assert by_id[model_id].training_datasets, model_id


def test_every_entry_with_a_doi_describes_itself(kraken):
    """`/models` serves `description`, and for a DOI entry it is the record's
    title — the one description nobody has to take on trust. The audit compares
    the two against Zenodo on every run; this is the offline half, that none is
    missing."""
    missing = [s.id for s in kraken if not s.description]

    assert missing == [], (
        f"{missing} carry a DOI but describe nothing. The title is in the record: "
        f"scripts/audit_registry.py prints it.")


def test_a_fine_tune_of_a_served_model_says_so(kraken):
    """`kraken-bifrost_old_norse` is a fine-tune of `kraken-catmus_medieval`,
    which is served here too: two such candidates are not independent, however
    different their ids look. The lineage has to be readable from the registry."""
    by_doi = {normalize_zenodo_id(str(s.zenodo_id)): s.id for s in kraken}
    lineage = {s.id: normalize_zenodo_id(str(s.base_model))
               for s in kraken if s.base_model}

    assert lineage, "no kraken entry records a base_model"
    assert lineage["kraken-bifrost_old_norse"] == "15030337"
    for model_id, base in lineage.items():
        assert base in by_doi, f"{model_id} names a base this registry does not serve"
        assert by_doi[base] != model_id


def test_every_recorded_mismatch_is_still_in_the_registry_with_a_doi(baseline, kraken):
    """The pin. An id that vanished, or lost its DOI, is one way to fix a
    mismatch and one way to hide one — so it fails here and a person decides."""
    gone = missing_from_registry(baseline, kraken)

    assert gone == [], (
        "ids recorded as mismatched that the registry no longer has. If that "
        "was a correction, re-record it: scripts/audit_registry.py "
        f"--write-baseline. {gone}")


def test_the_mismatch_set_can_shrink_but_not_grow(baseline, kraken):
    """The "never grow" half, as a statement about counts: 0 of 37 since PR #198
    (was 28 of 43). A new kraken entry nobody resolved is exactly what produced
    this issue — the block was filled in without per-entry checking — so the
    ratio is pinned too."""
    assert len(baseline) <= len(kraken)
    assert (len(kraken), len(baseline)) == (37, 0)


def test_a_removed_id_is_reported(kraken):
    assert missing_from_registry(["kraken-not-in-the-registry"], kraken) == \
        ["kraken-not-in-the-registry"]


def test_the_doi_that_publishes_no_weights_is_not_served(kraken):
    """zenodo.18732245 (MiDRASH Geniza) ships a metadata JSON and no .mlmodel, so
    kraken has nothing to load. Advertising it costs every caller a round trip to
    find that out — the lesson of #30, applied again in PR #198."""
    records = {normalize_zenodo_id(str(s.zenodo_id)) for s in kraken}

    assert WITHOUT_WEIGHTS not in records


def test_an_id_that_lost_its_doi_is_reported(kraken):
    """Dropping the DOI makes the entry unauditable rather than correct."""
    class _NoDoi:
        id = INZIGKOFEN[0]
        zenodo_id = None

    others = [s for s in kraken if s.id != INZIGKOFEN[0]]

    assert missing_from_registry([INZIGKOFEN[0]], [*others, _NoDoi()]) == [INZIGKOFEN[0]]


# ── the duplicates, which need no network ───────────────────────────────────
def test_no_doi_is_claimed_by_two_ids(kraken):
    """Five were, each under an underscore and a hyphen spelling of one name, so
    a caller could pick either and not know they were one model. PR #198 kept one
    id per record."""
    assert duplicate_dois(kraken) == {}


def test_the_registry_lists_exactly_the_models_it_serves(kraken):
    """It listed 43 entries for 38 records before PR #198."""
    unique = {normalize_zenodo_id(str(s.zenodo_id)) for s in kraken}

    assert (len(kraken), len(unique)) == (37, 37)


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


def test_the_pairs_the_issue_named_are_gone(kraken):
    """Shrinking is welcome; a new pair is not. These two were the issue's own
    examples, and the hyphen spellings no longer exist at all."""
    ids = {s.id for s in kraken}

    assert not {"kraken-catmus-medieval", "kraken-printed-french"} & ids
    assert duplicate_dois(kraken) == {}


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
    assert "37 entries for kraken, 37 with a DOI, 37 distinct records" in printed
    assert "0 ids recorded as mismatched" in printed


def test_the_offline_run_reports_no_duplicates(capsys):
    """It listed five pairs before PR #198; the section is now absent rather than
    empty, so a returning pair is visible in the output as well as in the tests."""
    offline("kraken")

    printed = capsys.readouterr().out
    assert "same DOI under more than one id" not in printed
    assert "kraken-printed-french" not in printed


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
    assert "37 distinct records" in result.stdout


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


# ── the id that cost a run, still resolvable ─────────────────────────────────
def test_the_id_that_cost_a_run_resolves_to_a_record(kraken):
    """`kraken-medieval_generic_b` was in `config/models.yaml` and the trainer
    refused it after prepare and compile had already run — "is not a valid DOI".
    It is `kraken-prima` since PR #198 renamed the registry after the records its
    DOIs load.

    The assertion came from `tests/test_training_base_models.py`, which went with
    the training package (#207). Its subject was never the trainer: it is this
    repository's own registry, and whether an id in it names weights. Stated
    without `resolve_base_model`, which lives in the other repo now.
    """
    by_id = {s.id: s for s in kraken}

    assert "kraken-prima" in by_id, "renamed away without a successor"
    assert "kraken-medieval_generic_b" not in by_id, "the old name came back"
    assert str(by_id["kraken-prima"].zenodo_id).startswith("10.5281/zenodo.")
