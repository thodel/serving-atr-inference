""""New" means unseen, not "not served" (#113).

`scripts/discover_models.py` computed "new" as "absent from models.yaml", and
models.yaml holds six HF repos against 208 candidates. So 206 were new, every
Monday, for ever — the script wrote no state. Nobody triaged the table, because
triaging it had no effect on the next one.

These pin the three things #113 says it is done when:

1. two runs over an unchanged Hub report **0 new**;
2. a candidate rejected with a reason does not come back;
3. a candidate whose upstream moved does come back, and the report says what
   moved.

And the two that make the file reviewable rather than a cache: a verdict
carries who made it, and `rejected` carries a reason — because "we looked at
this and said no" and "nobody has looked at this yet" are the distinction the
old report could not express at all.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atr_serving import discoveries  # noqa: E402
from atr_serving.discoveries import (  # noqa: E402
    CatalogueError,
    Entry,
    Observation,
    decide,
    dump,
    key_for,
    load,
    save,
    triage,
)
from scripts.discover_models import (  # noqa: E402
    DiscoveryReport,
    HFModel,
    ZenodoRecord,
    format_report_markdown,
    observations,
    report_to_json,
    triage_run,
)

MONDAY = "2026-10-05"
NEXT_MONDAY = "2026-10-12"
LATER = "2026-10-19"


def hf(name: str, downloads: int = 10, modified: str = "2026-09-01T00:00:00Z"):
    return Observation(id=name, source="hf", last_modified=modified,
                       downloads=downloads)


def decided(name: str, verdict: str, **kw) -> Entry:
    base = dict(id=name, source="hf", first_seen=MONDAY, last_seen=MONDAY,
                verdict=verdict, by="tobias", at=MONDAY,
                last_modified="2026-09-01T00:00:00Z", downloads=10)
    base.update(kw)
    return Entry(**base)


def catalogue_of(*entries: Entry) -> dict[str, Entry]:
    return {e.key: e for e in entries}


# ── "new" means unseen ──────────────────────────────────────────────────────
def test_the_first_run_sees_everything():
    result = triage({}, [hf("a/one"), hf("b/two")], today=MONDAY)

    assert [e.id for e in result.unseen] == ["a/one", "b/two"]
    assert result.counts["new"] == 2


def test_the_second_run_over_an_unchanged_hub_sees_nothing_new():
    """#113's first acceptance criterion, and the whole defect in one test."""
    first = triage({}, [hf("a/one"), hf("b/two")], today=MONDAY)

    second = triage(first.entries, [hf("a/one"), hf("b/two")], today=NEXT_MONDAY)

    assert second.unseen == []
    assert [e.id for e in second.undecided] == ["a/one", "b/two"]


def test_an_undecided_candidate_is_not_news_but_is_still_asked_about():
    """The honest middle: it is not new, and it is not settled either. The old
    report could only say "new", so a three-week-old row looked like an arrival."""
    first = triage({}, [hf("a/one")], today=MONDAY)

    second = triage(first.entries, [hf("a/one")], today=NEXT_MONDAY)

    assert second.unseen == []
    assert second.needs_a_decision == second.undecided
    assert second.entries[key_for("hf", "a/one")].first_seen == MONDAY
    assert second.entries[key_for("hf", "a/one")].last_seen == NEXT_MONDAY


def test_the_source_qualifies_the_key():
    """A Zenodo record is a number and nothing stops an HF repo being named
    one, so the two id spaces must not share a namespace."""
    result = triage({}, [Observation(id="7516057", source="hf"),
                         Observation(id="7516057", source="zenodo")],
                    today=MONDAY)

    assert len(result.entries) == 2
    assert set(result.entries) == {"hf:7516057", "zenodo:7516057"}


# ── a rejection sticks ──────────────────────────────────────────────────────
def test_a_rejected_candidate_does_not_come_back():
    """#113's second acceptance criterion."""
    catalogue = catalogue_of(decided("a/one", "rejected", reason="line-level only"))

    result = triage(catalogue, [hf("a/one")], today=NEXT_MONDAY)

    assert result.unseen == []
    assert result.undecided == []
    assert result.changed == []
    assert result.needs_a_decision == []


@pytest.mark.parametrize("verdict", ["watching", "rejected", "adopted"])
def test_no_decided_candidate_is_asked_about_again(verdict):
    catalogue = catalogue_of(decided("a/one", verdict, reason="because"))

    result = triage(catalogue, [hf("a/one")], today=NEXT_MONDAY)

    assert result.needs_a_decision == []


def test_a_decided_candidate_still_has_its_last_seen_updated():
    """So that "nothing matches this any more" stays answerable for it."""
    catalogue = catalogue_of(decided("a/one", "watching"))

    result = triage(catalogue, [hf("a/one")], today=NEXT_MONDAY)

    assert result.entries[key_for("hf", "a/one")].last_seen == NEXT_MONDAY


# ── changed is a finding ────────────────────────────────────────────────────
def test_a_new_upstream_version_brings_a_rejection_back():
    """#113's third criterion. The rejection was a judgement about the model as
    it was, so a model that moved is a different question."""
    catalogue = catalogue_of(decided("a/one", "rejected", reason="line-level only"))

    result = triage(catalogue, [hf("a/one", modified="2026-10-01T00:00:00Z")],
                    today=NEXT_MONDAY)

    assert len(result.changed) == 1
    assert result.changed[0].entry.verdict == "rejected"
    assert "2026-09-01T00:00:00Z → 2026-10-01T00:00:00Z" in result.changed[0].why


def test_an_order_of_magnitude_in_downloads_is_a_change():
    catalogue = catalogue_of(decided("a/one", "watching", downloads=11))

    result = triage(catalogue, [hf("a/one", downloads=110)], today=NEXT_MONDAY)

    assert "downloads 11 → 110" in result.changed[0].why


def test_downloads_creeping_up_is_not():
    """11 → 19 is not news about a model's reception; the old report had no way
    to say either, and reported every row every week instead."""
    catalogue = catalogue_of(decided("a/one", "watching", downloads=11))

    result = triage(catalogue, [hf("a/one", downloads=19)], today=NEXT_MONDAY)

    assert result.changed == []


def test_a_change_is_reported_once_not_every_week_after():
    """The signals are updated when they move, not only when a verdict is
    recorded — otherwise one upstream release becomes a weekly finding."""
    catalogue = catalogue_of(decided("a/one", "rejected", reason="no"))

    second = triage(catalogue, [hf("a/one", modified="2026-10-01T00:00:00Z")],
                    today=NEXT_MONDAY)
    third = triage(second.entries, [hf("a/one", modified="2026-10-01T00:00:00Z")],
                   today=LATER)

    assert len(second.changed) == 1
    assert third.changed == []


def test_an_undecided_candidate_that_moved_is_not_a_change():
    """It is still awaiting a verdict either way; reporting it twice would put
    it in two tables and make the counts not add up."""
    first = triage({}, [hf("a/one")], today=MONDAY)

    second = triage(first.entries, [hf("a/one", modified="2026-10-01T00:00:00Z")],
                    today=NEXT_MONDAY)

    assert second.changed == []
    assert second.undecided[0].id == "a/one"


def test_a_source_that_reports_no_signal_is_not_a_change():
    """Absent is not zero. Zenodo records carry no download count, and a
    missing field must not read as "it dropped to nothing"."""
    catalogue = catalogue_of(Entry(id="7516057", source="zenodo",
                                   first_seen=MONDAY, last_seen=MONDAY,
                                   verdict="rejected", reason="duplicate",
                                   by="tobias", at=MONDAY))

    result = triage(catalogue, [Observation(id="7516057", source="zenodo")],
                    today=NEXT_MONDAY)

    assert result.changed == []


# ── nothing is deleted ──────────────────────────────────────────────────────
def test_a_candidate_a_query_stopped_matching_keeps_its_verdict():
    """Deleting it would make it new again the next time a search term changes,
    and the verdict was the expensive part."""
    catalogue = catalogue_of(decided("a/one", "rejected", reason="no"))

    result = triage(catalogue, [], today=NEXT_MONDAY)

    assert [e.id for e in result.missing] == ["a/one"]
    assert result.entries[key_for("hf", "a/one")].verdict == "rejected"


def test_triage_does_not_touch_the_catalogue_it_was_given():
    """The caller writes the result, after the report rendered — so a run that
    fails in between must not have half-updated the file."""
    catalogue = catalogue_of(decided("a/one", "watching"))

    triage(catalogue, [hf("a/one", modified="2026-10-01T00:00:00Z")],
           today=NEXT_MONDAY)

    assert catalogue[key_for("hf", "a/one")].last_seen == MONDAY


# ── the file ────────────────────────────────────────────────────────────────
def test_a_round_trip_keeps_every_field(tmp_path):
    catalogue = catalogue_of(decided("a/one", "rejected", reason="line-level only"),
                             decided("b/two", "adopted"))
    path = tmp_path / "discovered.yaml"

    save(catalogue, path)

    assert load(path) == catalogue


def test_an_absent_file_is_an_empty_catalogue(tmp_path):
    assert load(tmp_path / "nope.yaml") == {}


def test_the_file_explains_itself(tmp_path):
    """It is hand-edited — that is how a verdict is normally recorded — so the
    rules have to be in it and survive every write."""
    text = dump(catalogue_of(decided("a/one", "watching")))

    assert text.startswith("#")
    assert "--decide" in text
    assert "required for 'rejected'" in text


def test_a_run_that_learned_nothing_writes_no_diff(tmp_path):
    """Sorted and stripped of empty fields, because this file is reviewed in a
    diff and a reordering would be read as a change."""
    catalogue = catalogue_of(decided("b/two", "watching"), decided("a/one", "new",
                                                                   by=None, at=None))

    assert dump(catalogue) == dump(dict(reversed(list(catalogue.items()))))


def test_nothing_empty_is_written(tmp_path):
    text = dump(catalogue_of(Entry(id="a/one", source="hf", first_seen=MONDAY,
                                   last_seen=MONDAY)))
    row = yaml.safe_load(text)["candidates"][0]

    assert "reason" not in row and "by" not in row
    assert row == {"id": "a/one", "source": "hf", "first_seen": MONDAY,
                   "last_seen": MONDAY, "verdict": "new"}


# ── the file is refused rather than overwritten ─────────────────────────────
def test_a_rejection_without_a_reason_is_refused(tmp_path):
    """The thing this module exists to prevent: a rejection nobody can learn
    from is a decision the next reader has to make again."""
    path = tmp_path / "discovered.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "candidates": [
        {"id": "a/one", "source": "hf", "first_seen": MONDAY, "last_seen": MONDAY,
         "verdict": "rejected", "by": "tobias", "at": MONDAY}]}), encoding="utf-8")

    with pytest.raises(CatalogueError, match="needs a reason"):
        load(path)


def test_a_decision_nobody_made_is_refused(tmp_path):
    path = tmp_path / "discovered.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "candidates": [
        {"id": "a/one", "source": "hf", "first_seen": MONDAY, "last_seen": MONDAY,
         "verdict": "adopted"}]}), encoding="utf-8")

    with pytest.raises(CatalogueError, match="needs a 'by'"):
        load(path)


@pytest.mark.parametrize("bad,match", [
    ({"verdict": "maybe"}, "is not one of"),
    ({"source": "github"}, "is not one of"),
])
def test_an_unknown_value_is_refused(tmp_path, bad, match):
    row = {"id": "a/one", "source": "hf", "first_seen": MONDAY,
           "last_seen": MONDAY, "verdict": "new", **bad}
    path = tmp_path / "discovered.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "candidates": [row]}),
                    encoding="utf-8")

    with pytest.raises(CatalogueError, match=match):
        load(path)


def test_the_refusal_names_the_entry(tmp_path):
    """A hand-edited file fails at the line somebody edited, not later with a
    KeyError somewhere else."""
    path = tmp_path / "discovered.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "candidates": [
        {"id": "a/one", "source": "hf", "first_seen": MONDAY, "last_seen": MONDAY},
        {"id": "b/two", "source": "hf", "first_seen": MONDAY, "last_seen": MONDAY,
         "verdict": "maybe"}]}), encoding="utf-8")

    with pytest.raises(CatalogueError, match=r"hf:b/two"):
        load(path)


def test_a_missing_field_is_refused(tmp_path):
    path = tmp_path / "discovered.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "candidates": [
        {"id": "a/one", "source": "hf"}]}), encoding="utf-8")

    with pytest.raises(CatalogueError, match="first_seen, last_seen"):
        load(path)


def test_a_future_catalogue_version_is_refused(tmp_path):
    path = tmp_path / "discovered.yaml"
    path.write_text(yaml.safe_dump({"version": 2, "candidates": []}), encoding="utf-8")

    with pytest.raises(CatalogueError, match="version 2"):
        load(path)


# ── recording a verdict ─────────────────────────────────────────────────────
def test_decide_records_who_and_when():
    catalogue = triage({}, [hf("a/one")], today=MONDAY).entries

    entry = decide(catalogue, "hf:a/one", "rejected", by="tobias", at=NEXT_MONDAY,
                   reason="line-level only, no page model")

    assert (entry.verdict, entry.by, entry.at) == ("rejected", "tobias", NEXT_MONDAY)
    assert catalogue["hf:a/one"].reason == "line-level only, no page model"


def test_decide_refuses_a_rejection_without_a_reason():
    catalogue = triage({}, [hf("a/one")], today=MONDAY).entries

    with pytest.raises(CatalogueError, match="needs a reason"):
        decide(catalogue, "hf:a/one", "rejected", by="tobias", at=MONDAY)

    assert catalogue["hf:a/one"].verdict == "new"


def test_decide_names_the_prefix_when_the_key_is_unknown():
    """The likeliest mistake is `--decide CATMuS/medieval` without `hf:`."""
    with pytest.raises(CatalogueError, match="hf, zenodo"):
        decide({}, "CATMuS/medieval", "watching", by="tobias", at=MONDAY)


def test_a_verdict_survives_the_next_run():
    catalogue = triage({}, [hf("a/one")], today=MONDAY).entries
    decide(catalogue, "hf:a/one", "rejected", by="tobias", at=MONDAY, reason="no")

    result = triage(catalogue, [hf("a/one")], today=NEXT_MONDAY)

    assert result.needs_a_decision == []
    assert result.entries["hf:a/one"].reason == "no"


# ── the script ──────────────────────────────────────────────────────────────
def report_with(*, hf_models=(), zenodo=()):
    report = DiscoveryReport(new_hf_models=list(hf_models),
                             new_zenodo_models=list(zenodo))
    report.hf_candidates = list(hf_models)
    report.zenodo_candidates = list(zenodo)
    return report


MODEL = HFModel(id="a/one", downloads=42, last_modified="2026-09-01T00:00:00Z",
                tags=["kraken"])
RECORD = ZenodoRecord(zenodo_id="7516057", title="CATMuS Medieval",
                      doi="10.5281/zenodo.7516057", keywords=[], zenodo_url="",
                      last_modified="2026-09-01T00:00:00Z")


def test_observations_carry_both_sources():
    seen = observations(report_with(hf_models=[MODEL], zenodo=[RECORD]))

    assert {(o.source, o.id) for o in seen} == {("hf", "a/one"), ("zenodo", "7516057")}
    assert seen[0].downloads == 42
    assert seen[1].last_modified == "2026-09-01T00:00:00Z"


def test_a_zenodo_record_without_an_updated_field_reports_no_signal():
    """`ZenodoRecord.last_modified` defaults to "", and "" must not be recorded
    as a timestamp that then differs from every real one."""
    bare = ZenodoRecord(zenodo_id="1", title="t", doi="", keywords=[], zenodo_url="")

    assert observations(report_with(zenodo=[bare]))[0].last_modified is None


def test_the_report_leads_with_the_verdict_counts(tmp_path):
    report = report_with(hf_models=[MODEL])
    report.triage = triage_run(report, catalogue=tmp_path / "c.yaml", today=MONDAY)

    rendered = format_report_markdown(report)

    assert "## Where the candidates stand" in rendered
    assert "| new | 1 |" in rendered
    assert "1 unseen** this run" in rendered
    assert "## Unseen before this run" in rendered
    assert "a/one" in rendered


def test_the_report_says_zero_when_the_answer_is_zero(tmp_path):
    path = tmp_path / "c.yaml"
    report = report_with(hf_models=[MODEL])
    save(triage_run(report, catalogue=path, today=MONDAY).entries, path)

    again = report_with(hf_models=[MODEL])
    again.triage = triage_run(again, catalogue=path, today=NEXT_MONDAY)
    rendered = format_report_markdown(again)

    assert "0 unseen** this run" in rendered
    assert "## Awaiting a verdict" in rendered
    assert "a/one" in rendered       # still asked about, not called new


def test_the_json_carries_the_catalogue_for_the_endpoint(tmp_path):
    """#116 serves this without scraping GitHub, which is half the reason the
    state is a file."""
    report = report_with(hf_models=[MODEL])
    report.triage = triage_run(report, catalogue=tmp_path / "c.yaml", today=MONDAY)

    body = report_to_json(report)

    assert body["catalogue"]["counts"]["new"] == 1
    assert body["catalogue"]["unseen"][0]["id"] == "a/one"
    assert body["catalogue"]["changed"] == []


def test_the_json_says_so_when_there_was_no_catalogue():
    assert report_to_json(report_with(hf_models=[MODEL]))["catalogue"] is None


def test_the_shipped_catalogue_is_readable():
    """It is in the repo so that the format is in review from the start."""
    entries = load(discoveries.CATALOGUE)

    assert isinstance(entries, dict)
    assert discoveries.CATALOGUE.name == "discovered.yaml"
    assert discoveries.CATALOGUE.parent.name == "config"


# ── the Zenodo half ─────────────────────────────────────────────────────────
#
# #113 asks for this to be fixed "while here", on the strength of an error it
# quotes:
#
#     Zenodo community 'htr-model' page 1: 400 ... size=200&...
#
# Two of its three readings of that line do not hold against the code, so only
# the third is acted on here:
#
# * ``communities=htr-model`` is not sent and never was. The second element of
#   each query pair is a label for the error message; the two Zenodo-wide
#   searches carry no ``communities`` at all.
# * The 400 was ``size=200``. Zenodo caps an anonymous page at 25, and
#   ``ZENODO_PAGE_SIZE`` was brought to 25 on 17.09.2026 (#66) — before this
#   issue was written, after that error was captured.
# * ``type=dataset`` is real, and it is a filter: it can only remove records. A
#   kraken model is deposited as a dataset by some and as software or other by
#   others, so it was dropping model records by construction.

from unittest.mock import MagicMock  # noqa: E402

import requests  # noqa: E402

from scripts.discover_models import ZENODO_COMMUNITIES, discover_zenodo_models  # noqa: E402


def zenodo_session(hits=()):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"hits": {"hits": list(hits)}, "links": {}}
    session = MagicMock(spec=requests.Session)
    session.get.return_value = response
    return session


def sent_params(session) -> list[dict]:
    return [call.kwargs["params"] for call in session.get.call_args_list]


def test_no_query_filters_the_resource_type_by_default():
    """The filter could only remove, and what it removed was model records."""
    session = zenodo_session()

    discover_zenodo_models(session)

    assert all("type" not in params for params in sent_params(session))


def test_a_resource_type_can_be_asked_for_again():
    """Kept as a setting because it cannot be checked from a container that
    cannot reach zenodo.org — whoever runs it on a box with internet can narrow
    it in one flag and read the difference in the counts."""
    session = zenodo_session()

    discover_zenodo_models(session, ["software", "dataset"])

    assert all(params["type"] == ["software", "dataset"]
               for params in sent_params(session))


def test_the_error_label_is_not_a_query_parameter():
    """#113 read "Zenodo community 'htr-model'" as a community being sent. The
    two Zenodo-wide searches send no `communities` at all."""
    session = zenodo_session()

    discover_zenodo_models(session)

    communities = {params.get("communities") for params in sent_params(session)}

    assert communities == {*ZENODO_COMMUNITIES, None}   # None: the two over all
    assert "htr-model" not in communities
    assert "htr" not in communities


def test_the_page_size_is_the_anonymous_cap():
    """size=200 was the 400 in #113's quoted error; 25 is the cap and was set
    on 17.09 (#66), before the issue and after the capture."""
    session = zenodo_session()

    discover_zenodo_models(session)

    assert all(params["size"] == 25 for params in sent_params(session))


def test_zero_records_and_no_error_is_itself_an_error():
    """#113 ranks silently returning 0 as the worst of three options: 39 of the
    registry's entries are Zenodo DOIs, so "Zenodo has nothing" is never the
    likely reading."""
    records, error = discover_zenodo_models(zenodo_session())

    assert records == []
    assert error is not None
    assert "broken query" in error
    assert "zenodo.org/api/records" in error     # how to check it on a box


def test_a_record_that_came_back_is_not_an_error():
    hit = {"id": 7516057, "updated": "2026-09-01T00:00:00Z",
           "metadata": {"title": "CATMuS Medieval", "doi": "10.5281/zenodo.7516057"}}

    records, error = discover_zenodo_models(zenodo_session([hit]))

    assert [r.zenodo_id for r in records] == ["7516057"]
    assert error is None


def test_a_record_carries_its_updated_timestamp():
    """So that "this moved since you rejected it" is answerable for Zenodo as
    it is for the Hub."""
    hit = {"id": 7516057, "updated": "2026-09-01T00:00:00Z", "metadata": {}}

    records, _ = discover_zenodo_models(zenodo_session([hit]))

    assert records[0].last_modified == "2026-09-01T00:00:00Z"
