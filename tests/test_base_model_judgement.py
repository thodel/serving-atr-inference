"""A candidate is only interesting if it can be a base_model (#115).

Discovery ranked by download count and text match, so
`microsoft/trocr-small-handwritten` (450 981 downloads) sat in the same
undifferentiated table as any `vision-encoder-decoder` repo that matched the
string `trocr`. The report answered "does this exist and does it mention HTR".

These pin the question that is worth a weekly notification — can kraken, TrOCR
or the VLM backend load this, and if not, why not — and the one thing that
makes it safe to answer by machine:

**a reason that is a fact about the repository may be recorded as a verdict; a
reason that is a fact about this run may not.** Otherwise a rate limit on a
Monday morning becomes a permanent rejection, and #113's catalogue is worth
having precisely because a verdict sticks.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atr_serving.base_models import (  # noqa: E402
    QLORA_CEILING,
    QLORA_MEASUREMENT,
    RepoFacts,
    judge,
)
from atr_serving.discoveries import CatalogueError, Entry, Observation, triage  # noqa: E402
from scripts.discover_models import (  # noqa: E402
    AUTOMATIC_BY,
    HF_SEARCH_TERMS,
    DiscoveryReport,
    fetch_repo_facts,
    format_arrivals,
    judge_unseen,
)

TODAY = "2026-10-05"

SAFE = ("model.safetensors", "config.json")
TOKENIZER = ("tokenizer.json",)


def facts(**kw) -> RepoFacts:
    base = dict(id="owner/model", files=SAFE)
    base.update(kw)
    return RepoFacts(**base)


# ── kraken ──────────────────────────────────────────────────────────────────
def test_an_mlmodel_is_a_kraken_base():
    verdict = judge(facts(files=("german_print.mlmodel", "README.md")))

    assert verdict.backend == "kraken"
    assert "german_print.mlmodel" in verdict.evidence


def test_the_mlmodel_decides_before_anything_else():
    """It is unambiguous, so it is checked before the heuristics that are not."""
    verdict = judge(facts(files=("m.mlmodel", "model.safetensors", "tokenizer.json"),
                          architectures=("VisionEncoderDecoderModel",)))

    assert verdict.backend == "kraken"


# ── TrOCR ───────────────────────────────────────────────────────────────────
def test_a_vision_encoder_decoder_with_a_tokenizer_is_a_trocr_base():
    verdict = judge(facts(files=SAFE + TOKENIZER,
                          architectures=("VisionEncoderDecoderModel",)))

    assert verdict.backend == "trocr"
    assert "tokenizer.json" in verdict.evidence


def test_the_architecture_is_read_from_the_config_when_that_is_where_it_is():
    verdict = judge(facts(files=SAFE + TOKENIZER,
                          config={"architectures": ["VisionEncoderDecoderModel"]}))

    assert verdict.backend == "trocr"


@pytest.mark.parametrize("tokenizer", ["tokenizer.json", "tokenizer_config.json",
                                       "vocab.json", "spiece.model", "vocab.txt"])
def test_any_tokenizer_will_do(tokenizer):
    verdict = judge(facts(files=SAFE + (tokenizer,),
                          architectures=("VisionEncoderDecoderModel",)))

    assert verdict.backend == "trocr"


def test_a_tokenizer_in_a_subdirectory_still_counts():
    verdict = judge(facts(files=SAFE + ("decoder/tokenizer.json",),
                          architectures=("VisionEncoderDecoderModel",)))

    assert verdict.backend == "trocr"


# ── the disqualifiers, and that they are facts ──────────────────────────────
def test_a_model_card_with_no_weights_is_rejected():
    """#115's own example: written once, never re-asked."""
    verdict = judge(facts(files=("README.md", ".gitattributes")))

    assert verdict.backend is None
    assert verdict.reason == "no weights file, only a model card"
    assert verdict.automatic


def test_an_encoder_only_backbone_is_rejected():
    """These match "vision" searches and produce features, not text."""
    verdict = judge(facts(architectures=("ViTModel",)))

    assert verdict.backend is None
    assert "no decoder" in verdict.reason
    assert verdict.automatic


def test_a_vision_encoder_decoder_without_a_tokenizer_is_rejected():
    """Nothing to resize for a new alphabet, so it cannot be fine-tuned here."""
    verdict = judge(facts(architectures=("VisionEncoderDecoderModel",)))

    assert "no alphabet to resize" in verdict.reason
    assert verdict.automatic


def test_a_vlm_above_the_band_is_rejected():
    verdict = judge(facts(parameters=32_000_000_000,
                          config={"vision_config": {"depth": 32},
                                  "chat_template": "{{ x }}"}))

    assert verdict.backend is None
    assert "32.0B parameters" in verdict.reason
    assert verdict.automatic


# ── the band cites its measurement ──────────────────────────────────────────
def test_the_band_carries_the_measurement_it_came_from():
    """#115's third done-when: "too big" has to be a fact with a provenance."""
    verdict = judge(facts(parameters=32_000_000_000,
                          config={"vision_config": {}, "chat_template": "x"}))

    assert QLORA_MEASUREMENT in verdict.reason
    assert "28.4 GiB" in verdict.reason
    assert "46 GiB A40" in verdict.reason
    assert "batch_size 1" in verdict.reason


def test_the_ceiling_is_above_the_measured_model_not_at_it():
    """~8B was measured *near* the ceiling at 28.4 of 46 GiB, so an 8.4B
    variant is still worth a look and a 13B is not."""
    assert 8_000_000_000 < QLORA_CEILING < 13_000_000_000


def test_a_vlm_inside_the_band_is_a_candidate():
    verdict = judge(facts(parameters=8_000_000_000,
                          config={"vision_config": {}, "chat_template": "x"}))

    assert verdict.backend == "vllm"
    assert "8.0B" in verdict.evidence


def test_a_chat_template_file_counts_as_one():
    verdict = judge(facts(files=SAFE + ("chat_template.json",),
                          parameters=4_000_000_000,
                          config={"vision_config": {}}))

    assert verdict.backend == "vllm"


def test_a_vision_tower_can_come_from_the_tag():
    """A small VLM that does not spell "OCR" in its card is exactly what the
    old search could not see."""
    verdict = judge(facts(tags=("image-text-to-text",), parameters=2_000_000_000,
                          config={"chat_template": "x"}))

    assert verdict.backend == "vllm"
    assert "tag image-text-to-text" in verdict.evidence


# ── what may not be recorded as a verdict ───────────────────────────────────
def test_an_unpublished_parameter_count_is_not_a_rejection():
    """Absent is not large. The band cannot be applied, so a person decides."""
    verdict = judge(facts(config={"vision_config": {}, "chat_template": "x"}))

    assert verdict.backend is None
    assert not verdict.automatic
    assert "does not publish" in verdict.reason


def test_a_hub_that_did_not_answer_is_not_a_rejection():
    """The property this whole design rests on: a rate limit on a Monday must
    not become a permanent verdict."""
    verdict = judge(RepoFacts(id="owner/model", fetched=False))

    assert verdict.backend is None
    assert not verdict.automatic
    assert "not a verdict" in verdict.reason


def test_nothing_matching_is_not_a_rejection():
    """A heuristic may be wrong about a repo it does not recognise; it is not
    wrong about a repo with no weights."""
    verdict = judge(facts(architectures=("SomethingNobodyHasHeardOf",)))

    assert verdict.backend is None
    assert not verdict.automatic
    assert "nothing identifies a backend" in verdict.reason


def test_a_vision_tower_without_a_chat_template_is_not_a_rejection():
    verdict = judge(facts(parameters=4_000_000_000, config={"vision_config": {}}))

    assert not verdict.automatic
    assert "no chat template" in verdict.reason


# ── the hub call ────────────────────────────────────────────────────────────
class FakeHub:
    """A hub that answers for the ids it was given and fails for the rest."""

    def __init__(self, repos: dict, fail: Exception | None = None) -> None:
        self.repos = repos
        self.fail = fail
        self.asked: list[str] = []

    def get(self, url, **kwargs):
        model_id = url.split("/api/models/", 1)[1]
        self.asked.append(model_id)
        if self.fail is not None:
            raise self.fail

        class Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        return Response(self.repos.get(model_id, {}))


def test_the_hub_answer_becomes_facts():
    hub = FakeHub({"owner/model": {
        "siblings": [{"rfilename": "model.safetensors"},
                     {"rfilename": "tokenizer.json"}],
        "tags": ["image-text-to-text"],
        "config": {"architectures": ["Qwen3VLForConditionalGeneration"],
                   "vision_config": {"depth": 24}},
        "safetensors": {"total": 8_000_000_000},
    }})

    got = fetch_repo_facts(hub, "owner/model")

    assert got.files == ("model.safetensors", "tokenizer.json")
    assert got.architectures == ("Qwen3VLForConditionalGeneration",)
    assert got.parameters == 8_000_000_000
    assert got.fetched


def test_a_hub_failure_is_facts_with_fetched_false_and_no_exception():
    import requests

    hub = FakeHub({}, fail=requests.exceptions.ConnectionError("nope"))

    got = fetch_repo_facts(hub, "owner/model")

    assert got.fetched is False
    assert got.parameters is None


def test_a_repo_without_safetensors_metadata_reports_no_count():
    got = fetch_repo_facts(FakeHub({"owner/model": {"siblings": []}}), "owner/model")

    assert got.parameters is None


# ── judging a run ───────────────────────────────────────────────────────────
def observed(*ids) -> list[Observation]:
    return [Observation(id=i, source="hf", downloads=1) for i in ids]


def judged_run(judgements: dict, *, observations=None):
    """A triage of `observations`, judged with a stubbed classifier."""
    result = triage({}, observations or observed(*judgements), today=TODAY)
    calls: list[str] = []

    def fetch(_session, model_id):
        calls.append(model_id)
        return judgements[model_id]

    found = judge_unseen(None, result, today=TODAY, fetch=fetch)
    return result, found, calls


def test_a_usable_candidate_keeps_its_backend_and_no_verdict():
    result, _, _ = judged_run({"a/one": facts(files=("m.mlmodel",))})

    entry = result.entries["hf:a/one"]
    assert entry.backend == "kraken"
    assert entry.verdict == "new"


def test_an_unusable_candidate_is_rejected_with_its_reason():
    result, _, _ = judged_run({"a/one": facts(files=("README.md",))})

    entry = result.entries["hf:a/one"]
    assert entry.verdict == "rejected"
    assert entry.reason == "no weights file, only a model card"


def test_the_machine_signs_its_own_verdicts():
    """So a reader can tell them from their own. #113 requires a `by`; this is
    which `by`."""
    result, _, _ = judged_run({"a/one": facts(files=("README.md",))})

    assert result.entries["hf:a/one"].by == AUTOMATIC_BY
    assert result.entries["hf:a/one"].at == TODAY


def test_a_machine_rejection_does_not_come_back():
    """The join with #113: this is the whole point of judging automatically."""
    result, _, _ = judged_run({"a/one": facts(files=("README.md",))})

    again = triage(result.entries, observed("a/one"), today="2026-10-12")

    assert again.needs_a_decision == []


def test_an_unreachable_hub_leaves_the_candidate_undecided():
    result, _, _ = judged_run({"a/one": RepoFacts(id="a/one", fetched=False)})

    assert result.entries["hf:a/one"].verdict == "new"
    assert result.entries["hf:a/one"].backend is None


def test_an_unreachable_hub_candidate_is_asked_about_again_next_week():
    """Which is the behaviour that makes the automatic rejection safe."""
    result, _, _ = judged_run({"a/one": RepoFacts(id="a/one", fetched=False)})

    again = triage(result.entries, observed("a/one"), today="2026-10-12")

    assert [e.id for e in again.needs_a_decision] == ["a/one"]


def test_only_the_unseen_cost_a_hub_call():
    """Judging 208 repositories a week when two are new is the same waste in a
    different place."""
    first, _, calls = judged_run({"a/one": facts(files=("m.mlmodel",))})
    assert calls == ["a/one"]

    second = triage(first.entries, observed("a/one"), today="2026-10-12")
    more: list[str] = []
    judge_unseen(None, second, today="2026-10-12",
                 fetch=lambda _s, i: more.append(i) or facts())

    assert more == []


def test_a_zenodo_candidate_is_not_judged_rather_than_misjudged():
    """There is no file listing for it here; kraken's own index (htrmopo) is
    the check #115 names, and it needs a box with internet."""
    result = triage({}, [Observation(id="7516057", source="zenodo")], today=TODAY)

    found = judge_unseen(None, result, today=TODAY,
                         fetch=lambda *a: pytest.fail("the hub was asked"))

    assert found == {}
    assert result.entries["zenodo:7516057"].verdict == "new"


def test_every_arrival_carries_a_backend_or_a_recorded_reason():
    """#115's first done-when, over one of each kind."""
    result, found, _ = judged_run({
        "a/kraken": facts(files=("m.mlmodel",)),
        "b/card": facts(files=("README.md",)),
        "c/huge": facts(parameters=70_000_000_000,
                        config={"vision_config": {}, "chat_template": "x"}),
        "d/unknown": facts(architectures=("Mystery",)),
    })

    for key, entry in result.entries.items():
        assert entry.backend or entry.reason or found[key].reason, key


# ── the catalogue refuses a backend it does not have ────────────────────────
def test_an_unknown_backend_is_refused_by_the_catalogue():
    """The field is written by the judgement and read by #116; a typo in a
    hand-edited file must not reach the endpoint as a backend."""
    from atr_serving.discoveries import dump, load

    entry = Entry(id="a/one", source="hf", first_seen=TODAY, last_seen=TODAY,
                  backend="ketos")
    path = ROOT / "tests" / "fixtures" / "_nope.yaml"
    path.write_text(dump({entry.key: entry}), encoding="utf-8")
    try:
        with pytest.raises(CatalogueError, match="backend 'ketos'"):
            load(path)
    finally:
        path.unlink()


# ── the report ──────────────────────────────────────────────────────────────
def test_the_arrivals_table_shows_what_decided_each_row():
    result, found, _ = judged_run({
        "a/kraken": facts(files=("m.mlmodel",)),
        "b/card": facts(files=("README.md",)),
    })
    report = DiscoveryReport(triage=result, judgements=found)

    table = format_arrivals(report)

    assert "**kraken**" in table
    assert "m.mlmodel" in table
    assert "rejected: no weights file, only a model card" in table


def test_a_rejected_arrival_stays_visible():
    """It is the week's work, done. Hiding it would make the report look
    emptier than the search was."""
    result, found, _ = judged_run({"b/card": facts(files=("README.md",))})
    report = DiscoveryReport(triage=result, judgements=found)

    assert "hf:b/card" in format_arrivals(report)


# ── the search widened only once rejections stick ──────────────────────────
def test_the_search_covers_the_small_vlm_space():
    """#115: adding terms without a rejection mechanism makes the report
    worse, which is why this issue and #113 are ordered."""
    terms = " ".join(HF_SEARCH_TERMS).lower()

    for name in ("smolvlm", "florence-2", "got-ocr", "qwen3-vl", "internvl"):
        assert name in terms


def test_the_search_uses_the_words_a_palaeographer_uses():
    terms = " ".join(HF_SEARCH_TERMS).lower()

    assert "manuscript transcription" in terms
    assert "medieval handwriting" in terms


def test_the_original_terms_are_still_there():
    terms = " ".join(HF_SEARCH_TERMS).lower()

    for name in ("kraken htr", "trocr", "lightonocr"):
        assert name in terms


# ── adopting verifies the name against the record (#115 × #101) ─────────────
#
# #115 is explicit: discovery is the intake path into the registry, so adopting
# a candidate has to verify what the DOI *resolves to* rather than trusting the
# name it was found under — "otherwise this issue becomes a faster way to make
# #101 worse". The check used here is #101's own `classify()`, because a second
# one would be a second answer to the same question.

from scripts.discover_models import check_the_name_against_the_record  # noqa: E402

BIFROST = "Bifrost: A Handwritten Text Recognition Model for Old Norse"


def zenodo_entry(title=BIFROST, record="15366732") -> Entry:
    return Entry(id=record, source="zenodo", first_seen=TODAY, last_seen=TODAY,
                 title=title)


def test_a_name_the_record_does_not_support_is_refused():
    """The exact row from #101: `kraken-late-medieval-german` → an Old Norse
    model. Adopting it under that name is how that table came to exist."""
    refusal = check_the_name_against_the_record(zenodo_entry(),
                                                "kraken-late-medieval-german")

    assert refusal is not None
    assert BIFROST in refusal
    assert "#101" in refusal


def test_a_name_the_record_supports_is_allowed():
    assert check_the_name_against_the_record(
        zenodo_entry(), "kraken-old-norse-bifrost") is None


def test_a_defensible_family_match_is_allowed():
    """#101's `classify()` is deliberately generous — a Persian model under an
    Arabic id is a grouping, not a mistake — and this must not be stricter than
    the audit it shares, or the two would disagree about the same pair."""
    entry = zenodo_entry(title="Printed Ottoman Base Model Trained on the "
                               "OpenITI Corpus", record="7050342")

    assert check_the_name_against_the_record(entry, "kraken-arabic-script") is None


def test_an_unresolved_record_is_refused_rather_than_waved_through():
    """No title means the check cannot run, and "cannot check" must not read as
    "checked and fine" — that is the state the whole of #101 came out of."""
    refusal = check_the_name_against_the_record(zenodo_entry(title=None),
                                                "kraken-anything")

    assert refusal is not None
    assert "audit_registry.py" in refusal


def test_the_check_is_the_audit_and_not_a_copy_of_it():
    """Identity, not similarity: #115 says "whatever check #101 produces should
    be the same one used here"."""
    import scripts.discover_models as script
    from scripts.audit_registry import classify

    source = Path(script.__file__).read_text(encoding="utf-8")

    assert "from scripts.audit_registry import classify" in source
    assert classify("kraken-late-medieval-german", BIFROST) == "mismatch"
