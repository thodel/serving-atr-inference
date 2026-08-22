"""Combining HuggingFace datasets into one corpus (#87).

The cases here are the real catalogue, not invented ones: the duplicate
Koenigsfelden projects and the republished u-17_* set are what motivated the
module, and Parzival is the concrete case where dates fit and script does not.
"""

import pytest

from atr_serving.training.corpus_plan import (
    MEDIEVAL_GERMAN,
    Candidate,
    CorpusPlanError,
    Target,
    job_request,
    language_score,
    parse_card,
    parse_period,
    parse_projects,
    period_score,
    plan_corpus,
    score_candidate,
    script_score,
)

CARD = """Geographical scope: Switzerland<br>Period: 1400-1550<br>Languages: Middle High German, Early Modern German<br>Type of document: Protocols<br>Provenance: State Archive of Zurich"""


def cand(repo, pages=1000, gb=10.0, period=(1400, 1550),
         languages=("middle high german",), doc_type="protocols", projects=()):
    return Candidate(repo=repo, pages=pages, gb=gb, period=period,
                     languages=languages, doc_type=doc_type, projects=tuple(projects))


class TestParsing:
    def test_a_real_card_yields_every_field(self):
        c = parse_card("dh-unibe/rats", CARD, pages=9885, gb=65.9,
                       projects=["a", "b"])
        assert c.period == (1400, 1550)
        assert "middle high german" in c.languages
        assert c.doc_type == "protocols"
        assert c.pages_per_project == 9885 / 2

    @pytest.mark.parametrize("text,expected", [
        ("Period: 1291-1550", (1291, 1550)),
        ("Period: 1200–1500", (1200, 1500)),      # en dash
        ("Period: 1550-1400", (1400, 1550)),      # reversed
        ("no period here", None),
    ])
    def test_period_forms(self, text, expected):
        assert parse_period(text) == expected

    def test_a_card_without_the_summary_block_still_parses(self):
        """koenigsfelden-charters-part-3 has tags but no `Key: value` summary."""
        c = parse_card("x", "Based on the Koenigsfelden Data Set.", 223, 0.6)
        assert c.period is None and c.languages == ()


class TestScoring:
    def test_a_dataset_inside_the_target_scores_full_period(self):
        assert period_score(cand("x", period=(1400, 1500)), MEDIEVAL_GERMAN) == 1.0

    def test_half_outside_scores_half(self):
        """1200-1500 against 1300-1600: 200 of 300 years land inside."""
        assert period_score(cand("x", period=(1200, 1500)), MEDIEVAL_GERMAN) == pytest.approx(2 / 3)

    def test_an_unknown_period_is_not_disqualifying(self):
        assert period_score(cand("x", period=None), MEDIEVAL_GERMAN) == 0.5

    def test_latin_counts_partially_because_charters_mix_it(self):
        assert language_score(cand("x", languages=("latin",)), MEDIEVAL_GERMAN) == 0.6

    def test_flemish_does_not_count(self):
        assert language_score(cand("x", languages=("flemish",)), MEDIEVAL_GERMAN) == 0.0

    def test_a_book_hand_is_penalised(self):
        """Parzival: 1200-1500 Middle High German, and the wrong script class."""
        assert script_score(cand("x", doc_type="manuscript"), MEDIEVAL_GERMAN) == 0.3

    def test_script_outweighs_period_which_is_what_thun_measured(self):
        """A right-period book hand must not outrank a wrong-period cursive.

        TRAINING_PLAN §9c: a Kurrent base a century late beat a Textura base of
        the right period by 40 % relative.
        """
        # A cursive half outside the period beats a book hand wholly inside it.
        # Not "period does not matter": a corpus entirely in the wrong century
        # still loses, and test_a_wholly_wrong_period_still_loses pins that.
        book = score_candidate(
            cand("book", period=(1300, 1600), doc_type="manuscript"), MEDIEVAL_GERMAN)
        cursive = score_candidate(
            cand("cursive", period=(1500, 1700), doc_type="letters"), MEDIEVAL_GERMAN)
        assert cursive.score > book.score

    def test_a_wholly_wrong_period_still_loses(self):
        """The other half of the claim, so the weighting cannot drift unnoticed."""
        book = score_candidate(
            cand("book", period=(1300, 1600), doc_type="manuscript"), MEDIEVAL_GERMAN)
        far = score_candidate(
            cand("far", period=(1700, 1900), doc_type="letters"), MEDIEVAL_GERMAN)
        assert far.score < book.score

    def test_the_rejection_reason_names_the_dimension(self):
        s = score_candidate(cand("x", languages=("flemish",)), MEDIEVAL_GERMAN)
        assert "flemish" in s.why


class TestDeduplication:
    """The catalogue really does republish the same projects under new repo names."""

    def test_a_project_is_claimed_by_the_higher_scoring_dataset(self):
        shared = ["FRAD068_032_01", "FRAD068_032_02"]
        plan = plan_corpus([
            cand("kf-post-1500", pages=3222, period=(1291, 1550),
                 doc_type="documents", projects=shared + ["own_1"]),
            cand("kf-colmar", pages=223, period=(1300, 1500),
                 doc_type="documents", projects=shared),
        ], MEDIEVAL_GERMAN)
        by_repo = {s.repo: s for s in plan.selections}
        # Scores are within SCORE_TIE_BAND, so the 3222-page dataset claims them,
        # not the 223-page one that scores 0.011 higher on period overlap.
        assert by_repo["kf-post-1500"].projects == tuple(shared + ["own_1"])
        assert "kf-colmar" not in by_repo
        assert any("already covered" in why for _, why in plan.rejected)

    def test_pages_shrink_with_the_projects_that_were_dropped(self):
        plan = plan_corpus([
            cand("big", pages=1000, projects=["a", "b", "c", "d"]),
            cand("small", pages=500, projects=["a", "b", "e", "f"]),
        ], MEDIEVAL_GERMAN, max_share=1.0)
        small = next(s for s in plan.selections if s.repo == "small")
        assert small.projects == ("e", "f")
        assert small.pages == 250                    # 500/4 per project x 2 kept
        assert small.dropped_duplicates == ("a", "b")

    def test_held_out_projects_never_enter_training(self):
        """The one error no metric detects afterwards."""
        plan = plan_corpus([cand("x", pages=1000, projects=["train_1", "GT_Test"])],
                           MEDIEVAL_GERMAN, exclude_projects=["GT_Test"])
        assert plan.selections[0].projects == ("train_1",)


class TestBalance:
    def test_no_dataset_may_dominate_the_corpus(self):
        plan = plan_corpus([
            cand("huge", pages=100_000, projects=["h"]),
            cand("small", pages=1_000, projects=["s"]),
        ], MEDIEVAL_GERMAN, max_share=0.5)
        assert plan.largest_share <= 0.5 + 1e-6
        assert any("capped" in n for n in plan.notes)

    def test_a_page_budget_scales_everything_down_proportionally(self):
        plan = plan_corpus([
            cand("a", pages=6000, projects=["a"]),
            cand("b", pages=4000, projects=["b"]),
        ], MEDIEVAL_GERMAN, max_share=1.0, max_pages=1000)
        assert plan.pages <= 1000
        by = {s.repo: s.pages for s in plan.selections}
        assert by["a"] > by["b"]                      # ratio preserved

    def test_a_balanced_corpus_is_left_alone(self):
        plan = plan_corpus([
            cand("a", pages=1000, projects=["a"]),
            cand("b", pages=1000, projects=["b"]),
        ], MEDIEVAL_GERMAN, max_share=0.6)
        assert all(s.capped_from is None for s in plan.selections)


class TestPlanOutput:
    def test_estimated_lines_are_derived_from_the_measured_ratio(self):
        plan = plan_corpus([cand("x", pages=1000, projects=["p"])], MEDIEVAL_GERMAN)
        assert plan.estimated_lines == int(1000 * 0.64 * 14.8)

    def test_every_plan_says_its_numbers_are_estimates(self):
        plan = plan_corpus([cand("x", projects=["p"])], MEDIEVAL_GERMAN)
        assert any("run prepare" in n for n in plan.notes)

    def test_an_empty_result_is_an_error_not_an_empty_corpus(self):
        with pytest.raises(CorpusPlanError, match="no candidate scored"):
            plan_corpus([cand("flemish", languages=("flemish",),
                              period=(1350, 1550), doc_type="protocol")],
                        Target(period=(1300, 1600), threshold=0.9))

    def test_a_nonsense_share_is_refused(self):
        with pytest.raises(CorpusPlanError, match="max_share"):
            plan_corpus([cand("x")], MEDIEVAL_GERMAN, max_share=0)


class TestJobRequest:
    """The emitted body must be one the trainer will actually accept."""

    def plan(self, **kw):
        return plan_corpus([
            cand("dh-unibe/rats", pages=9885, projects=["v1", "v2", "v3", "eval_v"]),
            cand("dh-unibe/bull", pages=8022, period=(1530, 1600),
                 doc_type="letters", projects=()),
        ], MEDIEVAL_GERMAN, max_share=1.0, **kw)

    def test_a_dataset_without_projects_asks_for_all_of_them_and_is_capped(self):
        """all_projects requires max_pages — the contract validator enforces it."""
        body = job_request(self.plan(), engine="vllm", model_id="m")
        bull = next(d for d in body["datasets"] if d["hf_repo"].endswith("bull"))
        assert bull["all_projects"] is True and bull["max_pages"] > 0

    def test_eval_projects_attach_to_their_own_dataset_entry(self):
        plan = self.plan(exclude_projects=["eval_v"])
        body = job_request(plan, engine="vllm", model_id="m",
                           eval_repo="dh-unibe/rats", eval_projects=["eval_v"])
        rats = next(d for d in body["datasets"] if d["hf_repo"].endswith("rats"))
        assert rats["eval_projects"] == ["eval_v"]
        assert "eval_v" not in rats["train_projects"]

    def test_an_eval_repo_outside_the_corpus_is_refused(self):
        """An eval-only entry has no train_projects, and hf_source rejects that.

        Emitting one produces a request that dies at submit; refusing here says so
        while the plan can still be changed.
        """
        with pytest.raises(CorpusPlanError, match="not in the corpus"):
            job_request(self.plan(), engine="vllm", model_id="m",
                        eval_repo="dh-unibe/somewhere-else",
                        eval_projects=["GT_Thun-Test_(DEMO_TEST)"])

    def test_training_on_the_evaluation_set_is_refused(self):
        """The one error no later metric can reveal."""
        with pytest.raises(CorpusPlanError, match="also selected for training"):
            job_request(self.plan(), engine="vllm", model_id="m",
                        eval_repo="dh-unibe/rats", eval_projects=["v1"])

    def test_every_entry_declares_page_granularity(self):
        body = job_request(self.plan(), engine="vllm", model_id="m")
        assert all(d["granularity"] == "page" for d in body["datasets"])


# ── parsing projects out of a card (#87) ────────────────────────────────────
REAL_CARD = """---
dataset_info:
  config_name: default
  features:
    - name: image
tags:
- image-to-text
- htr
- pagexml
license: mit
language:
- de
- la
---

# Dataset Card for image-text_rats-und-richtebuecher_xv-xvi

## Dataset Summary

Geographical scope: Switzerland<br>Period: 1400-1550<br>Languages: Middle High German<br>Type of document: Protocols<br>Provenance: State Archive of Zurich

### Projects Included

- Rats-undRichtebücher_MF_1_3543
- Rats-undRichtebücher_MF_1_3544
- escript_test

## Dataset Structure

### Features

- **image**: `Image(mode=None, decode=False)`
- **xml_content**: `Value('string')`
- **filename**: `Value('string')`
"""


class TestParseProjects:
    """Taking every "- " line collected tags and feature bullets as projects.

    On the real 32-dataset catalogue that produced 163 names shared between
    datasets — `config_name: default` in all 32 — so every dataset looked like a
    duplicate of every other and pages_per_project was divided by a count that
    was mostly Markdown.
    """

    def test_only_the_projects_section_is_read(self):
        assert parse_projects(REAL_CARD) == (
            "Rats-undRichtebücher_MF_1_3543",
            "Rats-undRichtebücher_MF_1_3544",
            "escript_test",
        )

    @pytest.mark.parametrize("noise", [
        "htr", "pagexml", "image-to-text", "de", "la",
        "**image**: `Image(mode=None, decode=False)`",
        "**filename**: `Value('string')`",
    ])
    def test_frontmatter_and_feature_bullets_are_not_projects(self, noise):
        assert noise not in parse_projects(REAL_CARD)

    def test_a_card_without_the_section_selects_the_dataset_whole(self):
        """koenigsfelden-charters-part-3 has tags and prose, no project list."""
        assert parse_projects("# Card\n\nBased on the Koenigsfelden Data Set.\n") == ()

    def test_parse_card_falls_back_to_the_section_when_given_no_list(self):
        c = parse_card("x", REAL_CARD, pages=300, gb=1.0)
        assert len(c.projects) == 3 and c.pages_per_project == 100

    def test_an_explicit_list_still_wins(self):
        c = parse_card("x", REAL_CARD, pages=300, gb=1.0, projects=["only_this"])
        assert c.projects == ("only_this",)


class TestSliversAndHoldouts:
    def test_a_held_out_project_is_not_reported_as_a_duplicate(self):
        """rats showed "-2 dup" for the two evaluation volumes, which it is not."""
        plan = plan_corpus([cand("a", pages=1000, projects=["p1", "p2", "held"])],
                           MEDIEVAL_GERMAN, exclude_projects=["held"])
        s = plan.selections[0]
        assert s.dropped_excluded == ("held",) and s.dropped_duplicates == ()
        assert any("held out by request" in n for n in plan.notes)

    def test_a_sliver_left_over_after_dedup_is_dropped(self):
        """Three datasets survived with 4, 14 and 33 pages — 0.4 % of the corpus
        for three extra prepare streams."""
        plan = plan_corpus([
            cand("big", pages=9000, projects=[f"p{i}" for i in range(9)]),
            cand("sliver", pages=100, projects=["p0", "p1", "own"]),
        ], MEDIEVAL_GERMAN, max_share=1.0, min_pages=100)
        assert [s.repo for s in plan.selections] == ["big"]
        assert any("below min_pages" in why for _, why in plan.rejected)

    def test_min_pages_zero_keeps_everything(self):
        plan = plan_corpus([
            cand("big", pages=9000, projects=[f"p{i}" for i in range(9)]),
            cand("sliver", pages=100, projects=["p0", "own"]),
        ], MEDIEVAL_GERMAN, max_share=1.0, min_pages=0)
        assert len(plan.selections) == 2

    def test_dropping_everything_is_an_error_not_an_empty_corpus(self):
        with pytest.raises(CorpusPlanError, match="every dataset fell below"):
            plan_corpus([cand("a", pages=10, projects=["p"])],
                        MEDIEVAL_GERMAN, min_pages=1000)
