"""The planner must never select a dataset whose card says it is machine output.

Planning 1800-1900 put `handwritten-bundesratsprotokolle_xix-xx` at 45 % of the
corpus with a score of 0.95. Its card reads, verbatim:

    --- Data has been automatically created, using ATR models ---
    These are ''automatically'' transcribed pages.
    !!!This data set does not contain Ground Truth!!!

The scorer read only period, language and script, so a good card for bad data
scored well.
"""
import pytest

from atr_serving.training.corpus_plan import (
    Target,
    declares_no_ground_truth,
    parse_card,
    score_candidate,
)

BUNDESRAT = (
    "--- Data has been automatically created, using ATR models ---\n"
    "These are ''automatically'' transcribed pages.\n"
    "The HuggingFace Hub automatically merges all parquet files when loading.\n"
    "!!!This data set does not contain Ground Truth!!!\n"
    "Period: 1848-1910<br>Languages: German<br>Type of document: Protocol<br>"
)
#: Every pagexml-hf card carries this line, including every good one.
GOOD = (
    "The HuggingFace Hub automatically merges all parquet files when loading.\n"
    "This dataset was created using pagexml-hf converter from Transkribus PageXML data.\n"
    "Period: 1803-1883<br>Languages: German<br>Type of document: Protocol<br>"
)


@pytest.mark.parametrize("phrase", [
    "This data set does not contain Ground Truth",
    "no ground truth is included",
    "These are ''automatically'' transcribed pages.",
    "Data has been automatically created, using ATR models",
    "created using HTR models",
])
def test_each_machine_output_phrase_is_recognised(phrase):
    assert declares_no_ground_truth(phrase)


def test_the_boilerplate_every_card_carries_is_not_a_veto():
    """A bare match on 'automatically' would have vetoed the whole org."""
    assert not declares_no_ground_truth(GOOD)


def test_a_machine_transcribed_dataset_scores_zero_whatever_its_metadata():
    target = Target(period=(1800, 1900))
    bad = score_candidate(parse_card("dh-unibe/bundesrat", BUNDESRAT, 148494, 1.0), target)
    good = score_candidate(parse_card("dh-unibe/zh", GOOD, 152786, 1.0), target)
    assert bad.score == 0.0
    assert "machine output" in bad.why
    assert good.score > 0.6
