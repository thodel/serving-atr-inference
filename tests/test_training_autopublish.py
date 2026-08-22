"""When a finished model is uploaded automatically, and when it is not (#88).

Publishing is outward-facing and the hub keeps history, so an unpublish is a
deletion that does not undo the copy someone already pulled. The decision is
therefore separated from the upload, and every refusal states its own reason —
"nothing happened" is the worst possibleanswer for an automatic action.
"""

import pytest

from atr_serving.training.autopublish import TOKEN_ENV, decide

TOKEN = {"HF_TOKEN": "hf_xxx"}


class TestTheThreshold:
    def test_a_model_above_the_threshold_is_published(self):
        d = decide(83.0, 80.0, "dh-unibe", TOKEN)
        assert d.publish is True and d.org == "dh-unibe"

    def test_exactly_at_the_threshold_counts(self):
        assert decide(80.0, 80.0, "dh-unibe", TOKEN).publish is True

    def test_below_it_does_not(self):
        d = decide(76.76, 80.0, "dh-unibe", TOKEN)
        assert d.publish is False and "76.76" in d.reason and "80.00" in d.reason

    def test_the_real_vlm_run_would_not_have_been_published(self):
        """qwen3vl-german-medieval-v1 scored 76.76 % (CER 0.2324)."""
        assert decide(76.76, 80.0, "dh-unibe", TOKEN).publish is False


class TestRefusals:
    def test_off_by_default(self):
        """A threshold of 0 must never publish, whatever the score."""
        assert decide(99.9, 0.0, "dh-unibe", TOKEN).publish is False

    def test_a_negative_threshold_does_not_mean_always(self):
        assert decide(10.0, -1.0, "dh-unibe", TOKEN).publish is False

    def test_an_unmeasured_run_is_never_published(self):
        """A model whose score is unknown is not a model that passed."""
        d = decide(None, 80.0, "dh-unibe", TOKEN)
        assert d.publish is False and "no char_accuracy" in d.reason

    def test_a_missing_token_is_reported_not_discovered_mid_upload(self):
        d = decide(95.0, 80.0, "dh-unibe", {})
        assert d.publish is False
        assert "no token" in d.reason and "publish_to_hub.py" in d.reason

    @pytest.mark.parametrize("name", TOKEN_ENV)
    def test_either_token_variable_works(self, name):
        assert decide(95.0, 80.0, "dh-unibe", {name: "hf_x"}).publish is True

    def test_a_blank_token_does_not_count(self):
        assert decide(95.0, 80.0, "dh-unibe", {"HF_TOKEN": "   "}).publish is False


class TestPrivacy:
    def test_an_automatic_publish_is_always_private(self):
        """Automation gets less latitude than the human at the shell, not more:
        making a trained model public stays a deliberate, separate decision."""
        assert decide(99.0, 80.0, "dh-unibe", TOKEN).private is True

    def test_a_refusal_reads_as_a_sentence_because_it_lands_on_the_job_record(self):
        assert str(decide(50.0, 80.0, "o", TOKEN)).startswith("skip: ")
        assert str(decide(90.0, 80.0, "o", TOKEN)).startswith("publish: ")
