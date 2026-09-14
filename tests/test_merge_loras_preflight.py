"""The three ways a LoRA merge is doomed before it starts.

Each case here was met on asterAIx on 2026-09-14, trying to merge the three
`dh-unibe` German-XIX adapters. All three attempts loaded the base first — 26
seconds and several GB — and only then failed, twice with a message that pointed
at the wrong thing. The preflight exists so the tool answers "which venv" instead
of the operator answering it from a stack trace.

Pure functions over metadata: no torch, no GPU, no download.
"""

import pytest

from scripts.merge_loras import (
    merged_state,
    missing_artifacts,
    Blocker,
    _version_tuple,
    peft_blocker,
    transformers_blocker,
    vllm_blocker,
)


# ── version parsing ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("0.20.0", (0, 20, 0)),
    ("0.19.1", (0, 19, 1)),
    ("4.57.6", (4, 57, 6)),
    ("0.17.1.dev0", (0, 17, 1)),      # stops at the first non-numeric part
    ("1", (1,)),
    ("", ()),
    (None, ()),
    ("not-a-version", ()),
])
def test_version_tuple(raw, expected):
    assert _version_tuple(raw) == expected


# ── peft: the qwen3vl case ───────────────────────────────────────────────────

def test_an_older_peft_than_wrote_the_adapter_is_fatal():
    """The live failure: peft 0.19.1 reading an adapter written by 0.20.0 dies as
    AttributeError("'list' object has no attribute 'keys'") — which names neither
    peft nor a version, so nobody looks at the venv."""
    b = peft_blocker("0.20.0", "0.19.1")
    assert b is not None and b.fatal
    assert "0.20.0" in b.text and "0.19.1" in b.text
    assert "vlm-train" in b.text, "the message has to name a venv that would work"


@pytest.mark.parametrize("installed", ["0.20.0", "0.21.0", "1.0.0"])
def test_an_equal_or_newer_peft_is_fine(installed):
    assert peft_blocker("0.20.0", installed) is None


@pytest.mark.parametrize("adapter,installed", [
    (None, "0.19.1"),        # adapter predates peft_version in the config
    ("0.20.0", None),        # peft not importable here
    ("weird", "0.19.1"),
    ("0.20.0", "weird"),
])
def test_what_cannot_be_compared_does_not_block(adapter, installed):
    """A preflight that blocks on its own uncertainty is a preflight that gets
    switched off, and then it protects nothing."""
    assert peft_blocker(adapter, installed) is None


# ── transformers: the qwen3.5 case ───────────────────────────────────────────

KNOWN = {"qwen3_vl", "qwen2_vl", "llama"}


def test_an_unknown_architecture_is_fatal_and_names_it():
    b = transformers_blocker("qwen3_5", KNOWN, "4.57.6")
    assert b is not None and b.fatal
    assert "qwen3_5" in b.text and "4.57.6" in b.text


def test_the_message_forecloses_the_trust_remote_code_reflex():
    """`Qwen/Qwen3.5-4B` ships no auto_map and no modeling_*.py, so there is no
    remote code to trust — but trying it is the first thing anyone does."""
    b = transformers_blocker("qwen3_5", KNOWN, "4.57.6")
    assert "auto_map" in b.text and "trust_remote_code" in b.text


def test_a_known_architecture_passes():
    assert transformers_blocker("qwen3_vl", KNOWN, "4.57.6") is None


def test_remote_code_makes_an_unknown_architecture_loadable():
    """With an auto_map the base carries its own implementation; transformers not
    knowing the type is then not an obstacle."""
    assert transformers_blocker("something_new", KNOWN, "4.57.6",
                                has_remote_code=True) is None


@pytest.mark.parametrize("model_type,known", [(None, KNOWN), ("qwen3_5", set())])
def test_transformers_check_stays_quiet_when_it_cannot_tell(model_type, known):
    assert transformers_blocker(model_type, known, "4.57.6") is None


# ── vLLM: merged, and still unservable ───────────────────────────────────────

#: What vLLM 0.11.0 on asterAIx actually reported, 2026-09-14.
VLLM_0_11_QWEN = {
    "Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration",
    "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM",
    "Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration",
}


def test_an_unservable_architecture_warns_without_blocking():
    """Not fatal: merging for a planned upgrade, or from a venv without vLLM, is
    legitimate. But 9 GB that this box can never load is worth saying out loud."""
    b = vllm_blocker(["Qwen3_5ForConditionalGeneration"], VLLM_0_11_QWEN)
    assert b is not None and not b.fatal
    assert "Qwen3_5ForConditionalGeneration" in b.text
    assert "driver" in b.text


def test_a_servable_architecture_is_silent():
    assert vllm_blocker(["Qwen3VLForConditionalGeneration"], VLLM_0_11_QWEN) is None


def test_no_vllm_in_this_venv_means_not_checked_not_unsupported():
    """merge_loras is meant to run from the training venv, which has no vLLM. An
    empty support set is absence of evidence."""
    assert vllm_blocker(["Qwen3_5ForConditionalGeneration"], set()) is None
    assert vllm_blocker([], VLLM_0_11_QWEN) is None


# ── the whole picture, as it actually presented ──────────────────────────────

def test_the_three_german_xix_models_as_they_behaved_on_asteraix():
    """One regression case per model, with the versions the box reported.

    qwen3vl is mergeable in one venv and not the other; the qwen3.5 pair is
    mergeable in neither and servable in neither.
    """
    # qwen3vl-german-xix-v1 — wrong venv only
    assert peft_blocker("0.20.0", "0.19.1").fatal is True          # .venvs/vllm
    assert peft_blocker("0.20.0", "0.20.0") is None                # .venvs/vlm-train
    assert transformers_blocker("qwen3_vl", {"qwen3_vl"}, "4.57.6") is None
    assert vllm_blocker(["Qwen3VLForConditionalGeneration"], VLLM_0_11_QWEN) is None

    # qwen3.5-{4b,2b}-german-xix-v1 — no venv on the box can load the base,
    # and the vLLM that could serve it needs a driver the box does not have.
    assert transformers_blocker("qwen3_5", {"qwen3_vl"}, "4.57.6").fatal is True
    assert vllm_blocker(["Qwen3_5ForConditionalGeneration"], VLLM_0_11_QWEN) is not None


def test_blocker_renders_its_severity():
    assert str(Blocker(True, "x")).startswith("BLOCKED:")
    assert str(Blocker(False, "x")).startswith("WARNING:")


# ── the half-written merge ───────────────────────────────────────────────────

#: `ls ~/atr-cache/vllm-merged/qwen3vl-german-xix-v1/` on asterAIx, 2026-09-14
#: 12:44 — after a run that printed FAILED and left this behind. 8.3 GB of
#: correct weights, and every retry skipped it because config.json was there.
HALF_WRITTEN = [
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
]

COMPLETE = HALF_WRITTEN + [
    "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
    "chat_template.jinja",
]


def test_the_half_written_directory_is_recognised_as_incomplete():
    """The regression this check exists for. `config.json` is present, so the old
    rule called it merged, skipped it on every retry, and resolve_model_path would
    have served a model with no tokenizer."""
    assert missing_artifacts(HALF_WRITTEN) == ["tokenizer", "processor"]


def test_a_complete_directory_is_complete():
    assert missing_artifacts(COMPLETE) == []


def test_weights_alone_are_not_a_model():
    assert set(missing_artifacts(["model.safetensors"])) == {"config", "tokenizer", "processor"}


def test_an_empty_directory_is_missing_everything():
    assert len(missing_artifacts([])) == 4


@pytest.mark.parametrize("processor_file", [
    "preprocessor_config.json",      # older Qwen2-VL-era layout
    "processor_config.json",         # the combined processor config
    "image_processor_config.json",
])
def test_either_processor_layout_counts(processor_file):
    """Which of these a model carries depends on its family, not on whether the
    merge worked — so all of them satisfy the same requirement."""
    files = ["config.json", "model.safetensors", "tokenizer.json", processor_file]
    assert missing_artifacts(files) == []


@pytest.mark.parametrize("tokenizer_file", [
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "spiece.model",
])
def test_either_tokenizer_layout_counts(tokenizer_file):
    files = ["config.json", "model.safetensors", tokenizer_file, "processor_config.json"]
    assert missing_artifacts(files) == []


def test_sharded_and_single_file_weights_both_count():
    base = ["config.json", "tokenizer.json", "processor_config.json"]
    assert missing_artifacts(base + ["model-00001-of-00002.safetensors"]) == []
    assert missing_artifacts(base + ["pytorch_model.bin"]) == []


def test_merged_state_on_a_real_directory(tmp_path):
    assert merged_state(tmp_path / "nope") == (False, [])

    half = tmp_path / "half"
    half.mkdir()
    for name in HALF_WRITTEN:
        (half / name).write_text("x")
    assert merged_state(half) == (True, ["tokenizer", "processor"])

    for name in ("tokenizer.json", "preprocessor_config.json"):
        (half / name).write_text("x")
    assert merged_state(half) == (True, [])
