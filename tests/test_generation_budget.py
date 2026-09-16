"""How many tokens a served model may generate (#131).

`ATR_VLLM_MAX_NEW_TOKENS` was 512 for every model. That is ample for a line crop
and short for a page: `qwen3vl-german-xix-v1` and `qwen3vl-8b-hebrew` are both
registered `level: page`, so the whole image goes in one call and the answer is a
whole page of text. Reaching the ceiling returns a normal 200 whose transcription
stops mid-sentence — visible since #123, but only to someone who looks.
"""

from __future__ import annotations

from atr_serving.config import Settings
from atr_serving.pipeline import generation_budget
from atr_serving.registry import ModelSpec


def spec(**kwargs) -> ModelSpec:
    return ModelSpec(id="m", engine="vllm", hf_repo="x/y", **kwargs)


def settings(**kwargs) -> Settings:
    return Settings(**kwargs)


def test_a_page_model_is_not_held_to_a_line_s_ceiling():
    assert generation_budget(spec(level="page"), settings()) == 4096


def test_a_line_model_keeps_the_line_ceiling():
    """512 was never wrong for a line; it was wrong as the answer for both."""
    assert generation_budget(spec(level="line"), settings()) == 512


def test_a_model_that_knows_its_own_length_wins():
    """A page of Hebrew and a page of Kurrent are not the same length."""
    assert generation_budget(spec(level="page", max_new_tokens=2048), settings()) == 2048
    assert generation_budget(spec(level="line", max_new_tokens=900), settings()) == 900


def test_the_budget_cannot_exceed_what_the_context_can_hold():
    """16384 has to hold the prompt and the image too. Asking for more output
    than fits does not produce a long transcription, it produces an error."""
    budget = generation_budget(
        spec(level="page", max_new_tokens=99999),
        settings(vllm_max_model_len=16384, vllm_prompt_reserve_tokens=4096),
    )
    assert budget == 16384 - 4096


def test_an_uncapped_context_takes_the_value_as_given():
    assert generation_budget(spec(level="page", max_new_tokens=99999),
                             settings(vllm_max_model_len=None)) == 99999


def test_the_page_default_fits_the_shipped_context():
    """The default must not need the clamp — a fallback that is always capped is
    two numbers pretending to be one."""
    s = settings()
    assert s.vllm_max_new_tokens_page <= s.vllm_max_model_len - s.vllm_prompt_reserve_tokens


def test_the_environment_still_overrides_both():
    s = settings(vllm_max_new_tokens=128, vllm_max_new_tokens_page=1024)
    assert generation_budget(spec(level="line"), s) == 128
    assert generation_budget(spec(level="page"), s) == 1024


def test_every_page_model_in_the_shipped_registry_gets_more_than_512():
    """The regression this fixes, stated over the actual registry."""
    from pathlib import Path

    from atr_serving.registry import load_registry

    s = settings()
    shipped = load_registry(Path(__file__).resolve().parents[1] / "config" / "models.yaml")
    pages = [m for m in shipped.all() if m.engine == "vllm" and m.level == "page"]
    assert pages, "the registry has no page-level vLLM model; this test is vacuous"
    assert all(generation_budget(m, s) > 512 for m in pages)
