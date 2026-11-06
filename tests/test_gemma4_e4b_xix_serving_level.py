"""`gemma4-e4b-german-xix-v1` is served per line, at the soft-token step it trained at.

Two facts from G6 (training-atr-models#86, 05.11.2026, 15 XIX validation pages,
vLLM 0.29.0 on asteraix) decide this entry, and each fails silently if lost:

* **It reads lines, not pages.** Lines CER 0.143 over 646 lines; whole pages
  1.16 with a length ratio of 0.21. Registered `level: page` it would return a
  fragment per page and a 200.
* **It reads only at `max_soft_tokens` 140.** Gemma 4 rounds a pixel request up
  to one of five soft-token steps; training asked for 262144 pixels and got step
  140. At vLLM's default step the same adapter reads lines at CER 0.90. The
  merged directory takes its processor from the base (step 280), so the step
  must come from the launch flag.
"""

import json

import pytest

from atr_serving.config import REPO_ROOT, Settings
from atr_serving.pipeline import visual_budget
from atr_serving.registry import load_registry

MODEL_ID = "gemma4-e4b-german-xix-v1"

#: Gemma 4's cell: patch 16 x pooling kernel 3.
CELL = 48
TRAINED_STEP = 140


@pytest.fixture(scope="module")
def spec():
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    assert MODEL_ID in reg
    return reg.get(MODEL_ID)


def test_it_is_served_the_way_it_was_measured(spec):
    assert spec.level == "line"


def test_vllm_is_launched_at_the_trained_soft_token_step(spec):
    args = spec.vllm_args
    assert "--mm-processor-kwargs" in args
    kwargs = json.loads(args[args.index("--mm-processor-kwargs") + 1])
    assert kwargs == {"max_soft_tokens": TRAINED_STEP}


def test_the_gateway_budget_is_what_that_step_carries(spec):
    class _Settings:
        vllm_visual_budget = True

    assert spec.max_pixels == TRAINED_STEP * CELL * CELL == 322560
    assert visual_budget(spec, _Settings()) == 322560


def test_it_runs_on_the_vllm_g6_measured(spec):
    assert spec.vllm_venv == "vllm-next"


def test_the_prompt_is_the_trained_one(spec):
    assert spec.prompt == "Transcribe the handwritten text in this image exactly as written."


#: What merge_loras.py measured on idhefix, 06.11.2026 (training-atr-models#83).
MERGED_MIB = 15147


def test_it_is_enabled_and_sized_for_the_merge(spec):
    """Merged on idhefix in .venvs/vllm-next; enabled so the first read through
    /recognize can happen at all (the gateway never serves a disabled vLLM
    entry). The launch is sized from vram_mb, so it must hold the merge."""
    assert spec.enabled is True and not spec.disabled_reason
    assert spec.vram_mb >= MERGED_MIB


def test_launch_args_reach_the_command_line(spec, tmp_path, monkeypatch):
    """The flags are only worth pinning if the launcher passes them on."""
    from atr_serving import manager

    monkeypatch.setattr(manager, "vllm_executable", lambda spec, settings: tmp_path / "vllm")
    monkeypatch.setattr(manager, "gpu_budget",
                        lambda spec, gpu, settings: manager.Budget(0.5, "test"))
    monkeypatch.setattr(manager, "resolve_model_path", lambda spec, settings: "/merged/m")
    launched = []

    class FakeProc:
        pass

    monkeypatch.setattr(manager.subprocess, "Popen",
                        lambda cmd, **kw: launched.append(cmd) or FakeProc())
    manager.VllmLauncher().start(spec, 8299, 1, Settings())
    cmd = launched[0]
    assert cmd[cmd.index("--mm-processor-kwargs") + 1] == '{"max_soft_tokens": 140}'
    assert cmd[cmd.index("--served-model-name") + 1] == MODEL_ID
