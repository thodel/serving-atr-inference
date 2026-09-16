import pytest

from atr_serving.config import REPO_ROOT, Settings
from atr_serving.manager import ManagerError, ModelManager
from atr_serving.registry import load_registry

HEBREW = "qwen3vl-8b-hebrew"
OCS = "qwen3vl-8b-old-church-slavonic"
LIGHTON = "lightonocr-catmus-caroline"  # pinned, 3000 MB


class FakeHandle:
    def __init__(self, port: int) -> None:
        self.port = port
        self.healthy = True
        self.terminated = False

    def is_healthy(self) -> bool:
        return self.healthy

    def terminate(self) -> None:
        self.terminated = True


class FakeLauncher:
    def __init__(self) -> None:
        self.handles: dict[str, FakeHandle] = {}
        self.starts: list[tuple[str, int, int]] = []

    def start(self, spec, port, gpu, settings) -> FakeHandle:
        self.starts.append((spec.id, port, gpu))
        h = FakeHandle(port)
        self.handles[spec.id] = h
        return h


def make_manager(budget: int = 24000, **overrides):
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    settings = Settings(vllm_vram_budget_mb=budget, vllm_gpu=1, vllm_port_base=8210,
                        **overrides)
    launcher = FakeLauncher()
    return ModelManager(reg, settings, launcher=launcher, sleep=lambda _s: None), launcher


def test_lazy_start_returns_port():
    m, launcher = make_manager()
    port = m.ensure_resident(HEBREW)
    assert port == 8210
    assert m.resident_model_ids() == [HEBREW]
    assert m.port_for(HEBREW) == 8210
    assert launcher.starts == [(HEBREW, 8210, 1)]


def test_lru_eviction_keeps_pinned():
    # budget 24000: lightonocr(3000)+one 8B(18000)=21000 fits; a 2nd 8B forces evict
    m, launcher = make_manager(budget=24000)
    m.ensure_resident(LIGHTON)
    m.ensure_resident(HEBREW)
    m.ensure_resident(OCS)
    ids = m.resident_model_ids()
    assert LIGHTON in ids                      # pinned never evicted
    assert OCS in ids
    assert HEBREW not in ids                    # LRU lazy evicted
    assert launcher.handles[HEBREW].terminated  # its subprocess was terminated


def test_reuse_is_idempotent_and_mru():
    m, _ = make_manager(budget=40000)
    p1 = m.ensure_resident(HEBREW)
    p2 = m.ensure_resident(HEBREW)
    assert p1 == p2
    assert m.resident_model_ids() == [HEBREW]


def test_relaunch_when_unhealthy():
    m, launcher = make_manager(budget=40000)
    m.ensure_resident(HEBREW)
    launcher.handles[HEBREW].healthy = False
    m.ensure_resident(HEBREW)
    assert len([s for s in launcher.starts if s[0] == HEBREW]) == 2


def test_non_vllm_model_raises():
    m, _ = make_manager()
    with pytest.raises(ManagerError):
        m.ensure_resident("kraken-catmus-medieval")
    with pytest.raises(ManagerError):
        m.ensure_resident("does-not-exist")


def test_shutdown_terminates_all():
    m, launcher = make_manager(budget=40000)
    m.ensure_resident(HEBREW)
    m.ensure_resident(LIGHTON)
    m.shutdown()
    assert m.resident_model_ids() == []
    assert all(h.terminated for h in launcher.handles.values())


# ── the card is not ours alone (#129) ────────────────────────────────────────

def _claim(monkeypatch, payload, seen=None):
    """What the trainer answers at /gpu-claim, or an exception for "no answer".

    ``seen`` collects ``(url, headers)`` per call."""
    class Response:
        def raise_for_status(self): pass
        def json(self): return payload

    def fake_get(url, timeout=None, headers=None):
        assert url.endswith("/gpu-claim")
        if seen is not None:
            seen.append((url, headers))
        if isinstance(payload, Exception):
            raise payload
        return Response()

    monkeypatch.setattr("atr_serving.manager.httpx.get", fake_get)


def _free_vram(monkeypatch, free_mb):
    monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory",
                        lambda index: None if free_mb is None else (free_mb, 46068))


def test_a_training_run_keeps_the_card(monkeypatch):
    """The incident: an inference request during a 33-hour run would OOM it."""
    from atr_serving.manager import GpuBusyError

    _claim(monkeypatch, {"gpu": 1, "claimed": True, "holding": True,
                         "jobs": [{"id": "20260915T053651Z-qwen3vl-german-pages-v4",
                                   "status": "training", "stage": "train",
                                   "holding": True}]})
    _free_vram(monkeypatch, 40000)          # plenty free between two peaks
    m, launcher = make_manager()
    with pytest.raises(GpuBusyError, match="qwen3vl-german-pages-v4"):
        m.ensure_resident(HEBREW)
    assert launcher.starts == []


def test_a_busy_card_is_not_a_broken_one(monkeypatch):
    """GpuBusyError is a ManagerError, so nothing that catches the base breaks."""
    from atr_serving.manager import GpuBusyError
    assert issubclass(GpuBusyError, ManagerError)


def test_models_already_resident_keep_serving_while_a_run_holds_the_card(monkeypatch):
    """Refusing a *launch* must not take down what is already up."""
    _claim(monkeypatch, {"claimed": False, "holding": False, "jobs": []})
    _free_vram(monkeypatch, 40000)
    m, _ = make_manager()
    port = m.ensure_resident(HEBREW)

    _claim(monkeypatch, {"claimed": True, "holding": True,
                         "jobs": [{"id": "v4", "stage": "train", "holding": True}]})
    assert m.ensure_resident(HEBREW) == port        # no launch, no refusal


def test_the_gateway_obeys_a_no_claim_answer(monkeypatch):
    """Which jobs claim the card is the trainer's decision, not this one's.

    This test was named `test_prepare_and_compile_do_not_claim_the_card` and said
    so in its docstring, which stopped being true in 0b59bcd: a job claims the
    card from its first stage, because a model launched during v4's prepare was
    still resident when its train began and took the run down. The assertion was
    always about the gateway obeying the answer it is given, so only the name was
    ever wrong — which is how a test comes to document a policy the code reversed.
    """
    _claim(monkeypatch, {"claimed": False, "jobs": []})
    _free_vram(monkeypatch, 40000)
    m, launcher = make_manager()
    assert m.ensure_resident(HEBREW) == 8210
    assert launcher.starts == [(HEBREW, 8210, 1)]


def test_a_full_card_is_refused_even_with_no_training(monkeypatch):
    """The other half: launching into a card that cannot hold the model."""
    from atr_serving.manager import GpuBusyError

    _claim(monkeypatch, {"claimed": False, "holding": False, "jobs": []})
    _free_vram(monkeypatch, 4000)
    m, launcher = make_manager()
    with pytest.raises(GpuBusyError, match="4000 MB free"):
        m.ensure_resident(HEBREW)
    assert launcher.starts == []


def test_an_unreachable_trainer_does_not_stop_inference(monkeypatch):
    """Fail open, and say so.

    Refusing every inference request whenever the trainer is down would trade a
    rare, recoverable fault for a constant one. The free-VRAM check still holds.
    """
    _claim(monkeypatch, RuntimeError("connection refused"))
    _free_vram(monkeypatch, 40000)
    m, launcher = make_manager()
    assert m.ensure_resident(HEBREW) == 8210
    assert launcher.starts == [(HEBREW, 8210, 1)]


def test_an_unreachable_trainer_still_respects_a_full_card(monkeypatch):
    from atr_serving.manager import GpuBusyError

    _claim(monkeypatch, RuntimeError("connection refused"))
    _free_vram(monkeypatch, 4000)
    m, _ = make_manager()
    with pytest.raises(GpuBusyError):
        m.ensure_resident(HEBREW)


def test_a_card_nvidia_smi_cannot_read_is_left_to_the_launch(monkeypatch):
    """No claim and no reading is not evidence of a problem."""
    _claim(monkeypatch, {"claimed": False, "holding": False, "jobs": []})
    _free_vram(monkeypatch, None)
    m, launcher = make_manager()
    assert m.ensure_resident(HEBREW) == 8210
    assert launcher.starts == [(HEBREW, 8210, 1)]


def test_the_15_09_incident_would_now_be_refused(monkeypatch):
    """The exact numbers of the failure this guard was supposed to prevent.

    v4 was in its train stage. /gpu-claim listed 44 job records off a CIFS share
    and took longer than the two-second probe, so the trainer's answer was lost.
    The card had 7,636 MB free — which fits lightonocr's 3,000 + 2,048 reserve
    exactly, and would have left v4 to die at its next peak. The guard said
    "ALLOWED".
    """
    from atr_serving.manager import GpuBusyError

    _claim(monkeypatch, RuntimeError("timed out"))
    _free_vram(monkeypatch, 7636)
    m, launcher = make_manager()
    with pytest.raises(GpuBusyError, match="did not answer"):
        m.ensure_resident(LIGHTON)          # 3000 MB, would have fitted
    assert launcher.starts == []


def test_an_idle_card_still_serves_when_the_trainer_is_down(monkeypatch):
    """Fail-open's actual purpose survives: an idle box keeps answering."""
    _claim(monkeypatch, RuntimeError("connection refused"))
    _free_vram(monkeypatch, 44000)
    m, launcher = make_manager()
    assert m.ensure_resident(LIGHTON) == 8210
    assert launcher.starts == [(LIGHTON, 8210, 1)]


def test_a_definite_no_claim_only_needs_the_model_to_fit(monkeypatch):
    """With the trainer answering, the strict bar is unnecessary and wrong.

    Another vLLM model resident is not a reason to refuse — that is what the LRU
    budget is for.
    """
    _claim(monkeypatch, {"claimed": False, "holding": False, "jobs": []})
    _free_vram(monkeypatch, 7636)
    m, launcher = make_manager()
    assert m.ensure_resident(LIGHTON) == 8210
    assert launcher.starts == [(LIGHTON, 8210, 1)]


# ── the trainer moved (#137) ─────────────────────────────────────────────────

ASTERAIX = "http://130.92.59.242:8204"


def test_the_gateway_no_longer_asks_a_remote_trainer_about_a_local_card(monkeypatch):
    """A run holding a card on asteraix must not refuse launches onto idhefix's.

    Once ATR_TRAIN_URL named asteraix, the 24-hour VLM run there would have
    answered ``holding`` to every launch here. The guard answers for a remote
    trainer itself, without a call, and the fit check decides.
    """
    from loguru import logger

    from atr_serving.manager import GpuBusyError

    def no_call(*args, **kwargs):
        raise AssertionError("the remote trainer was asked about a local card")

    monkeypatch.setattr("atr_serving.manager.httpx.get", no_call)
    said: list[str] = []
    sink = logger.add(lambda m: said.append(m.record["message"]), level="INFO")
    try:
        _free_vram(monkeypatch, 7636)
        m, launcher = make_manager(train_url=ASTERAIX)
        # 7636 MB is the 15.09. figure the "trainer did not answer" bar refuses.
        # Here there is nothing to be unsure about, so fitting is enough.
        assert m.ensure_resident(LIGHTON) == 8210

        _free_vram(monkeypatch, 4000)
        with pytest.raises(GpuBusyError, match="4000 MB free"):
            m.ensure_resident(HEBREW)            # the fit check still refuses
    finally:
        logger.remove(sink)
    assert launcher.starts == [(LIGHTON, 8210, 1)]
    assert len([s for s in said if "not on this box" in s]) == 1   # once, not per launch


@pytest.mark.parametrize("url", ["http://127.0.0.1:8204", "http://localhost:8204",
                                 "http://[::1]:8204"])
def test_a_local_trainer_is_still_asked_about_the_card(monkeypatch, url):
    """Until #139 removes it, the claim still protects a run on this box."""
    from atr_serving.manager import GpuBusyError

    seen: list = []
    _claim(monkeypatch, {"claimed": True, "holding": True,
                         "jobs": [{"id": "v4", "stage": "train", "holding": True}]},
           seen=seen)
    _free_vram(monkeypatch, 40000)
    m, launcher = make_manager(train_url=url, train_api_key="trainer-secret")
    with pytest.raises(GpuBusyError, match="v4"):
        m.ensure_resident(HEBREW)
    assert launcher.starts == []
    assert seen == [(f"{url}/gpu-claim", {"X-API-Key": "trainer-secret"})]


def test_a_keyless_local_trainer_gets_no_key_header(monkeypatch):
    seen: list = []
    _claim(monkeypatch, {"claimed": False, "holding": False, "jobs": []}, seen=seen)
    _free_vram(monkeypatch, 40000)
    m, _ = make_manager()
    m.ensure_resident(HEBREW)
    assert seen == [("http://127.0.0.1:8204/gpu-claim", {})]


def test_release_lazy_drops_the_evictable_and_keeps_the_pinned(monkeypatch):
    """What the trainer asks for at the stage boundary (#129).

    The window this closes: v4's prepare-time vLLM was still holding 16.5 GB
    when its train stage began, and the run died three minutes later.
    """
    _claim(monkeypatch, {"claimed": False, "holding": False, "jobs": []})
    _free_vram(monkeypatch, 44000)
    m, launcher = make_manager(budget=40000)
    m.ensure_resident(HEBREW)        # lazy
    m.ensure_resident(LIGHTON)       # pinned

    dropped, kept = m.release_lazy()
    assert dropped == [HEBREW] and kept == [LIGHTON]
    assert m.resident_model_ids() == [LIGHTON]
    assert launcher.handles[HEBREW].terminated
    assert not launcher.handles[LIGHTON].terminated


def test_release_lazy_on_an_empty_gateway_is_not_an_error(monkeypatch):
    m, _ = make_manager()
    assert m.release_lazy() == ([], [])


def test_a_released_port_can_be_reused(monkeypatch):
    """_drop returns the port to the pool; a later launch must get one."""
    _claim(monkeypatch, {"claimed": False, "holding": False, "jobs": []})
    _free_vram(monkeypatch, 44000)
    m, _ = make_manager()
    first = m.ensure_resident(HEBREW)
    m.release_lazy()
    assert m.ensure_resident(HEBREW) == first
