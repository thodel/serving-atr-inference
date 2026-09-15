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


def make_manager(budget: int = 24000):
    reg = load_registry(REPO_ROOT / "config" / "models.yaml")
    settings = Settings(vllm_vram_budget_mb=budget, vllm_gpu=1, vllm_port_base=8210)
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

def _claim(monkeypatch, payload):
    """What the trainer answers at /gpu-claim, or an exception for "no answer"."""
    class Response:
        def raise_for_status(self): pass
        def json(self): return payload

    def fake_get(url, timeout=None):
        assert url.endswith("/gpu-claim")
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

    _claim(monkeypatch, {"gpu": 1, "claimed": True,
                         "jobs": [{"id": "20260915T053651Z-qwen3vl-german-pages-v4",
                                   "status": "training", "stage": "train"}]})
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
    _claim(monkeypatch, {"claimed": False, "jobs": []})
    _free_vram(monkeypatch, 40000)
    m, _ = make_manager()
    port = m.ensure_resident(HEBREW)

    _claim(monkeypatch, {"claimed": True, "jobs": [{"id": "v4", "stage": "train"}]})
    assert m.ensure_resident(HEBREW) == port        # no launch, no refusal


def test_prepare_and_compile_do_not_claim_the_card(monkeypatch):
    """The trainer decides this, and answers claimed=false; the gateway obeys.

    v3 spent three and a half hours in prepare and compile. Blocking inference
    for that long would be a worse fault than the one this prevents.
    """
    _claim(monkeypatch, {"claimed": False,
                         "jobs": []})            # a compiling job is not listed
    _free_vram(monkeypatch, 40000)
    m, launcher = make_manager()
    assert m.ensure_resident(HEBREW) == 8210
    assert launcher.starts == [(HEBREW, 8210, 1)]


def test_a_full_card_is_refused_even_with_no_training(monkeypatch):
    """The other half: launching into a card that cannot hold the model."""
    from atr_serving.manager import GpuBusyError

    _claim(monkeypatch, {"claimed": False, "jobs": []})
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
    _claim(monkeypatch, {"claimed": False, "jobs": []})
    _free_vram(monkeypatch, None)
    m, launcher = make_manager()
    assert m.ensure_resident(HEBREW) == 8210
    assert launcher.starts == [(HEBREW, 8210, 1)]
