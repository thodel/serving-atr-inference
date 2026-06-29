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
