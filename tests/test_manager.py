import math

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


# ── the card is not ours alone ───────────────────────────────────────────────

def _free_vram(monkeypatch, free_mb):
    """What nvidia-smi reports free on GPU 1, or None for "cannot say"."""
    monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory",
                        lambda index: None if free_mb is None else (free_mb, 46068))


def _no_http(monkeypatch):
    """Fail the test on any HTTP call the manager makes."""
    def no_call(*args, **kwargs):
        raise AssertionError(f"the manager made an HTTP call: {args}")

    for verb in ("get", "post", "request"):
        monkeypatch.setattr(f"atr_serving.manager.httpx.{verb}", no_call)


def test_a_busy_card_is_not_a_broken_one():
    """GpuBusyError is a ManagerError, so nothing that catches the base breaks."""
    from atr_serving.manager import GpuBusyError
    assert issubclass(GpuBusyError, ManagerError)


def test_a_full_card_is_refused(monkeypatch):
    """Launching into a card that cannot hold the model is what produced the
    incidents this manager exists for."""
    from atr_serving.manager import GpuBusyError

    _free_vram(monkeypatch, 4000)
    m, launcher = make_manager()
    with pytest.raises(GpuBusyError, match="4000 MB free") as err:
        m.ensure_resident(HEBREW)
    assert "GET /gpu" in str(err.value)
    assert launcher.starts == []


def test_the_fit_check_asks_what_the_launcher_asks(monkeypatch):
    """Weights x MIN_HEADROOM + reserve, the launcher's own minimum: in between,
    the fit check used to pass and gpu_budget then refused with a 502."""
    from atr_serving.manager import MIN_HEADROOM, GpuBusyError

    need = math.ceil(18000 * MIN_HEADROOM) + 2048          # hebrew: 18000 MB
    _free_vram(monkeypatch, need - 1)
    m, launcher = make_manager()
    with pytest.raises(GpuBusyError, match=f"needs {need} MB"):
        m.ensure_resident(HEBREW)
    _free_vram(monkeypatch, need)
    assert m.ensure_resident(HEBREW) == 8210


def test_models_already_resident_keep_serving_on_a_full_card(monkeypatch):
    """Refusing a *launch* must not take down what is already up."""
    _free_vram(monkeypatch, 40000)
    m, _ = make_manager()
    port = m.ensure_resident(HEBREW)
    _free_vram(monkeypatch, 0)
    assert m.ensure_resident(HEBREW) == port        # no launch, no refusal


def test_a_card_nvidia_smi_cannot_read_is_left_to_the_launch(monkeypatch):
    """No reading is not evidence of a problem."""
    _free_vram(monkeypatch, None)
    m, launcher = make_manager()
    assert m.ensure_resident(HEBREW) == 8210
    assert launcher.starts == [(HEBREW, 8210, 1)]


ASTERAIX = "http://130.92.59.242:8204"


@pytest.mark.parametrize("url", [ASTERAIX, "http://127.0.0.1:8204",
                                 "http://localhost:8204", "http://[::1]:8204"])
def test_the_gateway_never_asks_a_trainer_about_its_card(monkeypatch, url):
    """Nothing trains on this box any more (#139), so no trainer is asked —
    not the one on asteraix, and not one that happens to sit on loopback.

    7636 MB free is the 15.09. figure the old "trainer did not answer" bar
    refused for lightonocr; with nobody to be unsure about, fitting is enough.
    """
    from atr_serving.manager import GpuBusyError

    _no_http(monkeypatch)
    _free_vram(monkeypatch, 7636)
    m, launcher = make_manager(train_url=url, train_api_key="trainer-secret")
    assert m.ensure_resident(LIGHTON) == 8210

    _free_vram(monkeypatch, 4000)
    with pytest.raises(GpuBusyError, match="4000 MB free"):
        m.ensure_resident(HEBREW)                # the fit check still refuses
    assert launcher.starts == [(LIGHTON, 8210, 1)]


def test_the_trainer_coordination_is_gone():
    """The pieces #139 removed, so a revert shows up here and not on the card."""
    import atr_serving.manager as manager

    assert not hasattr(ModelManager, "release_lazy")
    assert not hasattr(ModelManager, "_gpu_claim")
    assert not hasattr(ModelManager, "_refuse_while_training")
    assert "gpu_claim_timeout_s" not in Settings.model_fields
    assert not hasattr(manager, "is_loopback_url")


def test_a_released_port_can_be_reused(monkeypatch):
    """_drop returns the port to the pool; a later launch must get one."""
    _free_vram(monkeypatch, 44000)
    m, _ = make_manager()
    first = m.ensure_resident(HEBREW)
    m._drop(HEBREW)
    assert m.ensure_resident(HEBREW) == first


# ── evict, then check the card ───────────────────────────────────────────────

XIX = "qwen3vl-german-xix-v1"            # 12000 MB in the registry


class CardLauncher(FakeLauncher):
    """Models that take memory from GPU 1 while they run and give it back
    ``lag`` reads after they are terminated, as nvidia-smi may report it."""

    def __init__(self, monkeypatch, *, engines_mb, footprint, lag=0):
        super().__init__()
        self.engines_mb, self.footprint, self.lag = engines_mb, footprint, lag
        self.reads = 0
        self.gone_after: dict[str, int] = {}
        monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory", self.card_memory)

    def _holds(self, model, handle):
        if not handle.terminated:
            return True
        return self.reads <= self.gone_after.setdefault(model, self.reads - 1 + self.lag)

    def card_memory(self, index):
        self.reads += 1
        held = sum(self.footprint[m] for m, h in self.handles.items() if self._holds(m, h))
        return 46068 - self.engines_mb - held, 46068


def _xix_resident(monkeypatch, *, lag=0):
    """16.09.: xix resident, 16584 MiB beside 15830 MiB of engines, 13654 free."""
    sleeps: list[float] = []
    launcher = CardLauncher(monkeypatch, engines_mb=15830,
                            footprint={XIX: 16584, HEBREW: 28000}, lag=lag)
    m = ModelManager(load_registry(REPO_ROOT / "config" / "models.yaml"),
                     Settings(vllm_vram_budget_mb=28190, vllm_gpu=1, vllm_port_base=8210),
                     launcher=launcher, sleep=sleeps.append)
    assert m.registry.get(XIX).vram_mb == 12000
    m.ensure_resident(XIX)
    assert launcher.card_memory(1)[0] == 13654
    return m, launcher, sleeps


def test_a_launch_that_needs_an_eviction_gets_one(monkeypatch):
    """12000 + 18000 is over the budget, so xix goes. With the fit check first,
    13654 MiB free refused hebrew before the eviction could run, and xix was
    never evicted at all."""
    m, launcher, sleeps = _xix_resident(monkeypatch)
    m.ensure_resident(HEBREW)
    assert m.resident_model_ids() == [HEBREW]
    assert launcher.handles[XIX].terminated
    assert sleeps == []                        # freed at once, nothing to wait for


def test_memory_on_its_way_out_is_waited_for(monkeypatch):
    """nvidia-smi can still count a model that has just been terminated."""
    m, _, sleeps = _xix_resident(monkeypatch, lag=3)
    m.ensure_resident(HEBREW)
    assert m.resident_model_ids() == [HEBREW]
    assert sleeps == [1, 1, 1]


def test_a_card_that_stays_full_after_an_eviction_is_refused(monkeypatch):
    """Bounded: memory that does not come back is not waited for forever."""
    from atr_serving.manager import EVICTION_SETTLE_READS, GpuBusyError

    m, launcher, sleeps = _xix_resident(monkeypatch, lag=10_000)
    with pytest.raises(GpuBusyError, match="13654 MB free"):
        m.ensure_resident(HEBREW)
    assert len(sleeps) == EVICTION_SETTLE_READS - 1
    assert HEBREW not in launcher.handles


def test_a_launch_without_an_eviction_does_not_wait(monkeypatch):
    from atr_serving.manager import GpuBusyError

    sleeps: list[float] = []
    _free_vram(monkeypatch, 4000)
    m, _ = make_manager()
    m._sleep = sleeps.append
    with pytest.raises(GpuBusyError):
        m.ensure_resident(HEBREW)
    assert sleeps == []


# ── the budget after the split (#139) ────────────────────────────────────────

#: idhefix GPU 1, 16.09.2026 21:50 CEST.
CARD_MIB = 46068
ENGINES = {"atr-party.service": 8918, "atr-trocr.service": 3840,
           "atr-kraken.service": 3072}
XIX_CHILD_MIB = 16584
RESERVE_MIB = 2048


def _idhefix_gpu1(gateway_pid):
    from atr_serving import gpu

    card = gpu.Card(1, "NVIDIA A40", CARD_MIB, 32414, 13654, 0, 0)
    card.processes = [
        gpu.Process(pid=100 + i, used_mib=mib, service=unit, own_service=True)
        for i, (unit, mib) in enumerate(ENGINES.items())
    ]
    card.processes.append(gpu.Process(pid=gateway_pid + 1, used_mib=XIX_CHILD_MIB,
                                      service="atr-gateway.service",
                                      own_service=True))
    return card


def test_the_vram_budget_covers_the_whole_card_after_the_split(monkeypatch):
    """The budget + the engines + the reserve is the card, not more."""
    from atr_serving import gpu
    from atr_serving.manager import MEASURED_VRAM_BUDGET_MB, plan_vram_budget, vram_budget

    engines = sum(ENGINES.values())
    assert engines == 15830
    budget = plan_vram_budget(CARD_MIB, engines, RESERVE_MIB)
    assert budget + engines + RESERVE_MIB == CARD_MIB
    assert budget == MEASURED_VRAM_BUDGET_MB == 28190
    # The constant it replaces promised memory the card does not have.
    assert 30000 + engines + RESERVE_MIB > CARD_MIB

    # Read from the card: the gateway's own vLLM child is the budget being
    # spent, not an engine, so it is not subtracted.
    me = 4000
    monkeypatch.setattr(gpu, "descends_from", lambda pid, ancestor: pid == me + 1
                        and ancestor == me)
    derived = vram_budget(Settings(), [_idhefix_gpu1(me)], own_pid=me)
    assert derived.mb == 28190
    assert "15830 MiB engines" in derived.reason


def test_a_child_that_outlived_the_gateway_counts_as_an_engine(monkeypatch):
    """Still in atr-gateway.service, no longer ours to evict."""
    from atr_serving import gpu
    from atr_serving.manager import vram_budget

    monkeypatch.setattr(gpu, "descends_from", lambda pid, ancestor: False)
    derived = vram_budget(Settings(), [_idhefix_gpu1(4000)], own_pid=4000)
    assert derived.mb == CARD_MIB - 15830 - XIX_CHILD_MIB - RESERVE_MIB


def test_a_foreign_process_is_left_to_the_fit_check(monkeypatch):
    """The budget describes what is ours; the neighbours come and go."""
    from atr_serving import gpu
    from atr_serving.manager import vram_budget

    monkeypatch.setattr(gpu, "descends_from", lambda pid, ancestor: False)
    card = gpu.Card(1, "NVIDIA A40", CARD_MIB, 0, CARD_MIB, 0, 0)
    card.processes = [gpu.Process(pid=7, used_mib=10392, service="gunicorn.service")]
    assert vram_budget(Settings(), [card]).mb == CARD_MIB - RESERVE_MIB


def test_the_setting_overrides_the_reading(monkeypatch):
    from atr_serving import gpu
    from atr_serving.manager import vram_budget

    monkeypatch.setattr(gpu, "inspect", lambda *a, **k: pytest.fail("card was read"))
    budget = vram_budget(Settings(vllm_vram_budget_mb=24000))
    assert budget.mb == 24000 and "configured" in budget.reason


@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_budget_setting_means_derive_it(monkeypatch, value):
    monkeypatch.setenv("ATR_VLLM_VRAM_BUDGET_MB", value)
    assert Settings(_env_file=None).vllm_vram_budget_mb is None


def test_an_unreadable_card_falls_back_to_the_measurement(monkeypatch):
    from atr_serving import gpu
    from atr_serving.manager import MEASURED_VRAM_BUDGET_MB, vram_budget

    def no_smi(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi is not on PATH")

    monkeypatch.setattr(gpu, "inspect", no_smi)
    budget = vram_budget(Settings())
    assert budget.mb == MEASURED_VRAM_BUDGET_MB
    assert "unreadable" in budget.reason
    # And a card list without the vLLM card, too.
    assert vram_budget(Settings(vllm_gpu=3), []).mb == MEASURED_VRAM_BUDGET_MB


def test_the_derived_budget_decides_evictions(monkeypatch):
    """With the setting unset, the LRU evicts against the card as read: 28190
    holds one 8B (18000), and the second evicts the first."""
    from atr_serving import gpu

    reads: list[int] = []
    monkeypatch.setattr(gpu, "inspect",
                        lambda *a, **k: reads.append(1) or [_idhefix_gpu1(1)])
    monkeypatch.setattr(gpu, "descends_from", lambda pid, ancestor: pid == 2)
    _free_vram(monkeypatch, 44000)
    launcher = FakeLauncher()
    m = ModelManager(load_registry(REPO_ROOT / "config" / "models.yaml"),
                     Settings(vllm_gpu=1), launcher=launcher, sleep=lambda _s: None)
    m.ensure_resident(HEBREW)
    m.ensure_resident(OCS)
    assert m.resident_model_ids() == [OCS]
    assert launcher.handles[HEBREW].terminated
    assert len(reads) == 2                     # read per launch, not cached
