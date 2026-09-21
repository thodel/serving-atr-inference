import math
import socket

import httpx
import pytest

from atr_serving.config import REPO_ROOT, Settings
from atr_serving.manager import ManagerError, ModelManager
from atr_serving.registry import load_registry

HEBREW = "qwen3vl-8b-hebrew"
OCS = "qwen3vl-8b-old-church-slavonic"
LIGHTON = "lightonocr-catmus-caroline"  # pinned, 3000 MB


@pytest.fixture(autouse=True)
def _the_card_is_never_read_unless_a_test_says_so(monkeypatch):
    """``ensure_resident`` reads the vLLM card before every launch. Unstubbed,
    that is the host's nvidia-smi: on a Mac there is none and the launch goes
    ahead, on idhefix with xix resident (13654 MiB free on GPU 1) hebrew is
    refused, and five of these tests went red for a reason that is not in the
    code. "Cannot say" is the neutral answer; a test that wants a card sets one,
    which overrides this."""
    monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory", lambda index: None)
    monkeypatch.setattr("atr_serving.manager.gpu_probe.inspect",
                        lambda *a, **k: pytest.fail("the host's GPUs were inspected"))


class FakeHandle:
    def __init__(self, port: int, pid: int | None = None) -> None:
        self.port = port
        self.pid = pid
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


@pytest.fixture
def outgoing(monkeypatch):
    """Every connection attempt, recorded, and refused.

    Recorded because refusing alone proves nothing: the old ``_gpu_claim``
    wrapped its call in ``try/except Exception``, which swallows the refusal,
    and a call through ``httpx.Client`` never touches ``httpx.get``. Refused at
    the transports and at the socket, so no client — httpx's convenience
    functions, a ``Client``, urllib — gets past; a test asserts the record is
    empty afterwards.
    """
    calls: list[str] = []

    def refuse(target):
        calls.append(str(target))
        raise AssertionError(f"the manager tried to reach {target}")

    def verb(method):
        return lambda url, *a, **k: refuse(f"{method} {url}")

    for method in ("get", "post", "request"):
        monkeypatch.setattr(httpx, method, verb(method))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request",
                        lambda self, request: refuse(request.url))

    async def refuse_async(self, request):
        refuse(request.url)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse_async)
    monkeypatch.setattr(socket.socket, "connect", lambda self, address: refuse(address))
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, address: refuse(address))
    return calls


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
def test_the_gateway_never_asks_a_trainer_about_its_card(monkeypatch, outgoing, url):
    """Nothing trains on this box any more (#139), so no trainer is asked —
    not the one on asteraix, and not one that happens to sit on loopback.

    7636 MB free is the 15.09. figure the old "trainer did not answer" bar
    refused for lightonocr; with nobody to be unsure about, fitting is enough.
    """
    from atr_serving.manager import GpuBusyError

    _free_vram(monkeypatch, 7636)
    m, launcher = make_manager(train_url=url, train_api_key="trainer-secret")
    assert m.ensure_resident(LIGHTON) == 8210

    _free_vram(monkeypatch, 4000)
    with pytest.raises(GpuBusyError, match="4000 MB free"):
        m.ensure_resident(HEBREW)                # the fit check still refuses
    assert launcher.starts == [(LIGHTON, 8210, 1)]
    assert outgoing == []


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


# ── what to evict: the budget and the card together ──────────────────────────

XIX = "qwen3vl-german-xix-v1"            # 12000 MB in the registry
X4B = "qwen3.5-4b-german-xix-v1"         # 12000 MB
X2B = "qwen3.5-2b-german-xix-v1"         # 7000 MB


class CardLauncher(FakeLauncher):
    """Models that take memory from GPU 1 while they run and give it back
    ``lag`` reads after they are terminated, as nvidia-smi may report it.

    Each model is a ``vllm serve`` (the handle's pid) whose engine core, one pid
    up, holds the memory — so the manager finds a footprint only by descent.
    ``foreign_mb`` is a neighbour or an orphan: on the card, in no budget."""

    def __init__(self, monkeypatch, *, engines_mb, footprint, lag=0):
        super().__init__()
        self.engines_mb, self.footprint, self.lag = engines_mb, footprint, lag
        self.foreign_mb = 0
        self.reads = 0
        self.gone_after: dict[str, int] = {}
        self.parent: dict[int, int] = {}
        monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory", self.card_memory)
        monkeypatch.setattr("atr_serving.manager.gpu_probe.inspect", self.inspect)
        monkeypatch.setattr("atr_serving.manager.gpu_probe.descends_from", self.descends_from)

    def start(self, spec, port, gpu, settings) -> FakeHandle:
        h = super().start(spec, port, gpu, settings)
        h.pid = 5000 + 10 * len(self.starts)
        self.parent[h.pid + 1] = h.pid
        return h

    def descends_from(self, pid, ancestor):
        while pid is not None:
            if pid == ancestor:
                return True
            pid = self.parent.get(pid)
        return False

    def _holds(self, model, handle):
        if not handle.terminated:
            return True
        return self.reads <= self.gone_after.setdefault(model, self.reads - 1 + self.lag)

    def _held(self):
        return {m: h for m, h in self.handles.items() if self._holds(m, h)}

    def card_memory(self, index):
        self.reads += 1
        held = sum(self.footprint[m] for m in self._held())
        return 46068 - self.engines_mb - self.foreign_mb - held, 46068

    def inspect(self, *args, **kwargs):
        from atr_serving import gpu

        rows = [gpu.Process(pid=100, used_mib=self.engines_mb,
                            service="atr-party.service", own_service=True)]
        if self.foreign_mb:
            rows.append(gpu.Process(pid=200, used_mib=self.foreign_mb,
                                    service="gunicorn.service"))
        rows += [gpu.Process(pid=h.pid + 1, used_mib=self.footprint[m],
                             service="atr-gateway.service", own_service=True)
                 for m, h in self._held().items()]
        used = sum(p.used_mib for p in rows)
        card = gpu.Card(1, "NVIDIA A40", 46068, used, 46068 - used, 0, 0)
        card.processes = rows
        return [card]


#: What each model holds once autosized. xix is the 16.09. measurement; the
#: others are what plan_gpu_budget grants them on that card.
FOOTPRINT = {XIX: 16584, X4B: 16584, X2B: 11056, HEBREW: 28000, LIGHTON: 5490}


def _card(monkeypatch, *, lag=0, budget=28190):
    sleeps: list[float] = []
    launcher = CardLauncher(monkeypatch, engines_mb=15830, footprint=FOOTPRINT, lag=lag)
    m = ModelManager(load_registry(REPO_ROOT / "config" / "models.yaml"),
                     Settings(vllm_vram_budget_mb=budget, vllm_gpu=1, vllm_port_base=8210),
                     launcher=launcher, sleep=sleeps.append)
    return m, launcher, sleeps


def _xix_resident(monkeypatch, *, lag=0):
    """16.09.: xix resident, 16584 MiB beside 15830 MiB of engines, 13654 free."""
    m, launcher, sleeps = _card(monkeypatch, lag=lag)
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


def test_a_launch_the_card_cannot_take_evicts_nothing(monkeypatch):
    """8000 MiB of something foreign arrive beside xix. Evicting xix gives back
    16584 MiB; with 5654 free that is 22238, and hebrew needs 22748. The budget
    alone (12000 + 18000 > 28190) would evict xix and then refuse hebrew anyway,
    on every hebrew request, each one followed by a cold start of xix. Refuse,
    and leave xix serving."""
    from atr_serving.manager import GpuBusyError

    m, launcher, sleeps = _xix_resident(monkeypatch)
    launcher.foreign_mb = 8000
    for _ in range(2):
        with pytest.raises(GpuBusyError, match="5654 MB free") as err:
            m.ensure_resident(HEBREW)
        assert "would give back 16584 MB" in str(err.value)
        assert "nothing was evicted" in str(err.value)
        assert m.ensure_resident(XIX) == 8210   # still up, not relaunched
    assert m.resident_model_ids() == [XIX]
    assert not launcher.handles[XIX].terminated
    assert [s[0] for s in launcher.starts] == [XIX]
    assert sleeps == []                        # nothing evicted, nothing to wait for


def test_the_card_can_ask_for_more_evictions_than_the_budget(monkeypatch):
    """xix and the 2B resident, 2598 MiB free. For hebrew the budget wants one
    eviction (37000 - 12000 is within 28190); the card wants both, 2598 + 16584
    being short of 22748. Evicting only the first was a dead model and a 503."""
    m, launcher, sleeps = _xix_resident(monkeypatch)
    m.ensure_resident(X2B)
    assert launcher.card_memory(1)[0] == 2598
    m.ensure_resident(HEBREW)
    assert m.resident_model_ids() == [HEBREW]
    assert launcher.handles[XIX].terminated and launcher.handles[X2B].terminated
    assert sleeps == []


@pytest.mark.parametrize("resident, launch", [(XIX, X4B), (X4B, XIX), (HEBREW, LIGHTON)])
def test_a_launch_within_the_budget_still_evicts_for_the_card(monkeypatch, resident, launch):
    """xix and the 4B are 12000 MB each: 24000 is within 28190, so the budget
    never evicts, and 13654 MiB free is short of the 15848 either needs. With
    hebrew resident (28000 MiB, 2238 free) the same holds for the pinned
    lightonocr (18000 + 3000). Before, each got a 503 until something else
    evicted the resident — and nothing else does: lazy models have no idle
    unload."""
    m, launcher, sleeps = _card(monkeypatch)
    m.ensure_resident(resident)
    m.ensure_resident(launch)
    assert m.resident_model_ids() == [launch]
    assert launcher.handles[resident].terminated
    assert sleeps == []


def test_pinned_models_are_not_evicted_for_the_card(monkeypatch):
    """lightonocr pinned and xix lazy: xix is all there is to give."""
    from atr_serving.manager import GpuBusyError

    m, launcher, _ = _card(monkeypatch)
    m.ensure_resident(LIGHTON)
    m.ensure_resident(XIX)
    launcher.foreign_mb = 7000                 # 46068-15830-5490-16584-7000 = 1164
    with pytest.raises(GpuBusyError, match="would give back 16584 MB"):
        m.ensure_resident(HEBREW)
    assert m.resident_model_ids() == [LIGHTON, XIX]
    assert not any(h.terminated for h in launcher.handles.values())


def test_the_least_recently_used_goes_first(monkeypatch):
    """xix, the 2B, xix used again: the 2B is the one to go. Either eviction
    would do (11238 free, the 4B needs 15848; 12000 + 7000 + 12000 is over the
    budget by one model), so the order alone decides."""
    m, launcher, _ = _card(monkeypatch)
    launcher.footprint = {**FOOTPRINT, XIX: 12000, X2B: 7000}
    m.ensure_resident(XIX)
    m.ensure_resident(X2B)
    m.ensure_resident(XIX)
    m.ensure_resident(X4B)
    assert m.resident_model_ids() == [XIX, X4B]
    assert launcher.handles[X2B].terminated
    assert not launcher.handles[XIX].terminated


def test_an_unmeasured_resident_counts_at_its_weights(monkeypatch):
    """No pid to follow, so vram_mb, the low estimate: 5654 + 12000 is short of
    22748, and nothing goes. At the most autosizing grants it (12000 x 1.6 =
    19200) xix would go, and the 16584 MiB it really holds would leave hebrew
    510 MiB short — a dead model and a 503 on a guess."""
    from atr_serving.manager import GpuBusyError

    m, launcher, sleeps = _xix_resident(monkeypatch)
    launcher.handles[XIX].pid = None
    launcher.foreign_mb = 8000
    with pytest.raises(GpuBusyError, match="would give back 12000 MB"):
        m.ensure_resident(HEBREW)
    assert not launcher.handles[XIX].terminated and sleeps == []


def test_memory_on_its_way_out_is_waited_for(monkeypatch):
    """nvidia-smi can still count a model that has just been terminated."""
    m, _, sleeps = _xix_resident(monkeypatch, lag=3)
    m.ensure_resident(HEBREW)
    assert m.resident_model_ids() == [HEBREW]
    assert sleeps == [1, 1, 1]


def test_a_card_that_stays_full_after_an_eviction_is_refused(monkeypatch):
    """Bounded: memory that does not come back is not waited for forever. The
    one refusal after an eviction left: the plan counted memory the driver
    never returned."""
    from atr_serving.manager import EVICTION_SETTLE_READS, GpuBusyError

    m, launcher, sleeps = _xix_resident(monkeypatch, lag=10_000)
    with pytest.raises(GpuBusyError, match="13654 MB free") as err:
        m.ensure_resident(HEBREW)
    assert f"Evicted {XIX}" in str(err.value)
    assert len(sleeps) == EVICTION_SETTLE_READS - 1
    assert HEBREW not in launcher.handles


def test_the_derived_budget_leaves_the_foreigner_to_the_card(monkeypatch):
    """Budget unset, as on idhefix: one reading serves the budget and the
    footprints. The 8000 MiB neighbour is not in the budget (still 28190, so
    12000 + 18000 asks for xix to go) — the card's arithmetic is what keeps
    xix: the probe that found the regression ran exactly this."""
    import os

    from atr_serving.manager import GpuBusyError, vram_budget

    m, launcher, sleeps = _card(monkeypatch, budget=None)
    assert m.settings.vllm_vram_budget_mb is None
    reads = []
    monkeypatch.setattr("atr_serving.manager.gpu_probe.inspect",
                        lambda *a, **k: reads.append(1) or launcher.inspect())
    m.ensure_resident(XIX)
    launcher.parent[launcher.handles[XIX].pid] = os.getpid()   # a child of ours
    launcher.foreign_mb = 8000
    assert vram_budget(m.settings, launcher.inspect()).mb == 28190
    reads.clear()
    with pytest.raises(GpuBusyError, match="nothing was evicted"):
        m.ensure_resident(HEBREW)
    assert len(reads) == 1
    assert m.resident_model_ids() == [XIX]
    assert not launcher.handles[XIX].terminated and sleeps == []


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
