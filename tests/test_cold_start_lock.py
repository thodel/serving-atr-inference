"""Four concurrent requests for a cold model are one cold start (#164).

On 21.09.2026 a batch with ``--concurrency 4`` hit the gateway just after a
restart. `qwen3.5-4b-german-xix-v2` was not resident, so all four requests ran
through `ensure_resident` at once, all four read the same 33 801 MiB free, all
four took a port, and all four launched a vLLM sized at 19 200 MiB as though it
were alone on the card. Together they did not fit. All four died with code 1,
and the caller saw 502 "vLLM process exited (code 1) during startup".

`ensure_resident` runs in `run_in_threadpool`, so the concurrency is real and
the tests have to be real too: these run threads, because four calls made one
after another pass without the fix.

Two layers, and both are needed:

* per model, one start whose outcome every caller takes — otherwise four
  requests are four consecutive startup timeouts when the start fails;
* over all models, one launch at a time — otherwise two *different* models both
  pass the free-memory check before either has claimed any.
"""

import threading
import time

import pytest

from atr_serving.config import REPO_ROOT, Settings
from atr_serving.manager import ManagerError, ModelManager
from atr_serving.registry import load_registry

HEBREW = "qwen3vl-8b-hebrew"
OCS = "qwen3vl-8b-old-church-slavonic"
LIGHTON = "lightonocr-catmus-caroline"

START_S = 0.25      # long enough that the other threads are queued on the lock
JOIN_S = 30.0       # a thread still running at this point is a deadlock


@pytest.fixture(autouse=True)
def _the_card_is_never_read(monkeypatch):
    """As in test_manager.py: the host's nvidia-smi must not decide these."""
    monkeypatch.setattr("atr_serving.manager.gpu_probe.card_memory", lambda index: None)
    monkeypatch.setattr("atr_serving.manager.gpu_probe.inspect",
                        lambda *a, **k: pytest.fail("the host's GPUs were inspected"))


class FakeHandle:
    def __init__(self, port: int) -> None:
        self.port = port
        self.healthy = True
        self.terminated = False

    def is_healthy(self) -> bool:
        return self.healthy

    def terminate(self) -> None:
        self.terminated = True


class SlowLauncher:
    """A launcher that takes time, counts overlap, and can fail.

    ``resident_when_started`` is what #164's fourth criterion is about: what the
    manager already held when this launch was planned. Without the cross-model
    lock the second launch is planned while the first holds nothing yet, and
    this list comes back empty for both.
    """

    def __init__(self, manager_box: list, fail: str | None = None,
                 delay: float = START_S) -> None:
        self.starts: list[tuple[str, int]] = []
        self.resident_when_started: list[tuple[str, tuple[str, ...]]] = []
        self.inside = 0
        self.most_at_once = 0
        self._manager_box = manager_box
        self._fail = fail
        self._delay = delay
        self._lock = threading.Lock()

    def start(self, spec, port, gpu, settings) -> FakeHandle:
        manager = self._manager_box[0]
        with self._lock:
            self.inside += 1
            self.most_at_once = max(self.most_at_once, self.inside)
            self.starts.append((spec.id, port))
            self.resident_when_started.append(
                (spec.id, tuple(manager.resident_model_ids())))
        try:
            time.sleep(self._delay)
            if self._fail is not None:
                raise ManagerError(self._fail)
            return FakeHandle(port)
        finally:
            with self._lock:
                self.inside -= 1


def make_manager(fail: str | None = None, budget: int = 40000, delay: float = START_S):
    box: list = []
    registry = load_registry(REPO_ROOT / "config" / "models.yaml")
    settings = Settings(vllm_vram_budget_mb=budget, vllm_gpu=1, vllm_port_base=8210)
    launcher = SlowLauncher(box, fail=fail, delay=delay)
    manager = ModelManager(registry, settings, launcher=launcher,
                           sleep=lambda _s: None)
    box.append(manager)
    return manager, launcher


def run_together(calls):
    """Run ``calls`` in threads that all start at the same moment.

    The barrier is what makes this a test of the lock: every thread is past the
    "already resident?" fast path before any of them can finish a launch.
    """
    gate = threading.Barrier(len(calls))
    results: list = [None] * len(calls)

    def worker(index, call):
        gate.wait()
        try:
            results[index] = call()
        except BaseException as exc:   # noqa: BLE001 — the result under test
            results[index] = exc

    threads = [threading.Thread(target=worker, args=(i, c), daemon=True)
               for i, c in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_S)
        assert not thread.is_alive(), "a caller never came back — deadlock"
    return results


# ── one start per model ─────────────────────────────────────────────────────
def test_four_concurrent_requests_start_one_vllm():
    """The incident, as a test. Without the lock this is four starts on four
    ports, which is what put four 19 200 MiB instances on one card."""
    manager, launcher = make_manager()

    results = run_together([lambda: manager.ensure_resident(HEBREW)] * 4)

    assert launcher.starts == [(HEBREW, 8210)]
    assert results == [8210, 8210, 8210, 8210]


def test_the_four_callers_get_the_port_of_that_one_instance():
    manager, launcher = make_manager()

    results = run_together([lambda: manager.ensure_resident(HEBREW)] * 4)

    assert len(set(results)) == 1
    assert manager.port_for(HEBREW) == results[0]
    assert manager.resident_model_ids() == [HEBREW]


def test_only_one_port_is_taken_for_one_model():
    """Each of the four launches took its own port in the incident, which is how
    four instances could listen at once."""
    manager, launcher = make_manager()

    run_together([lambda: manager.ensure_resident(HEBREW)] * 4)

    assert manager._ports.in_use() == {8210}


def test_a_failed_start_reaches_every_caller_with_its_reason():
    manager, launcher = make_manager(fail="vLLM process exited (code 1) during startup")

    results = run_together([lambda: manager.ensure_resident(HEBREW)] * 4)

    assert all(isinstance(r, ManagerError) for r in results), results
    assert all("code 1" in str(r) for r in results), results


def test_a_failed_start_is_tried_once_not_once_per_caller():
    """Four tries would be four startup timeouts in a row — on the box, four
    times `vllm_startup_timeout_s` before the first caller hears anything."""
    manager, launcher = make_manager(fail="boom")

    run_together([lambda: manager.ensure_resident(HEBREW)] * 4)

    assert launcher.starts == [(HEBREW, 8210)]


def test_a_failed_start_leaves_no_port_taken():
    """A port held for a model that is not running moves the next launch to the
    next port, and the one after that to the next again."""
    manager, launcher = make_manager(fail="boom")

    run_together([lambda: manager.ensure_resident(HEBREW)] * 4)

    assert manager._ports.in_use() == set()
    assert manager.port_for(HEBREW) is None
    assert manager.resident_model_ids() == []


def test_the_next_request_after_a_failure_is_a_fresh_attempt():
    """The shared outcome is one attempt, not a permanent verdict: a start that
    failed because the card was full must be retried when it is not."""
    manager, launcher = make_manager(fail="boom", delay=0.0)

    with pytest.raises(ManagerError):
        manager.ensure_resident(HEBREW)
    launcher._fail = None

    assert manager.ensure_resident(HEBREW) == 8210
    assert len(launcher.starts) == 2


# ── one launch at a time, across models ─────────────────────────────────────
def test_two_different_models_do_not_launch_at_the_same_time():
    """The other half of the incident: two models that each fit alone, planned
    against the same free card, and launched into it together."""
    manager, launcher = make_manager()

    run_together([lambda: manager.ensure_resident(HEBREW),
                  lambda: manager.ensure_resident(OCS)])

    assert launcher.most_at_once == 1
    assert sorted(manager.resident_model_ids()) == sorted([HEBREW, OCS])


def test_the_second_plan_sees_the_first_model_on_the_card():
    """#164's fourth criterion. Whichever goes second must have been planned
    while the first was already resident — otherwise its budget is sized as
    though the card were empty."""
    manager, launcher = make_manager()

    run_together([lambda: manager.ensure_resident(HEBREW),
                  lambda: manager.ensure_resident(OCS)])

    first, second = launcher.resident_when_started
    assert first[1] == ()
    assert second[1] == (first[0],)


def test_each_model_gets_its_own_port():
    manager, launcher = make_manager()

    results = run_together([lambda: manager.ensure_resident(HEBREW),
                            lambda: manager.ensure_resident(OCS)])

    assert sorted(results) == [8210, 8211]
    assert manager._ports.in_use() == {8210, 8211}


def test_a_failed_launch_does_not_block_the_next_model():
    """The lock is released on the way out of a failure, not only on success."""
    manager, launcher = make_manager(fail="boom", delay=0.05)

    results = run_together([lambda: manager.ensure_resident(HEBREW),
                            lambda: manager.ensure_resident(OCS)])

    assert all(isinstance(r, ManagerError) for r in results), results
    assert {spec for spec, _ in launcher.starts} == {HEBREW, OCS}


# ── a resident model costs nothing ──────────────────────────────────────────
def test_a_resident_model_waits_for_no_cold_start():
    """#164's fifth criterion. The lock is held for minutes during a real cold
    start; a request for a model already on the card must not queue behind it."""
    manager, launcher = make_manager(delay=0.0)
    assert manager.ensure_resident(LIGHTON) == 8210
    launcher._delay = 5.0        # a cold start nobody should wait out

    started = threading.Event()
    done = threading.Event()

    def cold():
        started.set()
        try:
            manager.ensure_resident(HEBREW)
        finally:
            done.set()

    thread = threading.Thread(target=cold, daemon=True)
    thread.start()
    started.wait(JOIN_S)
    time.sleep(0.2)              # the cold start is inside launcher.start by now

    begun = time.monotonic()
    assert manager.ensure_resident(LIGHTON) == 8210
    waited = time.monotonic() - begun

    assert waited < 1.0, f"a resident model waited {waited:.1f}s on the launch lock"
    assert not done.is_set()
    thread.join(JOIN_S)


def test_an_unhealthy_resident_is_relaunched_once_for_four_callers():
    """The relaunch path takes the same route: four callers finding the same
    wedged instance must not start four replacements."""
    manager, launcher = make_manager(delay=0.0)
    manager.ensure_resident(HEBREW)
    launcher._delay = START_S
    manager._resident[HEBREW].handle.healthy = False

    results = run_together([lambda: manager.ensure_resident(HEBREW)] * 4)

    assert len(launcher.starts) == 2, launcher.starts
    assert len(set(results)) == 1
    assert manager.port_for(HEBREW) == results[0]
