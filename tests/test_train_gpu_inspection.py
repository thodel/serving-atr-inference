"""#414: what holds GPU memory, and whose it is — the probe, and the gateway's
``GET /gpu`` (#139), which reads this box's cards with it.

``/train/gpu`` read this box's cards too until #137/#139; it is the trainer's
reading now, and tests/test_train_remote_trainer.py covers it.

Offline throughout: nvidia-smi is stubbed and /proc is stubbed, so the suite runs
on a laptop with no GPU. What is tested is the attribution, because that is the
part that turns a memory figure into something worth waking someone for.
"""

from __future__ import annotations

import pytest

from atr_serving import gpu


CARDS = [["0", "NVIDIA A100", "40960", "34300", "6660", "0", "0"],
         ["1", "NVIDIA A100", "40960", "1600", "39360", "97", "43"]]
UUIDS = [["GPU-aaa"], ["GPU-bbb"]]


def _smi_stub(apps):
    def fake(query, *, per_app):
        if per_app:
            return apps
        return UUIDS if query == "uuid" else CARDS
    return fake


@pytest.fixture
def procs(monkeypatch):
    """A settable /proc: {pid: (user, age, command, ppid[, unit])}."""
    table: dict = {}

    def info(pid):
        row = table.get(pid)
        return (None, None, None) if row is None else (row[0], row[1], row[2])

    def ancestors(pid, limit=32):
        chain, seen = [], set()
        cur = pid
        while cur in table and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = table[cur][3]
        return chain

    def unit(pid):
        row = table.get(pid)
        return row[4] if row and len(row) > 4 else None

    monkeypatch.setattr(gpu, "_proc_info", info)
    monkeypatch.setattr(gpu, "_ancestors", ancestors)
    monkeypatch.setattr(gpu, "_unit_of", unit)
    return table


def test_a_pid_with_no_proc_entry_is_orphaned(monkeypatch, procs):
    """The 27 530 MiB under '[Not Found]' — a dead parent's CUDA context."""
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["2743851", "27530", "GPU-aaa"]]))
    cards = gpu.inspect({})
    held = cards[0].processes[0]
    assert held.orphaned is True
    assert held.registered is False
    assert held.used_mib == 27530
    assert held.user is None and held.command is None


def test_a_process_of_a_known_job_is_registered(procs, monkeypatch):
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["4242", "12000", "GPU-aaa"]]))
    procs[4242] = ("tobias", 900.0, "ketos train ...", 1)
    cards = gpu.inspect({4242: "job-a"})
    assert cards[0].processes[0].registered is True
    assert cards[0].processes[0].job_id == "job-a"


def test_a_data_loader_child_counts_as_registered(procs, monkeypatch):
    """A worker of a healthy run is a child, not a stray.

    Matching on the exact pid alone would flag every DataLoader worker and bury
    the one row that matters.
    """
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["5001", "3000", "GPU-aaa"]]))
    procs[4242] = ("tobias", 900.0, "ketos train ...", 1)
    procs[5001] = ("tobias", 890.0, "python -c from multiprocessing...", 4242)
    cards = gpu.inspect({4242: "job-a"})
    assert cards[0].processes[0].registered is True
    assert cards[0].processes[0].job_id == "job-a"


def test_a_hand_started_run_is_not_registered(procs, monkeypatch):
    """The 30-hour ketos run belonging to no job."""
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["7777", "20000", "GPU-bbb"]]))
    procs[7777] = ("tobias", 108000.0, "ketos train -f page ...", 1)
    cards = gpu.inspect({4242: "job-a"})
    stray = cards[1].processes[0]
    assert stray.registered is False and stray.orphaned is False
    assert stray.age_s == 108000.0 and stray.user == "tobias"


def test_cards_carry_their_own_processes_only(procs, monkeypatch):
    monkeypatch.setattr(gpu, "_smi", _smi_stub(
        [["4242", "12000", "GPU-aaa"], ["7777", "20000", "GPU-bbb"]]))
    procs[4242] = ("tobias", 900.0, "a", 1)
    procs[7777] = ("tobias", 900.0, "b", 1)
    cards = gpu.inspect({})
    assert [p.pid for p in cards[0].processes] == [4242]
    assert [p.pid for p in cards[1].processes] == [7777]


def test_the_idle_card_holding_memory_is_visible(monkeypatch, procs):
    """0 % utilisation with 34 GB used is the shape of the incident."""
    monkeypatch.setattr(gpu, "_smi", _smi_stub([]))
    cards = gpu.inspect({})
    assert cards[0].utilisation_pct == 0 and cards[0].memory_used_mib == 34300
    assert cards[1].utilisation_pct == 97


def test_not_supported_values_do_not_crash_the_probe(monkeypatch, procs):
    """nvidia-smi writes '[N/A]' where a number belongs."""
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["4242", "[N/A]", "GPU-aaa"]]))
    procs[4242] = ("tobias", 1.0, "x", 1)
    assert gpu.inspect({})[0].processes[0].used_mib == 0


# ── GET /gpu: this box's cards (#139) ────────────────────────────────────────

import os                                          # noqa: E402
import socket                                      # noqa: E402

from fastapi.testclient import TestClient          # noqa: E402

from atr_serving.app import create_app             # noqa: E402
from atr_serving.config import Settings            # noqa: E402

KEY = "test-key"
AUTH = {"X-API-Key": KEY}
ME = os.getpid()


class _Manager:
    def __init__(self, resident=()):
        self.resident = list(resident)

    def resident_model_ids(self):
        return self.resident


def _idhefix(job_pids=None):
    """idhefix, 16.09.2026 21:50 CEST: the neighbours on card 0, the engines and
    the gateway's vLLM child on card 1."""
    assert not job_pids, "GET /gpu attributed pids to jobs, and nothing trains here"
    rag = gpu.Card(0, "NVIDIA A40", 46068, 10392, 35676, 0, 0)
    rag.processes = [
        gpu.Process(pid=3000 + i, used_mib=2598, service="gunicorn.service",
                    user="change", age_s=2365200.0, command="gunicorn app:app")
        for i in range(4)
    ]
    ours = gpu.Card(1, "NVIDIA A40", 46068, 32414, 13654, 0, 0)
    ours.processes = [
        gpu.Process(pid=100, used_mib=8918, service="atr-party.service", own_service=True),
        gpu.Process(pid=101, used_mib=3840, service="atr-trocr.service", own_service=True),
        gpu.Process(pid=102, used_mib=3072, service="atr-kraken.service", own_service=True),
        gpu.Process(pid=ME + 7, used_mib=16584, service="atr-gateway.service",
                    own_service=True, command="vllm serve qwen3vl-german-xix-v1"),
    ]
    return [rag, ours]


def _client(monkeypatch, cards=_idhefix, resident=("qwen3vl-german-xix-v1",), **kw):
    monkeypatch.setattr(gpu, "inspect", cards)
    monkeypatch.setattr(gpu, "descends_from",
                        lambda pid, ancestor: ancestor == ME and pid == ME + 7)
    monkeypatch.setattr(gpu, "service_of",
                        lambda pid: "atr-gateway.service" if pid == ME else None)
    app = create_app(Settings(api_key=KEY, require_auth=True, **kw))
    app.state.model_manager = _Manager(resident)

    def no_trainer(*args, **kwargs):
        raise AssertionError("GET /gpu asked the trainer")

    app.state.trainer_client = type("NoTrainer", (), {
        "gpu": no_trainer, "list_jobs": no_trainer, "health": no_trainer})()
    return TestClient(app)


def test_gpu_needs_the_key(monkeypatch):
    c = _client(monkeypatch, cards=lambda *a, **k: pytest.fail("read without a key"))
    assert c.get("/gpu").status_code == 401
    assert c.get("/gpu", headers={"X-API-Key": "wrong"}).status_code == 401


def test_gpu_reports_this_boxs_cards_without_job_attribution(monkeypatch):
    body = _client(monkeypatch).get("/gpu", headers=AUTH).json()
    assert body["host"] == socket.gethostname()
    # No trainer here: absent, not false — false reads as "trainer unreachable".
    assert "job_attribution_available" not in body and "known_job_pids" not in body
    rag, ours = body["cards"]
    assert all(p["registered"] is False and p["job_id"] is None
               for card in body["cards"] for p in card["processes"])

    # Card 1: the engines and the vLLM child are ours; nothing is unaccounted.
    assert ours["service_mib"] == 32414 and ours["unaccounted_mib"] == 0
    # Card 0: the neighbours' memory is memory we cannot have ...
    assert rag["unaccounted_mib"] == 10392 and rag["orphaned_mib"] == 0
    # ... and every row of it names its unit, which is what makes a reader
    # (atr_status._classify) call it foreign rather than raise the alarm.
    assert all(p["service"] == "gunicorn.service" and not p["orphaned"]
               and not p["own_service"] for p in rag["processes"])


def test_gpu_tells_the_vllm_child_from_the_engines(monkeypatch):
    body = _client(monkeypatch).get("/gpu", headers=AUTH).json()
    vllm = body["vllm"]
    assert vllm["gpu"] == 1
    assert vllm["service"] == "atr-gateway.service"
    assert vllm["pids"] == [ME + 7]
    assert vllm["residents"] == [{"id": "qwen3vl-german-xix-v1", "vram_mb": 12000,
                                  "residency": "lazy"}]
    engines = [p for p in body["cards"][1]["processes"]
               if p["own_service"] and p["pid"] not in vllm["pids"]]
    assert sum(p["used_mib"] for p in engines) == 15830
    # The budget the next launch would get, from the same reading.
    assert vllm["budget_mb"] == 28190
    assert "15830 MiB engines" in vllm["budget"]


def test_gpu_shows_a_configured_budget_as_configured(monkeypatch):
    body = _client(monkeypatch, vllm_vram_budget_mb=24000).get("/gpu", headers=AUTH).json()
    assert body["vllm"]["budget_mb"] == 24000
    assert "configured" in body["vllm"]["budget"]


def test_a_box_without_nvidia_smi_says_so(monkeypatch):
    def no_smi(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi is not on PATH")

    r = _client(monkeypatch, cards=no_smi).get("/gpu", headers=AUTH)
    assert r.status_code == 503 and "nvidia-smi" in r.json()["detail"]


def test_a_wedged_driver_is_a_502(monkeypatch):
    import subprocess

    def wedged(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 8)

    r = _client(monkeypatch, cards=wedged).get("/gpu", headers=AUTH)
    assert r.status_code == 502
    assert "TimeoutExpired" in r.json()["detail"]


def test_gpu_is_read_off_the_event_loop(monkeypatch):
    """nvidia-smi can take its whole 8 s timeout; the loop must keep serving."""
    import threading

    loop_thread: list[int] = []
    read_in: list[int] = []

    def cards(*args, **kwargs):
        read_in.append(threading.get_ident())
        return _idhefix()

    c = _client(monkeypatch, cards=cards)

    @c.app.middleware("http")
    async def remember(request, call_next):
        loop_thread.append(threading.get_ident())
        return await call_next(request)

    assert c.get("/gpu", headers=AUTH).status_code == 200
    assert read_in and loop_thread and read_in[0] != loop_thread[0]


def test_a_resident_the_registry_no_longer_knows_is_still_listed(monkeypatch):
    body = _client(monkeypatch, resident=("gone-model",)).get("/gpu", headers=AUTH).json()
    assert body["vllm"]["residents"] == [{"id": "gone-model", "vram_mb": None,
                                          "residency": None}]


def test_descends_from_walks_the_parent_chain(procs):
    """vllm serve starts an engine core, and that child holds the memory."""
    procs[10] = ("tobias", 1.0, "uvicorn atr_serving.app:app", 1)
    procs[11] = ("tobias", 1.0, "vllm serve", 10)
    procs[12] = ("tobias", 1.0, "VLLM::EngineCore", 11)
    procs[20] = ("tobias", 1.0, "python -m trocr_svc", 1)
    assert gpu.descends_from(12, 10) and gpu.descends_from(10, 10)
    assert not gpu.descends_from(20, 10)
    assert not gpu.descends_from(999, 10)          # gone: not ours to evict


def test_as_rows_has_the_contract_shape(procs):
    card = gpu.Card(1, "NVIDIA A40", 46068, 1000, 45068, 0, 0)
    card.processes = [gpu.Process(pid=1, used_mib=600, orphaned=True),
                      gpu.Process(pid=2, used_mib=400, service="atr-trocr.service",
                                  own_service=True)]
    (row,) = gpu.as_rows([card])
    assert row["unaccounted_mib"] == 600 and row["orphaned_mib"] == 600
    assert row["service_mib"] == 400
    assert [p["pid"] for p in row["processes"]] == [1, 2]


# ── service attribution (#414 follow-up) ─────────────────────────────────────
#
# Live on asterAIx the first version reported 10 440 MiB "unregistered" on an idle
# card. All of it was explainable: four gunicorn workers of a neighbouring RAG
# service, and one of our own trocr engines. A number that is permanently large
# for good reasons is a number people stop reading — which is how a sixteen-hour
# orphan stays invisible.

def test_the_unit_comes_from_the_cgroup(tmp_path, monkeypatch):
    def fake_open(path, *a, **k):
        import io
        shapes = {
            "/proc/1/cgroup":
                "0::/user.slice/user-1007.slice/user@1007.service/app.slice/atr-trocr.service\n",
            "/proc/2/cgroup": "0::/system.slice/gunicorn.service\n",
            "/proc/3/cgroup": "0::/user.slice/user-1007.slice/session-3.scope\n",
        }
        if path in shapes:
            return io.StringIO(shapes[path])
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", fake_open)
    assert gpu._unit_of(1) == "atr-trocr.service"
    assert gpu._unit_of(2) == "gunicorn.service"
    assert gpu._unit_of(3) is None          # a login shell is not a service
    assert gpu._unit_of(99) is None         # no such pid


def test_our_engine_is_marked_and_a_foreign_service_is_not(procs, monkeypatch):
    monkeypatch.setattr(gpu, "_smi", _smi_stub(
        [["100", "1600", "GPU-aaa"], ["200", "2610", "GPU-aaa"]]))
    procs[100] = ("tobias", 4600.0, "trocr", 1, "atr-trocr.service")
    procs[200] = ("change", 2251639.0, "gunicorn", 1, "gunicorn.service")
    ours, theirs = gpu.inspect({}).__getitem__(0).processes
    assert ours.service == "atr-trocr.service" and ours.own_service is True
    assert theirs.service == "gunicorn.service" and theirs.own_service is False
    # Neither belongs to a training job; only one of them is somebody else's.
    assert ours.registered is False and theirs.registered is False


def test_an_orphan_has_no_unit_and_stays_unaccounted(procs, monkeypatch):
    monkeypatch.setattr(gpu, "_smi", _smi_stub([["2743851", "27530", "GPU-aaa"]]))
    held = gpu.inspect({})[0].processes[0]
    assert held.service is None and held.own_service is False
    assert held.orphaned is True
