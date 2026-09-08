"""#414: /train/gpu — what holds GPU memory, and whether the trainer knows it.

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


# ── the route ────────────────────────────────────────────────────────────────

from fastapi.testclient import TestClient          # noqa: E402

from atr_serving.app import create_app             # noqa: E402
from atr_serving.clients import TrainerError       # noqa: E402
from atr_serving.config import Settings            # noqa: E402

KEY = "test-key"
AUTH = {"X-API-Key": KEY}


class _Trainer:
    def __init__(self, jobs=None, raises=None):
        self.jobs = jobs or []
        self.raises = raises

    async def list_jobs(self):
        if self.raises:
            raise self.raises
        return {"jobs": self.jobs}


def _client(trainer, monkeypatch, cards):
    monkeypatch.setattr(gpu, "inspect", lambda job_pids: cards(job_pids))
    app = create_app(Settings(api_key=KEY, require_auth=True))
    app.state.trainer_client = trainer
    return TestClient(app)


def _two_cards(job_pids):
    """One idle card holding an orphan and a stray, one card doing real work."""
    a = gpu.Card(0, "A100", 40960, 34300, 6660, 0, 0)
    a.processes = [
        gpu.Process(pid=2743851, used_mib=27530, orphaned=True, registered=False),
        gpu.Process(pid=7777, used_mib=6000, registered=7777 in job_pids,
                    job_id=job_pids.get(7777), user="tobias", age_s=108000.0,
                    command="ketos train"),
        # One of ours, holding memory legitimately.
        gpu.Process(pid=8888, used_mib=1600, service="atr-trocr.service",
                    own_service=True, user="tobias", age_s=4600.0,
                    command="trocr engine"),
    ]
    b = gpu.Card(1, "A100", 40960, 1600, 39360, 97, 43)
    b.processes = [gpu.Process(pid=4242, used_mib=1600, registered=4242 in job_pids,
                               job_id=job_pids.get(4242), user="tobias",
                               age_s=900.0, command="python train.py")]
    return [a, b]


def test_the_endpoint_needs_the_api_key(monkeypatch):
    c = _client(_Trainer(), monkeypatch, _two_cards)
    assert c.get("/train/gpu").status_code == 401


def test_it_totals_what_no_job_accounts_for(monkeypatch):
    c = _client(_Trainer(jobs=[{"id": "job-a", "pid": 4242}]), monkeypatch, _two_cards)
    body = c.get("/train/gpu", headers=AUTH).json()
    idle, busy = body["cards"]
    # The engine's 1600 MiB is explainable and stays out of the total.
    assert idle["unaccounted_mib"] == 27530 + 6000
    assert idle["service_mib"] == 1600
    assert idle["orphaned_mib"] == 27530
    assert busy["unaccounted_mib"] == 0
    assert body["job_attribution_available"] is True
    assert body["known_job_pids"] == 1


def test_an_unreachable_trainer_still_reports_the_cards(monkeypatch):
    """Losing attribution must not lose the memory figures."""
    c = _client(_Trainer(raises=TrainerError(503, "trainer down")),
                monkeypatch, _two_cards)
    body = c.get("/train/gpu", headers=AUTH).json()
    assert body["job_attribution_available"] is False
    assert body["cards"][0]["memory_used_mib"] == 34300
    # Everything unregistered, and the flag above says why.
    assert body["cards"][1]["unaccounted_mib"] == 1600


def test_a_box_without_nvidia_smi_says_so(monkeypatch):
    def boom(job_pids):
        raise FileNotFoundError("nvidia-smi is not on PATH")
    c = _client(_Trainer(), monkeypatch, boom)
    r = c.get("/train/gpu", headers=AUTH)
    assert r.status_code == 503 and "nvidia-smi" in r.json()["detail"]


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
