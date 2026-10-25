"""A measurement takes itself down, also after an abort (#183).

The environment of one measurement is 19–27 GB (venv, merged model, pages) and
a vLLM on a whole card. It was removed by hand, and the one time it was not it
lay there unnoticed — idhefix at 260 GB (#143). These pin:

* the directory is the measurement's, with a marker, and the merged model goes
  into it (``ATR_VLLM_MERGED_DIR``) rather than into the shared cache;
* the server is stopped by pid and process group, never by pattern, with
  SIGKILL after a grace period and its children seen before the first signal;
* the card is waited for, and an unreadable card is "not checked", not free;
* success and interrupts delete, a failed step and ``--keep`` keep, and a
  kept directory says where it is and how to remove it;
* ``gc`` reports what lies around and removes only orphans whose server is
  gone — never an unattributed directory, never a live one;
* and the whole thing, run as a real bash script with a fake ``nvidia-smi``
  and a server whose child ignores SIGTERM, leaves nothing behind.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from atr_serving import eval_workspace as ew  # noqa: E402
from atr_serving.gpu import Card, Process  # noqa: E402

import eval_workspace as cli  # noqa: E402

MINE = os.getpid()
DEAD = 4_194_305


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Never the developer's own ~/atr-eval or ~/.atr-eval."""
    monkeypatch.setenv("ATR_EVAL_ROOT", str(tmp_path / "atr-eval"))
    monkeypatch.setenv("ATR_EVAL_LEASE_DIR", str(tmp_path / "leases"))


def card(index=0, processes=()):
    return Card(index=index, name="NVIDIA A40", memory_total_mib=46_068,
                memory_used_mib=100, memory_free_mib=45_968, utilisation_pct=0,
                memory_utilisation_pct=0, processes=list(processes))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


# ── the directory ───────────────────────────────────────────────────────────
def test_a_workspace_is_a_marked_directory_under_the_root():
    ws = ew.create("qwen3vl-medieval-german-v3", MINE)
    assert ws.path.parent == ew.workspace_root()
    assert ws.path.name.endswith(f"-qwen3vl-medieval-german-v3-{MINE}")
    marker = json.loads(ws.marker.read_text())
    assert marker["pid"] == MINE and marker["state"] == "running"
    assert marker["model"] == "qwen3vl-medieval-german-v3"
    assert (ws.path / "merged").is_dir()
    assert ew.load(ws.path).pid == MINE


def test_the_merged_model_goes_into_the_workspace_not_the_shared_cache(tmp_path):
    """merge_loras writes to Settings.vllm_merged_dir, ~/atr-cache/vllm-merged by
    default — outside anything a cleanup would delete. The 8–17 GB of a merge
    are the biggest part of a measurement."""
    ws = ew.create("odd name/with 'quote'", MINE, root=tmp_path / "with space")
    out = subprocess.run(["bash", "-c", ws.shell() + '; printf "%s\\n%s\\n" "$WS" "$ATR_VLLM_MERGED_DIR"'
                          '; bash -c \'echo "$ATR_VLLM_MERGED_DIR"\''],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[0] == str(ws.path)
    assert out[1] == out[2] == str(ws.path / "merged"), "exported, so merge_loras sees it"
    from atr_serving.config import Settings
    assert Settings(_env_file=None).vllm_merged_dir != ws.path / "merged"
    os.environ["ATR_VLLM_MERGED_DIR"] = out[1]
    try:
        assert Settings(_env_file=None).vllm_merged_dir == ws.path / "merged"
    finally:
        del os.environ["ATR_VLLM_MERGED_DIR"]


def test_remove_refuses_anything_that_is_not_a_workspace(tmp_path):
    ws = ew.create("m", MINE)
    elsewhere = ew.Workspace(path=tmp_path / "precious", pid=MINE)
    elsewhere.path.mkdir()
    (elsewhere.path / ew.MARKER).write_text("{}")
    with pytest.raises(ValueError, match="not directly under"):
        ew.remove(elsewhere)
    assert elsewhere.path.exists()
    ws.marker.unlink()
    with pytest.raises(ValueError, match="no measurement.json"):
        ew.remove(ws)
    assert ws.path.exists()


# ── the decision ────────────────────────────────────────────────────────────
def _finish(ws, status, **kw):
    released = []
    done = ew.finish(ws, status, gpu=1,
                     stop=lambda w: ("stopped", [111, 112]),
                     wait=lambda g, pids: ("free", f"gpu {g}: 45968 MiB free"),
                     release=lambda g, pid: released.append((g, pid)) or True, **kw)
    return done, released


@pytest.mark.parametrize("status,kept", [(0, False), (130, False), (143, False),
                                         (129, False), (1, True), (2, True), (137, True)])
def test_success_and_interrupts_delete_a_failed_step_keeps(status, kept):
    ws = ew.create("m", MINE)
    (ws.path / "merged" / "w.bin").write_bytes(b"x" * 4096)
    done, released = _finish(ws, status)
    assert done.kept is kept
    assert ws.path.exists() is kept
    assert released == [(1, MINE)], "the lease is released whatever happens to the directory"


def test_keep_keeps_a_success():
    ws = ew.create("m", MINE)
    done, _ = _finish(ws, 0, keep=True)
    assert done.kept and ws.path.exists()
    assert ew.load(ws.path).state == "kept"


def test_a_kept_directory_says_where_how_big_and_how_to_remove_it():
    ws = ew.create("m", MINE)
    (ws.path / "merged" / "w.bin").write_bytes(b"x" * 1_000_000)
    done, _ = _finish(ws, 1)
    text = "\n".join(done.lines)
    assert f"KEPT {ws.path}" in text and "a step failed (status 1)" in text
    assert "MB" in text
    assert f"rm -rf '{ws.path}'" in text
    marker = ew.load(ws.path)
    assert marker.state == "kept" and marker.status == 1


def test_the_order_is_server_card_lease_directory():
    ws = ew.create("m", MINE)
    seen = []
    ew.finish(ws, 0, gpu=0,
              stop=lambda w: seen.append(("stop", w.path.exists())) or ("stopped", [5]),
              wait=lambda g, pids: seen.append(("wait", pids)) or ("free", ""),
              release=lambda g, pid: seen.append(("release", g)) or True)
    assert seen == [("stop", True), ("wait", [5]), ("release", 0)]
    assert not ws.path.exists()


def test_a_busy_card_is_a_warning_and_a_non_zero_exit(monkeypatch, capsys):
    ws = ew.create("m", MINE)
    monkeypatch.setattr(ew, "stop_server", lambda w, grace_s: ("would not die", [7]))
    monkeypatch.setattr(ew, "wait_card_free",
                        lambda g, pids, timeout_s: ("busy", "gpu 0 still holds pid 7"))
    assert cli.main(["finish", str(ws.path), "--status", "0", "--gpu", "0"]) == 3
    assert "WARNING: the card is not free" in capsys.readouterr().err


# ── stopping the server ─────────────────────────────────────────────────────
class Procs:
    """Fake processes: which are alive, what each signal does to them."""

    def __init__(self, alive, ignores_term=()):
        self.alive = set(alive)
        self.ignores_term = set(ignores_term)
        self.sent = []

    def kill(self, pid, sig):
        self.sent.append(("kill", pid, sig))
        if sig == signal.SIGKILL or pid not in self.ignores_term:
            self.alive.discard(pid)

    def killpg(self, pgid, sig):
        self.sent.append(("killpg", pgid, sig))

    def is_alive(self, pid):
        return pid in self.alive


def _stop(ws, procs, family=lambda pid: [], group=lambda pgid: []):
    clock = Clock()
    return ew.stop_server(ws, grace_s=5, kill=procs.kill, killpg=procs.killpg,
                          alive=procs.is_alive, family=family, group=group,
                          sleep=clock.sleep, clock=clock)


def test_the_server_and_its_children_are_stopped_by_pid():
    ws = ew.Workspace(path=Path("/x"), pid=MINE, server_pid=100, server_pgid=100)
    procs = Procs({100, 101})
    verdict, pids = _stop(ws, procs, family=lambda pid: [101])
    assert verdict == "stopped" and pids == [100, 101]
    assert ("killpg", 100, signal.SIGTERM) in procs.sent
    assert not any(s[2] == signal.SIGKILL for s in procs.sent)


def test_a_child_that_ignores_term_gets_kill_after_the_grace():
    """vLLM's EngineCore outliving `vllm serve` holds the card just the same."""
    ws = ew.Workspace(path=Path("/x"), pid=MINE, server_pid=100, server_pgid=100)
    procs = Procs({100, 101}, ignores_term={101})
    verdict, _ = _stop(ws, procs, family=lambda pid: [101])
    assert verdict == "killed"
    assert ("kill", 101, signal.SIGKILL) in procs.sent and not procs.alive


def test_children_are_seen_before_the_first_signal():
    """Once the parent is gone its children are reparented, and no ppid walk
    finds them any more."""
    ws = ew.Workspace(path=Path("/x"), pid=MINE, server_pid=100)
    procs = Procs({100, 101}, ignores_term={101})
    calls = []

    def family(pid):
        calls.append(sorted(procs.alive))
        return [101] if 100 in procs.alive else []
    _stop(ws, procs, family=family)
    assert calls == [[100, 101]], "asked once, before anything was signalled"
    assert not procs.alive


def test_a_group_the_server_does_not_lead_is_never_signalled():
    """A server started without setsid shares the shell's group; killpg on it
    would take the shell — and the trap doing the cleanup — down too."""
    ws = ew.Workspace(path=Path("/x"), pid=MINE, server_pid=100, server_pgid=None)
    procs = Procs({100})
    _stop(ws, procs)
    assert not any(s[0] == "killpg" for s in procs.sent)


def test_the_group_members_count_even_when_reparented():
    ws = ew.Workspace(path=Path("/x"), pid=MINE, server_pid=100, server_pgid=100)
    procs = Procs({100, 150})
    verdict, pids = _stop(ws, procs, group=lambda pgid: [100, 150] if pgid == 100 else [])
    assert pids == [100, 150] and verdict == "stopped"


def test_no_server_and_a_gone_server_are_said():
    procs = Procs(set())
    assert _stop(ew.Workspace(path=Path("/x"), pid=MINE), procs) == ("no server", [])
    ws = ew.Workspace(path=Path("/x"), pid=MINE, server_pid=100)
    assert _stop(ws, procs)[0] == "already gone"
    assert procs.sent == [], "nothing is signalled that is not there"


def test_record_server_waits_for_setsid():
    """$! is known before the child has run setsid."""
    ws = ew.create("m", MINE)
    answers = iter([555, 555, 100])
    ew.record_server(ws, 100, getpgid=lambda pid: next(answers), sleep=lambda s: None)
    assert ew.load(ws.path).server_pgid == 100
    ew.record_server(ws, 100, getpgid=lambda pid: 555, sleep=lambda s: None, settle_s=0.1)
    assert ew.load(ws.path).server_pgid is None


def _fake_proc(tmp_path, rows):
    proc = tmp_path / "proc"
    for pid, (ppid, pgid) in rows.items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "stat").write_text(f"{pid} (py thon) S {ppid} {pgid} {pgid} 0\n")
    (proc / "self").mkdir(parents=True, exist_ok=True)
    return proc


def test_descendants_and_group_members_are_read_off_proc(tmp_path):
    proc = _fake_proc(tmp_path, {100: (1, 100), 101: (100, 100), 102: (101, 100),
                                 150: (1, 100), 200: (1, 200)})
    assert ew.descendants(100, proc) == [101, 102]
    assert ew.group_members(100, proc) == [100, 101, 102, 150]
    assert ew.group_members(None, proc) == []


# ── waiting for the card ────────────────────────────────────────────────────
def test_the_card_is_waited_for_until_our_processes_are_gone():
    clock = Clock()
    readings = iter([[card(0, [Process(pid=101, used_mib=9000)])],
                     [card(0, [Process(pid=101, used_mib=9000)])],
                     [card(0)]])
    verdict, detail = ew.wait_card_free(0, [100, 101], inspect=lambda: next(readings),
                                        sleep=clock.sleep, clock=clock)
    assert verdict == "free" and clock.t == 2 * ew.POLL_S


def test_a_card_still_holding_ours_after_the_wait_is_busy():
    clock = Clock()
    held = [card(0, [Process(pid=101, used_mib=9000)])]
    verdict, detail = ew.wait_card_free(0, [101], inspect=lambda: held, timeout_s=5,
                                        sleep=clock.sleep, clock=clock)
    assert verdict == "busy" and "pid 101 (9000 MiB)" in detail


def test_an_unreadable_card_is_not_checked_never_free():
    def broken():
        raise FileNotFoundError("nvidia-smi is not on PATH")
    assert ew.wait_card_free(0, [1], inspect=broken)[0] == "not checked"
    assert ew.wait_card_free(3, [1], inspect=lambda: [card(0)])[0] == "not checked"
    assert ew.wait_card_free(None, [1], inspect=lambda: [card(0)])[0] == "not checked"


def test_a_process_nvidia_smi_cannot_name_is_unknown_not_free():
    clock = Clock()
    verdict, detail = ew.wait_card_free(
        0, [101], inspect=lambda: [card(0, [Process(pid=None, used_mib=9000)])],
        timeout_s=3, sleep=clock.sleep, clock=clock)
    assert verdict == "not checked" and "cannot name" in detail


def test_a_foreign_process_is_named_not_waited_for():
    clock = Clock()
    verdict, detail = ew.wait_card_free(
        0, [101], inspect=lambda: [card(0, [Process(pid=4242, used_mib=20000)])],
        sleep=clock.sleep, clock=clock)
    assert verdict == "free" and clock.t == 0
    assert "not ours: pid 4242 20000 MiB" in detail


# ── gc ──────────────────────────────────────────────────────────────────────
def _workspaces(alive_pids):
    live_ws = ew.create("live", MINE)
    orphan = ew.create("orphan", DEAD)
    orphan_running = ew.create("orphan-server", DEAD - 1)
    orphan_running.server_pid = 777
    orphan_running.save()
    kept = ew.create("kept", DEAD - 2)
    kept.state = "kept"
    kept.reason = "a step failed (status 1)"
    kept.save()
    away = ew.create("away", DEAD - 3)
    away.host = "asteraix-not-this-host"
    away.save()
    stray = ew.workspace_root() / "someone-elses-dir"
    stray.mkdir()
    (stray / "big.bin").write_bytes(b"x" * 10_000)
    return live_ws, orphan, orphan_running, kept, away, stray


def test_scan_tells_live_orphaned_kept_elsewhere_and_unattributed_apart():
    live_ws, orphan, orphan_running, kept, away, stray = _workspaces({MINE, 777})
    entries = {e.path: e for e in ew.scan(alive=lambda pid: pid in {MINE, 777},
                                          group=lambda pgid: [])}
    assert entries[live_ws.path].verdict == "in use"
    assert entries[orphan.path].verdict == "orphaned"
    assert entries[orphan_running.path].verdict == "orphaned"
    assert entries[orphan_running.path].server_alive
    assert "STILL RUNNING (stop it: kill -TERM 777)" in entries[orphan_running.path].line()
    assert entries[kept.path].verdict == "kept"
    assert entries[away.path].verdict == "elsewhere", "a pid from another host means nothing here"
    assert entries[stray].verdict == "unattributed" and entries[stray].size >= 10_000


def test_gc_reports_and_removes_nothing_without_apply(capsys):
    paths = _workspaces({MINE})
    assert cli.main(["gc"]) == 0
    out = capsys.readouterr().out
    assert "orphaned" in out and "6 directories" in out
    assert all(p.path.exists() if hasattr(p, "path") else p.exists() for p in paths)


def test_gc_apply_removes_only_orphans_whose_server_is_gone(monkeypatch, capsys):
    live_ws, orphan, orphan_running, kept, away, stray = _workspaces({MINE, 777})
    scan = ew.scan
    monkeypatch.setattr(ew, "scan", lambda root: scan(root, alive=lambda pid: pid in {MINE, 777},
                                                      group=lambda pgid: []))
    assert cli.main(["gc", "--apply"]) == 0
    assert not orphan.path.exists()
    for still in (live_ws.path, orphan_running.path, kept.path, away.path, stray):
        assert still.exists(), still
    assert cli.main(["gc", "--apply", "--kept"]) == 0
    assert not kept.path.exists() and stray.exists() and live_ws.path.exists()


# ── the real thing: bash, traps, a server, a fake nvidia-smi ────────────────
FAKE_SMI = """#!/bin/bash
case "$1" in
  --query-gpu=uuid) echo "GPU-0";;
  --query-gpu=*) echo "0, NVIDIA A40, 46068, 100, 45968, 0, 0";;
  --query-compute-apps=*)
    [ -f "$FAKE_GPU_PIDS" ] && while read -r p; do
      kill -0 "$p" 2>/dev/null && echo "$p, 9000, GPU-0"
    done < "$FAKE_GPU_PIDS";;
esac
exit 0
"""

#: A server whose child ignores SIGTERM — the EngineCore that outlives its parent
#: — and a grandchild already reparented before anything is stopped: no ppid
#: walk finds it, only the process group the server leads (setsid).
SERVER = """import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"])
detached = subprocess.run(["bash", "-c", sys.executable
    + " -c 'import time; time.sleep(600)' >/dev/null 2>&1 & echo $!"],
    capture_output=True, text=True).stdout.strip()
with open(os.environ["FAKE_GPU_PIDS"], "a") as fh:
    fh.write(f"{os.getpid()}\\n{child.pid}\\n{detached}\\n")
open(os.environ["SERVER_READY"], "w").close()
time.sleep(600)
"""


def _measure(tmp_path, body: str, extra_env=None):
    (tmp_path / "bin").mkdir()
    smi = tmp_path / "bin" / "nvidia-smi"
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)
    (tmp_path / "server.py").write_text(SERVER)
    script = tmp_path / "measure.sh"
    script.write_text(textwrap.dedent(f"""\
        set -euo pipefail
        source {ROOT}/scripts/eval_cleanup.sh
        """) + textwrap.dedent(body))
    pids = tmp_path / "gpu-pids"
    pids.write_text("")
    env = dict(os.environ, PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
               ATR_EVAL_PYTHON=sys.executable, FAKE_GPU_PIDS=str(pids),
               SERVER_READY=str(tmp_path / "ready"), SERVER=str(tmp_path / "server.py"),
               ATR_EVAL_TERM_GRACE_S="2", ATR_EVAL_CARD_WAIT_S="10")
    env.update(extra_env or {})
    done = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True,
                          timeout=120)
    started = [int(p) for p in pids.read_text().split()]
    return done, started


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie is dead for the card; it only waits for its reaper.
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


SERVE_AND = """\
eval_begin test-model {keep}
head -c 2000000 /dev/zero > "$ATR_VLLM_MERGED_DIR/model.safetensors"
eval_serve {py} "$SERVER"
while [ ! -e "$SERVER_READY" ]; do sleep 0.1; done
{then}
"""

pytestmark_bash = pytest.mark.skipif(
    subprocess.run(["which", "setsid"], capture_output=True).returncode != 0,
    reason="setsid (util-linux) is not installed")


@pytestmark_bash
@pytest.mark.parametrize("then,status,kept", [
    ("true", 0, False),
    ("false", 1, True),
    ("kill -TERM $$; sleep 30", 143, False),
    ("kill -INT $$; sleep 30", 130, False),
])
def test_a_real_measurement_leaves_nothing_unexpected(tmp_path, then, status, kept):
    done, started = _measure(tmp_path, SERVE_AND.format(keep="", py=sys.executable, then=then))
    err = done.stderr
    assert done.returncode == status, err
    assert len(started) == 3
    assert not any(_alive(p) for p in started), f"server or its child survived:\n{err}"
    assert "server: killed" in err, "the child ignored TERM and needed KILL"
    assert "card: free" in err, err
    assert "lease on gpu 0: released" in err
    assert not list((tmp_path / "leases").glob("gpu*.lock")), "the lease is gone"
    remaining = list((tmp_path / "atr-eval").iterdir())
    assert len(remaining) == int(kept), remaining
    if kept:
        assert "KEPT" in err and "rm -rf" in err
        assert (remaining[0] / "server.log").exists(), "the server's log is kept with it"


@pytestmark_bash
def test_keep_keeps_a_successful_measurement(tmp_path):
    done, started = _measure(tmp_path, SERVE_AND.format(keep="--keep", py=sys.executable,
                                                        then="true"))
    assert done.returncode == 0, done.stderr
    assert not any(_alive(p) for p in started), "kept is the directory, never the server"
    assert len(list((tmp_path / "atr-eval").iterdir())) == 1


@pytestmark_bash
def test_a_refused_card_leaves_nothing_behind(tmp_path):
    """A card with a foreign process is not overbooked; the refusal comes
    before the directory, so there is nothing to clean up."""
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (tmp_path / "gpu-pids-pre").write_text(f"{foreign.pid}\n")
        done, _ = _measure(tmp_path, "eval_begin test-model\necho should-not-run\n",
                           extra_env={"FAKE_GPU_PIDS": str(tmp_path / "gpu-pids-pre")})
    finally:
        foreign.kill()
        foreign.wait()
    assert done.returncode != 0 and "should-not-run" not in done.stdout
    assert f"pid {foreign.pid}" in done.stderr
    assert not (tmp_path / "atr-eval").exists(), "no directory was made"
    assert not list(tmp_path.glob("leases/gpu*.lock")), "no lease was written"
