"""#213: who holds the swap is read off /proc, repeatably and without touching anything.

A fake /proc under tmp_path stands in for idhefix on 11.10.2026: a June training
run swapped out to 6 GiB with 14 MiB resident and an .arrow on the share still
open, a vLLM from the boot, a live gunicorn, and processes without swap that
must not appear. The script is stdlib and runs under the login shell's Python.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import swap_holders as sh  # noqa: E402

BOOT = 1_700_000_000
TICKS = 100


def _proc(tmp_path: Path, pids: dict, *, meminfo=True, pressure=True) -> Path:
    proc = tmp_path / "proc"
    proc.mkdir(parents=True)
    (proc / "stat").write_text(f"cpu 1 2 3\nbtime {BOOT}\n")
    if meminfo:
        (proc / "meminfo").write_text("MemTotal:  263716000 kB\nSwapTotal:  10485756 kB\n"
                                      "SwapFree:    848000 kB\n")
    if pressure:
        (proc / "pressure").mkdir()
        (proc / "pressure" / "memory").write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=5\n"
                                                  "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
    for pid, spec in pids.items():
        d = proc / str(pid)
        d.mkdir()
        (d / "status").write_text(
            f"Name:\t{spec['name']}\nState:\t{spec.get('state', 'S (sleeping)')}\n"
            f"Uid:\t{spec.get('uid', os.getuid())}\t{spec.get('uid', os.getuid())}\n"
            f"VmRSS:\t{spec.get('rss_kib', 0)} kB\nVmSwap:\t{spec.get('swap_kib', 0)} kB\n")
        (d / "cmdline").write_bytes(spec.get("cmdline", spec["name"]).replace(" ", "\0").encode() + b"\0")
        start_ticks = spec.get("start_ticks", 0)
        utime = spec.get("cpu", 0)
        (d / "stat").write_text(f"{pid} ({spec['name']}) S 1 1 1 0 -1 0 0 0 0 0 {utime} 7 0 0 20 0 1 0 "
                                f"{start_ticks} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
        fd = d / "fd"
        fd.mkdir()
        (d / "fdinfo").mkdir()
        flags = spec.get("fd_flags", {})
        for i, target in enumerate(spec.get("fds", [])):
            os.symlink(target, fd / str(i))
            (d / "fdinfo" / str(i)).write_text(f"pos:\t0\nflags:\t{flags.get(i, '0100000')}\nmnt_id:\t30\n")
    (proc / "self").mkdir()           # not a pid: must be skipped
    return proc


IDHEFIX = {
    1216483: {"name": "python", "swap_kib": 6057 * 1024, "rss_kib": 14 * 1024, "start_ticks": 100 * 86400 * TICKS,
              "cmdline": "kraken-env/bin/python /home/tobias/Repo/kraken-htr-space/train.py prepare",
              "fds": ["/dev/pts/3", "/mnt/wbkolleg_dh_1/Data_Kraken-training/ground_truth/chunks/0497440.arrow",
                      "/dev/shm/torch_1216483_x (deleted)"]},
    2274: {"name": "python3", "swap_kib": 563 * 1024, "rss_kib": 11 * 1024, "start_ticks": 0, "uid": 0,
           "cmdline": "python3 -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
    4100: {"name": "gunicorn", "swap_kib": 651 * 1024, "rss_kib": 8_400_000, "start_ticks": 50 * 86400 * TICKS,
           "state": "R (running)", "cmdline": "gunicorn: worker [rag]"},
    5000: {"name": "sshd", "swap_kib": 0, "rss_kib": 9000},
}


def test_holders_are_the_processes_with_swap_largest_first_with_their_share_files(tmp_path):
    proc = _proc(tmp_path, IDHEFIX)
    now = BOOT + 122 * 86400 + 14 * 3600
    rows = sh.holders(proc, "/mnt/wbkolleg_dh_1", now=now, ticks=TICKS)
    assert [h.pid for h in rows] == [1216483, 4100, 2274], "sshd holds no swap and is not listed"
    june = rows[0]
    assert round(june.mib) == 6057 and june.rss_kib == 14 * 1024 and june.state == "S"
    assert june.command.startswith("kraken-env/bin/python") and "train.py prepare" in june.command
    assert june.share_files == ["/mnt/wbkolleg_dh_1/Data_Kraken-training/ground_truth/chunks/0497440.arrow"], \
        "the pts and the deleted shm are not share files"
    assert int(june.age_s // 86400) == 22, "started 100 days after boot, now 122 days 14 h after boot"
    assert rows[2].age_s == pytest.approx(122 * 86400 + 14 * 3600), "the vLLM from the boot"
    assert rows[2].user == "root" and rows[1].state == "R"


def test_fds_this_user_may_not_read_are_said_not_counted_as_none(tmp_path, monkeypatch):
    proc = _proc(tmp_path, {7: {"name": "x", "swap_kib": 1024, "fds": ["/mnt/wbkolleg_dh_1/a"]}})

    real = os.listdir

    def listdir(path):
        if str(path).endswith("/7/fd"):
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)
    monkeypatch.setattr(sh.os, "listdir", listdir)
    (holder,) = sh.holders(proc, "/mnt/wbkolleg_dh_1", now=BOOT + 10, ticks=TICKS)
    assert holder.share_files is None
    assert "fds not readable (run as root)" in holder.line("/mnt/wbkolleg_dh_1")


def test_the_summary_keeps_full_and_under_pressure_apart(tmp_path):
    proc = _proc(tmp_path, {})
    summary = sh.swap_summary(proc)
    assert summary["used_pct"] == 91.9 and summary["pressure_some"]["avg300"] == 0.0
    assert round(summary["total_mib"]) == 10240
    bare = sh.swap_summary(_proc(tmp_path / "bare", {}, meminfo=False, pressure=False))
    assert bare["total_mib"] is None and bare["pressure_some"] is None, "unreadable is None, not 0"


def test_main_prints_the_finding_and_touches_nothing(tmp_path, capsys):
    proc = _proc(tmp_path, IDHEFIX)
    before = sorted(p.name for p in proc.iterdir())
    assert sh.main(["--proc", str(proc), "--share", "/mnt/wbkolleg_dh_1", "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert "swap: 9412 of 10240 MiB in use (91.9 %)" in out
    assert "no pressure: the swap is full, not busy" in out
    assert "1216483" in out and "0497440.arrow" in out and "4100" in out
    assert "2274" not in out, "--top 2"
    assert "2 process(es) listed" in out
    assert sorted(p.name for p in proc.iterdir()) == before


def test_unreadable_proc_files_are_said_and_an_empty_box_is_said(tmp_path, capsys):
    proc = _proc(tmp_path / "a", {}, meminfo=False, pressure=False)
    assert sh.main(["--proc", str(proc)]) == 0
    out = capsys.readouterr().out
    assert "could not be read — nothing measured" in out and "unknown, not zero" in out
    assert "no process holds swapped memory" in out
    none = _proc(tmp_path / "b", {})
    (none / "meminfo").write_text("SwapTotal:  0 kB\nSwapFree:  0 kB\n")
    assert sh.main(["--proc", str(none)]) == 0
    out = capsys.readouterr().out
    assert "swap: none configured" in out
    assert "the swap is full" not in out, "no swap and no pressure is not a full swap"


# ── the rule: found by a timer, not by accident ─────────────────────────────
DAY = 86400


def _share(tmp_path: Path) -> Path:
    """A share with the June .arrow, unchanged since the 10th of June."""
    share = tmp_path / "share"
    arrow = share / "Data_Kraken-training" / "ground_truth" / "chunks" / "0497440.arrow"
    arrow.parent.mkdir(parents=True)
    arrow.write_bytes(b"x" * 4096)
    june = BOOT + 100 * DAY
    os.utime(arrow, (june, june))
    return share


def _idhefix(share: Path, **june) -> dict:
    """IDHEFIX with the arrow on *share*, opened for writing (O_WRONLY|O_CREAT|O_LARGEFILE)."""
    pids = {pid: dict(spec) for pid, spec in IDHEFIX.items()}
    pids[1216483].update(dict(cpu=91_000, fd_flags={1: "0100101"},
                              fds=["/dev/pts/3", str(share / "Data_Kraken-training/ground_truth/chunks/0497440.arrow")]),
                         **june)
    pids[2274].update(cpu=4_200)
    pids[4100].update(cpu=10_000_000)
    return pids


def _rule(tmp_path, proc, share, now, capsys):
    code = sh.main(["--rule", "--proc", str(proc), "--share", str(share),
                    "--state", str(tmp_path / "state" / "swap.json"), "--now", str(now)])
    return code, capsys.readouterr().out


def test_the_first_reading_names_candidates_and_never_calls_them_stale(tmp_path, capsys):
    share = _share(tmp_path)
    proc = _proc(tmp_path, _idhefix(share))
    code, out = _rule(tmp_path, proc, share, BOOT + 122 * DAY, capsys)
    assert code == 0, "idleness is unknown on a first reading: unknown is not stale"
    assert "CANDIDATE pid 1216483" in out and "CANDIDATE pid 2274" in out
    assert "idleness unknown, not idle" in out
    assert "4100" not in out, "the live gunicorn (8 GiB resident, running) is not a candidate"
    assert "STALE" not in out
    state = json.loads((tmp_path / "state" / "swap.json").read_text())
    assert state[f"1216483:{100 * DAY * TICKS}"]["cpu"] == 91_000 + 7


def test_a_week_without_cpu_time_makes_them_stale_and_the_exit_says_so(tmp_path, capsys):
    share = _share(tmp_path)
    proc = _proc(tmp_path, _idhefix(share))
    _rule(tmp_path, proc, share, BOOT + 122 * DAY, capsys)
    code, out = _rule(tmp_path, proc, share, BOOT + 129 * DAY, capsys)
    assert code == 1, "stale: the timer's unit fails, and shows in systemctl --failed"
    assert "STALE     pid 1216483" in out and "STALE     pid 2274" in out
    assert "CPU time unchanged for 7d 00h" in out
    assert "holds open for writing: " + str(share) in out and "0497440.arrow" in out
    assert "unchanged 29d" in out, "the arrow untouched since the 10th of June"
    assert "Nothing was signalled" in out


def test_less_than_a_week_idle_is_still_a_candidate(tmp_path, capsys):
    share = _share(tmp_path)
    proc = _proc(tmp_path, _idhefix(share))
    _rule(tmp_path, proc, share, BOOT + 122 * DAY, capsys)
    code, out = _rule(tmp_path, proc, share, BOOT + 125 * DAY, capsys)
    assert code == 0 and "CANDIDATE pid 1216483" in out and "3d 00h" in out


def test_a_process_whose_cpu_time_moved_is_not_listed(tmp_path, capsys):
    share = _share(tmp_path)
    _rule(tmp_path, _proc(tmp_path / "a", _idhefix(share)), share, BOOT + 122 * DAY, capsys)
    busy = _proc(tmp_path / "b", _idhefix(share, cpu=95_000))
    code, out = _rule(tmp_path, busy, share, BOOT + 130 * DAY, capsys)
    assert "1216483" not in out, "swapped out but working: not a corpse"
    assert "STALE     pid 2274" in out and code == 1
    # and the clock restarts: one more week unchanged from here is needed
    code, out = _rule(tmp_path, busy, share, BOOT + 131 * DAY, capsys)
    assert "CANDIDATE pid 1216483" in out


def test_a_reused_pid_is_a_new_process(tmp_path, capsys):
    share = _share(tmp_path)
    _rule(tmp_path, _proc(tmp_path / "a", _idhefix(share)), share, BOOT + 122 * DAY, capsys)
    reborn = _proc(tmp_path / "b", _idhefix(share, start_ticks=101 * DAY * TICKS))
    code, out = _rule(tmp_path, reborn, share, BOOT + 130 * DAY, capsys)
    assert "CANDIDATE pid 1216483" in out and "STALE     pid 1216483" not in out


def test_young_or_resident_holders_are_not_candidates(tmp_path, capsys):
    share = _share(tmp_path)
    pids = {10: {"name": "young", "swap_kib": 4_000_000, "rss_kib": 1024,
                 "start_ticks": 120 * DAY * TICKS},
            11: {"name": "small", "swap_kib": 100 * 1024, "rss_kib": 1, "start_ticks": 0},
            12: {"name": "resident", "swap_kib": 1_000_000, "rss_kib": 100_000, "start_ticks": 0}}
    code, out = _rule(tmp_path, _proc(tmp_path, pids), share, BOOT + 122 * DAY, capsys)
    assert code == 0 and "CANDIDATE" not in out and "no stale swap holder" in out


def test_an_unreadable_state_makes_idleness_unknown_not_idle(tmp_path, capsys):
    share = _share(tmp_path)
    proc = _proc(tmp_path, _idhefix(share))
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "swap.json").write_text("{half")
    code, out = _rule(tmp_path, proc, share, BOOT + 129 * DAY, capsys)
    assert code == 0 and "STALE" not in out and "CANDIDATE pid 1216483" in out


def test_a_file_open_only_for_reading_is_not_called_written(tmp_path, capsys):
    share = _share(tmp_path)
    proc = _proc(tmp_path, _idhefix(share, fd_flags={1: "0100000"}))
    holder = next(h for h in sh.holders(proc, str(share), now=BOOT + 122 * DAY, ticks=TICKS)
                  if h.pid == 1216483)
    assert holder.share_files and holder.share_writing == []


def test_idleness_adds_up_over_readings_rather_than_restarting_at_each(tmp_path, capsys):
    """Two readings four days apart are eight idle days, not four."""
    share = _share(tmp_path)
    proc = _proc(tmp_path, _idhefix(share))
    for day in (122, 126):
        _rule(tmp_path, proc, share, BOOT + day * DAY, capsys)
    code, out = _rule(tmp_path, proc, share, BOOT + 130 * DAY, capsys)
    assert code == 1 and "STALE     pid 1216483" in out and "8d 00h" in out
