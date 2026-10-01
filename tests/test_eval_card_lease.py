"""Two measurements on one box do not need to talk to each other (#184).

Two sessions were measuring on asteraix at the same time and divided the cards
and the ports **in a chat window** — card 0 / port 8299 against card 1 / port
8300. That works exactly as long as both sides are talking. Before that, a
measurement went through the production gateway and cost a real user three 502s
and a 503, because card 1 holds one VLM beside the engines and nobody asked who
needed it (#165).

These pin the four rules, and each of them is a thing that went wrong:

* a card is claimed in a file the other session can read;
* a stale lock is **taken over**, not respected — a lock nobody can clear is
  worse than no lock, because the next session works around it by hand;
* a free card is chosen, not configured;
* the production gateway is refused, and `--i-know` alone is not enough.
"""

import json
import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atr_serving import eval_lease  # noqa: E402
from atr_serving.eval_lease import (  # noqa: E402
    GATEWAY_PORT,
    JOURNAL_WINDOW_S,
    Lease,
    LeaseError,
    claim,
    free_port,
    gateway_refusal,
    read_lease,
    release,
    write_lease,
)
from atr_serving.gpu import Card, Process  # noqa: E402

NOW = 1_800_000_000.0
MINE = os.getpid()
#: A pid that cannot exist: Linux caps at 2^22 by default.
DEAD = 4_194_305


@pytest.fixture(autouse=True)
def leases_in_a_tmp_dir(tmp_path, monkeypatch):
    """Never the developer's own ~/.atr-eval: these write lock files."""
    monkeypatch.setenv("ATR_EVAL_LEASE_DIR", str(tmp_path / "atr-eval"))
    monkeypatch.setattr("atr_serving.eval_lease.gpu_probe.descends_from",
                        lambda pid, ancestor: False)


def card(index: int, free: int = 46_000, processes=()) -> Card:
    return Card(index=index, name="NVIDIA A40", memory_total_mib=46_068,
                memory_used_mib=46_068 - free, memory_free_mib=free,
                utilisation_pct=0, memory_utilisation_pct=0,
                processes=list(processes))


def held_by(pid: int, used_mib: int = 20_000, **kw) -> Process:
    return Process(pid=pid, used_mib=used_mib, **kw)


# ── the lock is a file the other session can read ───────────────────────────
def test_a_claim_writes_what_the_next_reader_needs():
    lease = claim(pid=MINE, model="qwen3vl-medieval-german-v3",
                  cards=[card(1)], port=8301)

    written = json.loads(lease.path.read_text(encoding="utf-8"))
    assert written["pid"] == MINE
    assert written["port"] == 8301
    assert written["model"] == "qwen3vl-medieval-german-v3"
    assert written["started_at"].endswith("Z")
    assert written["host"]


def test_the_lock_is_named_after_the_card():
    lease = claim(pid=MINE, cards=[card(1)], port=8301)

    assert lease.path.name == "gpu1.lock"


def test_a_lease_round_trips():
    write_lease(Lease(gpu=0, pid=MINE, port=8299, model="m", started_at="t"))

    got = read_lease(0)
    assert (got.gpu, got.pid, got.port, got.model) == (0, MINE, 8299, "m")


def test_the_lock_is_never_seen_half_written(tmp_path):
    """Written whole and moved into place, so a reader does not meet the state
    `read_lease` has to tolerate."""
    directory = eval_lease.lease_dir()
    write_lease(Lease(gpu=0, pid=MINE, port=8299))

    assert [p.name for p in directory.iterdir()] == ["gpu0.lock"]


def test_an_unreadable_lock_counts_as_no_lock():
    """A half-written file is the signature of a claim that died; treating it
    as a held card would make the card unusable until somebody deleted it."""
    directory = eval_lease.lease_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "gpu0.lock").write_text("{not json", encoding="utf-8")

    assert read_lease(0) is None
    assert claim(pid=MINE, cards=[card(0)], port=8299).gpu == 0


def test_a_lock_without_a_pid_counts_as_no_lock():
    directory = eval_lease.lease_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "gpu0.lock").write_text('{"port": 8299}', encoding="utf-8")

    assert read_lease(0) is None


# ── a live lock is respected ────────────────────────────────────────────────
def test_a_card_held_by_a_live_measurement_is_not_taken():
    """The case the chat window was solving."""
    write_lease(Lease(gpu=1, pid=MINE, port=8299, model="theirs",
                      started_at="2026-10-01T11:00:00Z"))

    with pytest.raises(LeaseError, match="held by pid"):
        claim(pid=MINE, cards=[card(1)], gpu=1)


def test_the_refusal_names_who_holds_it_and_since_when():
    """So the other session does not have to be asked."""
    write_lease(Lease(gpu=1, pid=MINE, port=8299, model="theirs",
                      started_at="2026-10-01T11:00:00Z"))

    with pytest.raises(LeaseError) as raised:
        claim(pid=MINE, cards=[card(1)], gpu=1)

    message = str(raised.value)
    assert "theirs" in message
    assert "2026-10-01T11:00:00Z" in message
    assert ":8299" in message


def test_a_second_card_is_chosen_when_the_first_is_held():
    write_lease(Lease(gpu=1, pid=MINE, port=8299))

    lease = claim(pid=MINE, cards=[card(0), card(1)], port=8300)

    assert lease.gpu == 0


def test_a_lock_held_by_someone_elses_process_is_respected(monkeypatch):
    """`os.kill(pid, 0)` raises PermissionError for another user's live
    process, and that means alive — not gone."""
    def kill(_pid, _sig):
        raise PermissionError

    monkeypatch.setattr(os, "kill", kill)
    write_lease(Lease(gpu=1, pid=12345, port=8299))

    assert eval_lease.live(12345)
    with pytest.raises(LeaseError, match="held by pid 12345"):
        claim(pid=MINE, cards=[card(1)], gpu=1)


# ── a stale lock is taken over ──────────────────────────────────────────────
def test_a_dead_holders_lock_is_taken_over():
    """A measurement that was killed leaves its file behind, and a lock nobody
    can clear is worse than no lock."""
    write_lease(Lease(gpu=1, pid=DEAD, port=8299, model="died"))

    lease = claim(pid=MINE, cards=[card(1)], port=8300)

    assert lease.gpu == 1
    assert lease.pid == MINE
    assert lease.took_over == DEAD


def test_the_takeover_is_recorded_for_the_next_reader():
    """"The card was not simply idle — somebody's measurement died on it" is
    worth knowing, and it is the kind of thing nobody writes down by hand."""
    write_lease(Lease(gpu=1, pid=DEAD, port=8299))

    lease = claim(pid=MINE, cards=[card(1)], port=8300)

    assert json.loads(lease.path.read_text(encoding="utf-8"))["took_over"] == DEAD


def test_the_takeover_replaces_the_old_lock_rather_than_adding_one():
    write_lease(Lease(gpu=1, pid=DEAD, port=8299))

    claim(pid=MINE, cards=[card(1)], port=8300)

    assert read_lease(1).pid == MINE
    assert len(list(eval_lease.lease_dir().iterdir())) == 1


# ── a free card is chosen, not configured ───────────────────────────────────
def test_a_card_with_a_foreign_process_is_not_taken():
    """"Card 1" was free on Tuesday and held xix-v2 on Wednesday."""
    busy = card(1, free=9_000, processes=[held_by(999, service="atr-party.service")])

    with pytest.raises(LeaseError, match="atr-party.service"):
        claim(pid=MINE, cards=[busy], gpu=1)


def test_a_card_with_too_little_memory_is_not_taken():
    with pytest.raises(LeaseError, match="9,?000 MiB free"):
        claim(pid=MINE, cards=[card(1, free=9_000)], gpu=1)


def test_the_emptiest_card_is_preferred():
    lease = claim(pid=MINE, cards=[card(0, free=21_000), card(1, free=45_000)],
                  port=8299)

    assert lease.gpu == 1


def test_our_own_processes_are_not_foreign(monkeypatch):
    """A merge or a loader the measurement itself started is a child, not a
    stranger — flagging those would make a card never claimable."""
    monkeypatch.setattr("atr_serving.eval_lease.gpu_probe.descends_from",
                        lambda pid, ancestor: pid == 4242 and ancestor == MINE)

    lease = claim(pid=MINE, cards=[card(1, processes=[held_by(4242)])], port=8299)

    assert lease.gpu == 1


def test_the_measurements_own_pid_is_not_foreign():
    lease = claim(pid=MINE, cards=[card(1, processes=[held_by(MINE)])], port=8299)

    assert lease.gpu == 1


def test_no_reading_means_no_claim():
    """Without nvidia-smi, "free" is a guess, and a guess is what put four vLLM
    instances on one card (#164)."""
    with pytest.raises(LeaseError, match="no cards"):
        claim(pid=MINE, cards=[])


def test_asking_for_a_card_that_does_not_exist_says_which_there_are():
    with pytest.raises(LeaseError, match="index 7"):
        claim(pid=MINE, cards=[card(0), card(1)], gpu=7)


def test_every_refusal_is_listed_not_just_the_first():
    """So the answer to "why can I not measure" is one message."""
    write_lease(Lease(gpu=0, pid=MINE, port=8299, model="theirs"))

    with pytest.raises(LeaseError) as raised:
        claim(pid=MINE, cards=[card(0), card(1, free=5_000)])

    message = str(raised.value)
    assert "gpu 0" in message and "gpu 1" in message


# ── the port is chosen too ──────────────────────────────────────────────────
def test_a_port_is_chosen_rather_than_wired_in():
    """8299 was hard-wired, which is what made two sessions negotiate."""
    lease = claim(pid=MINE, cards=[card(1)])

    assert lease.port >= eval_lease.PORT_BASE


def test_a_port_another_lease_reserved_is_skipped():
    """A lease is a server about to be up, which a bind test cannot see."""
    write_lease(Lease(gpu=0, pid=MINE, port=eval_lease.PORT_BASE))

    assert free_port() == eval_lease.PORT_BASE + 1


def test_a_port_something_is_listening_on_is_skipped():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        port = taken.getsockname()[1]
        taken.listen(1)

        assert free_port(base=port, limit=3) == port + 1


def test_no_free_port_is_an_error_and_not_a_collision():
    write_lease(Lease(gpu=0, pid=MINE, port=9000))
    write_lease(Lease(gpu=1, pid=MINE, port=9001))

    with pytest.raises(LeaseError, match="no free port"):
        free_port(base=9000, limit=2)


# ── releasing ───────────────────────────────────────────────────────────────
def test_releasing_removes_the_lock():
    lease = claim(pid=MINE, cards=[card(1)], port=8299)

    assert release(1, pid=MINE)
    assert not lease.path.exists()


def test_a_late_trap_does_not_release_a_live_holders_lease(monkeypatch):
    """The failure this guards: a `trap` firing after the card was taken over
    would clear the lock of the measurement that is now using it."""
    monkeypatch.setattr(eval_lease, "live", lambda pid: pid == 12345)
    write_lease(Lease(gpu=1, pid=12345, port=8299, model="theirs"))

    assert not release(1, pid=MINE)
    assert read_lease(1).pid == 12345


def test_releasing_a_card_nobody_holds_is_not_an_error():
    assert release(1, pid=MINE) is False


def test_a_stale_lease_can_be_tidied_up_by_anyone():
    """It belongs to nobody — the next claim would take it over, so a person
    clearing it should not have to delete a file by hand."""
    write_lease(Lease(gpu=1, pid=DEAD, port=8299, model="died"))

    assert release(1, pid=MINE)
    assert read_lease(1) is None


# ── never the production gateway ────────────────────────────────────────────
def test_the_production_gateway_is_refused():
    """The one mistake here that reached a user: three 502s and a 503 (#165)."""
    refusal = gateway_refusal(f"http://idhefix:{GATEWAY_PORT}")

    assert refusal is not None
    assert "#165" in refusal
    assert "--i-know" in refusal


def test_a_local_vllm_is_not_refused():
    assert gateway_refusal("http://127.0.0.1:8301") is None


def test_i_know_alone_is_not_enough():
    """A refusal that can be waved through without looking is not a refusal."""
    refusal = gateway_refusal(f"http://idhefix:{GATEWAY_PORT}", i_know=True)

    assert refusal is not None
    assert "journal" in refusal


def test_a_quiet_journal_lets_i_know_through():
    long_ago = NOW - JOURNAL_WINDOW_S - 60

    assert gateway_refusal(f"http://idhefix:{GATEWAY_PORT}", i_know=True,
                           journal=lambda: [long_ago], now=NOW) is None


def test_a_busy_journal_refuses_even_with_i_know():
    refusal = gateway_refusal(f"http://idhefix:{GATEWAY_PORT}", i_know=True,
                              journal=lambda: [NOW - 120], now=NOW)

    assert "Somebody is using it" in refusal
    assert "2 minutes ago" in refusal


def test_an_unreadable_journal_refuses():
    """Not knowing is not quiet. A missing unit or no permission must not read
    as "nobody is using it"."""
    refusal = gateway_refusal(f"http://idhefix:{GATEWAY_PORT}", i_know=True,
                              journal=lambda: None, now=NOW)

    assert refusal is not None
    assert "could not be read" in refusal


def test_the_window_is_half_an_hour():
    assert JOURNAL_WINDOW_S == 1800


# ── the shell interface ─────────────────────────────────────────────────────
def test_the_claim_prints_something_a_shell_can_eval(monkeypatch, capsys):
    """`eval "$(claim_gpu.py --pid $$)"` is the whole interface, so the line has
    to be three assignments and nothing else."""
    import scripts.claim_gpu as cli

    monkeypatch.setattr(cli.eval_lease, "claim",
                        lambda **kw: Lease(gpu=1, pid=MINE, port=8301))
    monkeypatch.setattr(sys, "argv", ["claim_gpu.py", "--pid", str(MINE)])

    assert cli.main() == 0

    line = capsys.readouterr().out.strip()
    assert line.startswith("GPU=1; PORT=8301; LEASE=")
    assert line.count(";") == 2


def test_a_box_without_nvidia_smi_refuses_instead_of_crashing(monkeypatch, capsys):
    """A traceback out of a `$(...)` substitution would be assigned to GPU."""
    import scripts.claim_gpu as cli

    monkeypatch.setattr(sys, "argv", ["claim_gpu.py", "--pid", str(MINE)])
    monkeypatch.setattr("atr_serving.eval_lease.gpu_probe.inspect",
                        lambda *a, **k: (_ for _ in ()).throw(
                            FileNotFoundError("nvidia-smi is not on PATH")))

    assert cli.main() == 1
    assert capsys.readouterr().out == ""


def test_the_takeover_note_does_not_go_to_stdout(monkeypatch, capsys):
    """stdout is eval'd by a shell; a comment line in it would be a syntax
    error in the caller, not a note."""
    import scripts.claim_gpu as cli

    monkeypatch.setattr(cli.eval_lease, "claim",
                        lambda **kw: Lease(gpu=1, pid=MINE, port=8301,
                                           took_over=DEAD))
    monkeypatch.setattr(sys, "argv", ["claim_gpu.py", "--pid", str(MINE)])
    cli.main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "GPU=1; PORT=8301; LEASE=" + str(
        eval_lease.lease_dir() / "gpu1.lock")
    assert str(DEAD) in captured.err


def test_the_journal_reader_returns_none_when_journalctl_is_missing(monkeypatch):
    """Which, through `gateway_refusal`, is a refusal — so a box without
    systemd does not silently become measurable."""
    import scripts.claim_gpu as cli

    monkeypatch.setattr(cli.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError))

    assert cli.journal_hits() is None
