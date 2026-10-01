#!/usr/bin/env python3
"""Claim a card and a port for a measurement, so the other session can see it.

Two sessions measuring on asteraix at the same time divided the cards and the
ports in a chat window (#184). This is that division in a file. A measurement
claims before it starts and releases in a ``trap``, so an interrupted run does
not leave a card that looks taken for ever:

    eval=$(scripts/claim_gpu.py --model qwen3vl-medieval-german-v3 --pid $$) || exit 1
    eval "$eval"                      # GPU=1 PORT=8301 LEASE=~/.atr-eval/gpu1.lock
    trap 'scripts/claim_gpu.py --release "$GPU" --pid $$' EXIT
    CUDA_VISIBLE_DEVICES=$GPU vllm serve … --host 127.0.0.1 --port "$PORT"

``--pid $$`` is the point: the lock belongs to the shell that holds the card,
not to this script, which exits immediately. A lock whose pid is gone is taken
over by the next claim rather than respected.

    scripts/claim_gpu.py --show                 # who holds what
    scripts/claim_gpu.py --release 1 --pid $$    # drop one, if it is still ours
    scripts/claim_gpu.py --check-url http://idhefix:8200   # refuse the gateway

Nothing here starts, stops or kills anything. The lease is advisory, which is
all a lock file can be; its value is that the other session can read it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_serving import eval_lease  # noqa: E402
from atr_serving.eval_lease import MAX_CARDS, LeaseError  # noqa: E402

#: Read for ``--check-url --i-know``. A journal that cannot be read is a
#: refusal, so a missing unit or no permission does not read as "quiet".
GATEWAY_UNIT = os.environ.get("ATR_GATEWAY_UNIT", "atr-gateway")


def journal_hits(unit: str = GATEWAY_UNIT, window_s: int | None = None):
    """Epoch seconds of recent gateway log lines, or None if unreadable.

    Deliberately crude: any line the gateway wrote counts as use. A measurement
    is about to put load on a card somebody may be serving from, and "it logged
    something in the last half hour" is the conservative reading of that.
    """
    window_s = window_s or eval_lease.JOURNAL_WINDOW_S
    try:
        done = subprocess.run(
            ["journalctl", "-u", unit, "--since", f"-{window_s // 60}min",
             "--output=short-unix", "--no-pager"],
            capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    stamps = []
    for line in done.stdout.splitlines():
        head = line.split(" ", 1)[0]
        try:
            stamps.append(float(head))
        except ValueError:
            continue
    return stamps


def show() -> int:
    held = [lease for lease in
            (eval_lease.read_lease(index) for index in range(MAX_CARDS))
            if lease is not None]
    if not held:
        print("no card is claimed")
        return 0
    for lease in held:
        state = "live" if eval_lease.live(lease.pid) else "STALE (will be taken over)"
        print(f"{lease.describe()} — {state}")
    return 0


def check_url(url: str, i_know: bool) -> int:
    refusal = eval_lease.gateway_refusal(url, i_know=i_know, journal=journal_hits,
                                         now=time.time())
    if refusal:
        print(f"ERROR: {refusal}", file=sys.stderr)
        return 1
    print(f"ok: {url}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]))
    parser.add_argument("--pid", type=int, default=os.getppid(),
                        help="the process the lease belongs to (use $$ from a "
                             "shell; default: this script's parent)")
    parser.add_argument("--model", help="recorded in the lock, for the next reader")
    parser.add_argument("--gpu", type=int,
                        help="ask for one card in particular; still refused if "
                             "it is held or busy")
    parser.add_argument("--port", type=int, help="skip the port search")
    parser.add_argument("--min-free-mib", type=int, default=eval_lease.MIN_FREE_MIB)
    parser.add_argument("--show", action="store_true", help="print who holds what")
    parser.add_argument("--release", type=int, metavar="GPU",
                        help="drop the lease on this card, if --pid still holds it")
    parser.add_argument("--check-url", metavar="URL",
                        help="refuse the production gateway before measuring it")
    parser.add_argument("--i-know", action="store_true",
                        help="with --check-url: allow the gateway, but only if "
                             "its journal has been quiet")
    args = parser.parse_args()

    if args.show:
        return show()
    if args.check_url:
        return check_url(args.check_url, args.i_know)
    if args.release is not None:
        dropped = eval_lease.release(args.release, pid=args.pid)
        print(f"gpu {args.release}: " + ("released" if dropped else
              "not ours or not held — left alone"))
        return 0

    try:
        lease = eval_lease.claim(pid=args.pid, model=args.model, gpu=args.gpu,
                                 port=args.port, min_free_mib=args.min_free_mib)
    except LeaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if lease.took_over:
        print(f"# took over a stale lock from pid {lease.took_over}", file=sys.stderr)
    # Shell-eval'able, so the caller needs no parsing and no temporary file.
    print(f"GPU={lease.gpu}; PORT={lease.port}; LEASE={lease.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
