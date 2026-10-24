#!/usr/bin/env python3
"""A measurement's directory, its server and its card — set up, and taken down in a trap (#183).

    eval "$(scripts/eval_workspace.py create --model M --pid $$)"   # WS=…; ATR_VLLM_MERGED_DIR=…
    scripts/eval_workspace.py server "$WS" --pid "$!"              # after `setsid vllm serve … &`
    scripts/eval_workspace.py finish "$WS" --status "$?" --gpu "$GPU" [--keep]
    scripts/eval_workspace.py gc                                   # what lies around, how big
    scripts/eval_workspace.py gc --apply [--kept]                  # remove the orphans [and kept]

``scripts/eval_cleanup.sh`` wraps the first three into ``eval_begin`` /
``eval_serve`` and arms the traps; use that from a measurement script.

``finish`` stops the server by pid and process group (never by pattern), waits
until the card holds none of its processes, releases the card's lease (#184)
and deletes the directory — unless ``--keep`` was given or a step failed
(any status but 0 and the interrupts 129/130/143), in which case it says where
the directory is, how big, and how to remove it.

``gc`` never kills anything and never deletes a directory without a marker, a
directory whose owner is alive, or one whose server is still running.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr_serving import eval_workspace as ew  # noqa: E402


def cmd_create(args) -> int:
    ws = ew.create(args.model, args.pid, root=args.root)
    print(ws.shell())
    return 0


def _load(path: str) -> ew.Workspace | None:
    ws = ew.load(Path(path))
    if ws is None:
        print(f"ERROR: {path} is not a measurement directory (no readable "
              f"{ew.MARKER}); nothing done", file=sys.stderr)
    return ws


def cmd_server(args) -> int:
    ws = _load(args.workspace)
    if ws is None:
        return 1
    ew.record_server(ws, args.pid)
    if ws.server_pgid is None:
        print(f"# server pid {args.pid} does not lead its own process group; start it "
              "with setsid so its children are stopped with it", file=sys.stderr)
    return 0


def cmd_finish(args) -> int:
    ws = _load(args.workspace)
    if ws is None:
        return 1
    if args.gpu is not None:
        ws.gpu = args.gpu
    grace = float(os.environ.get("ATR_EVAL_TERM_GRACE_S", ew.TERM_GRACE_S))
    card_wait = float(os.environ.get("ATR_EVAL_CARD_WAIT_S", ew.CARD_WAIT_S))
    done = ew.finish(ws, args.status, keep=args.keep, gpu=ws.gpu, root=args.root,
                     stop=lambda w: ew.stop_server(w, grace_s=grace),
                     wait=lambda g, pids: ew.wait_card_free(g, pids, timeout_s=card_wait))
    for line in done.lines:
        print(f"[eval cleanup] {line}", file=sys.stderr)
    return 0 if done.card != "busy" else 3


def cmd_gc(args) -> int:
    entries = ew.scan(args.root)
    if not entries:
        print(f"nothing under {args.root or ew.workspace_root()}")
        return 0
    total = 0
    for entry in entries:
        print(entry.line())
        total += entry.size or 0
    unknown = sum(1 for e in entries if e.size is None)
    print(f"\n{len(entries)} director{'y' if len(entries) == 1 else 'ies'}, "
          f"{ew.human(total)}" + (f" (+ {unknown} of unknown size)" if unknown else ""))
    removable = [e for e in entries
                 if (e.verdict == "orphaned" or (args.kept and e.verdict == "kept"))
                 and not e.server_alive]
    if not args.apply:
        if removable:
            print(f"{len(removable)} would be removed with --apply"
                  + ("" if args.kept else " (kept ones only with --kept)"))
        return 0
    freed = 0
    for entry in removable:
        try:
            ew.remove(entry.ws, args.root)
        except (OSError, ValueError) as exc:
            print(f"not removed: {entry.path}: {exc}", file=sys.stderr)
            continue
        freed += entry.size or 0
        print(f"removed {entry.path} ({ew.human(entry.size)})")
    print(f"freed {ew.human(freed)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]))
    parser.add_argument("--root", type=Path, default=None,
                        help="where measurement directories live (default ~/atr-eval, "
                             "or $ATR_EVAL_ROOT)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="make a measurement directory; prints shell lines")
    p.add_argument("--model")
    p.add_argument("--pid", type=int, default=os.getppid(),
                   help="the shell the directory belongs to ($$)")
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("server", help="record the server's pid")
    p.add_argument("workspace")
    p.add_argument("--pid", type=int, required=True)
    p.set_defaults(func=cmd_server)

    p = sub.add_parser("finish", help="stop, wait for the card, release, delete or keep")
    p.add_argument("workspace")
    p.add_argument("--status", type=int, required=True, help="the shell's exit status")
    p.add_argument("--gpu", type=int)
    p.add_argument("--keep", action="store_true")
    p.set_defaults(func=cmd_finish)

    p = sub.add_parser("gc", help="list measurement directories and what they hold")
    p.add_argument("--apply", action="store_true",
                   help="remove orphaned directories whose server is not running")
    p.add_argument("--kept", action="store_true",
                   help="with --apply: also the ones kept after a failure or --keep")
    p.set_defaults(func=cmd_gc)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
