# A measurement on a spare card: claim, set up, take down

Part of #179 ("one measurement is one command"). Two pieces exist so far: claiming a
card visibly (#184, `scripts/claim_gpu.py`) and taking a measurement down on every
exit (#183, `scripts/eval_cleanup.sh` + `scripts/eval_workspace.py`). The venv (#180)
and the test pages (#181) are still built by hand, but they go into the directory that
is taken down.

## Why

One measurement occupies **19–27 GB** (venv 5 GB, merged model 8–17 GB, test pages)
and a vLLM on a whole card. Until now it was removed by hand. The one time it was not
removed, it lay there unnoticed, and that is how idhefix came to 260 GB (#143). A vLLM
that keeps running after an abort holds its card.

## The skeleton

```bash
#!/usr/bin/env bash
set -euo pipefail
source scripts/eval_cleanup.sh

eval_begin qwen3vl-medieval-german-v3          # or: eval_begin <id> --keep
# GPU, PORT: a card nothing foreign runs on, claimed in ~/.atr-eval/gpu<N>.lock
# WS:        ~/atr-eval/<stamp>-<id>-<pid>/ with measurement.json
# ATR_VLLM_MERGED_DIR=$WS/merged, exported, so merge_loras writes into WS

uv venv "$WS/venv" …                            # #180, by hand for now
.venvs/vlm-train/bin/python scripts/merge_loras.py --only qwen3vl-medieval-german-v3
eval_serve "$WS/venv/bin/vllm" serve "$ATR_VLLM_MERGED_DIR/qwen3vl-medieval-german-v3" \
    --host 127.0.0.1 --port "$PORT"
…wait for /health, run eval_granularity.py (training-atr-models)…
# the report goes OUTSIDE "$WS": the directory is deleted
```

On exit, whether it succeeded, a step failed, Ctrl-C was pressed, or a `kill` came, the
trap does four things in order:

1. **It stops the server by pid and process group, never by pattern.** `pkill -f`
   killed the measurement's own SSH command twice, because that command's line
   contained the pattern. `eval_serve` starts the server with `setsid`, so it leads its
   own group. SIGTERM goes to the group, then SIGKILL after 60 s, then to every
   descendant seen before the first signal. A vLLM EngineCore that outlives
   `vllm serve` holds the card just the same.
2. **It waits for the card** (up to 120 s) until none of those processes is on it. If
   the card cannot be read, the result is printed as `not checked`, never as free. A
   card that is still busy prints a WARNING, and the trap kills nothing by name.
3. **It releases the lease** if it is still ours.
4. **It deletes `$WS`**, with two exceptions:

   | exit | directory |
   |---|---|
   | 0 | deleted |
   | 129 / 130 / 143 (HUP, Ctrl-C, TERM) | deleted. Nobody debugs an interrupted run, and 27 GB nobody asked to keep is what #143 was made of |
   | any other (a failed step) | **kept**, with its size, the reason in `measurement.json`, `server.log`, and the `rm -rf` line that removes it |
   | `--keep` | kept |

   The server is stopped in every case. Only the directory is ever kept.

A second Ctrl-C while the trap is running is ignored. A cleanup cut short is exactly
the abort the trap exists for, and its waits are bounded.

**One limit of bash:** a SIGTERM sent to the script's shell alone, not to its whole
group, is handled only once the current foreground command returns. Ctrl-C reaches the
whole group and is not affected. Run long steps as `step & wait $!` if the script
should react to `kill <pid>` straight away.

## What lies around: `gc`

```
scripts/eval_workspace.py gc                    # every directory, its size, whose it is
scripts/eval_workspace.py gc --apply            # remove the orphans
scripts/eval_workspace.py gc --apply --kept     # … and the kept ones
```

| verdict | meaning | removed by `--apply` |
|---|---|---|
| `in use` | its shell is alive | never |
| `orphaned` | its shell died without the trap (SIGKILL, OOM kill, reboot) | yes, unless its server is still running; then it prints the `kill -TERM -<pgid>` for a person to run |
| `kept` | a failed step or `--keep` | only with `--kept` |
| `elsewhere` | recorded on another host; its pids mean nothing here | never |
| `unattributed` | no `measurement.json` | never, because nobody can say whose it is |

`gc` never kills anything. `remove` deletes only a directory directly under the root
that has a marker. That check is what stands between an empty `$WS` and `rm -rf`.

Environment: `ATR_EVAL_ROOT` (default `~/atr-eval`), `ATR_EVAL_LEASE_DIR` (default
`~/.atr-eval`), `ATR_EVAL_PYTHON` (default `.venvs/gateway/bin/python`, else `python3`),
and `ATR_EVAL_TERM_GRACE_S` / `ATR_EVAL_CARD_WAIT_S` for the two waits.
