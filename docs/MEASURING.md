# A measurement on a spare card: claim, set up, take down

Part of #179 ("one measurement is one command"). Four pieces exist:

- claiming a card visibly (#184, `scripts/claim_gpu.py`);
- taking a measurement down on every exit (#183, `scripts/eval_cleanup.sh` +
  `scripts/eval_workspace.py`);
- the venv with Python headers, in one call (#180, `scripts/eval_env.sh`);
- held-out test pages with the proof they were not trained on (#181,
  `scripts/eval_pages.py`).

All of them write into the directory that is taken down.

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

bash scripts/eval_env.sh "$WS/env"              # #180; --vllm next for a vllm-next model
python3 scripts/eval_pages.py qwen3vl-medieval-german-v3 --n 15 --out "$WS/pages"   # #181
.venvs/vlm-train/bin/python scripts/merge_loras.py --only qwen3vl-medieval-german-v3
eval_serve "$WS/env/serve.sh" "$ATR_VLLM_MERGED_DIR/qwen3vl-medieval-german-v3"
…wait for /health, run eval_granularity.py (training-atr-models)…
# the report goes OUTSIDE "$WS": the directory is deleted
```

## The venv: `scripts/eval_env.sh`

```
bash scripts/eval_env.sh <dir> [--vllm 0.11|next]
```

- **Python headers without root.** asteraix has no `python3.12-dev`, and Triton
  compiles a helper against `Python.h` when vLLM starts. Without the headers vLLM
  dies with a gcc error after the weights are loaded. The script installs uv's
  standalone CPython 3.12 into `<dir>/uvpy` (it carries `include/python3.12`) with
  `--no-bin`, and removes any python shim uv leaves in `~/.local/bin` all the same.
- **The same pins as the gateway.** `--vllm 0.11` builds what `make_venvs.sh vllm`
  builds, and `--vllm next` what `make_venvs.sh vllm-next` builds. The variant follows
  the model's `vllm_venv` in `config/models.yaml`, not the host. A test keeps the
  torch pins of both scripts equal.
- **`serve.sh`.** It sets `CPATH`, `CUDA_VISIBLE_DEVICES=$GPU` and
  `--host 127.0.0.1 --port $PORT`, and refuses to start without `GPU` and `PORT`, which
  `eval_begin` sets.
- **Idempotent.** `eval_env.json` is written last, and only after `vllm --version`
  runs. A second call on a built directory checks and builds nothing. A directory built
  for the other variant or from other pins, or holding anything the script did not put
  there, is refused, never mixed.

## On exit

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
