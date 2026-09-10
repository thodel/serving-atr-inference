# Running this trainer on UBELIX

The same VLM training subsystem as on asterAIx, driven by **Slurm** instead of the
`atr-train` service. Nothing in `src/` or `engines/` changes — everything here is
environment and job plumbing. Full context and cost estimates:
[`docs/UBELIX_PLAN.md`](../docs/UBELIX_PLAN.md).

| file | what it is |
|---|---|
| `vlm-train.def` | Apptainer image: Ubuntu 24.04 + python3.12 + torch 2.8.0+cu128, mirroring `docs/asteraix-environment.md`. The repo is **not** baked in — it is bind-mounted, so code edits need no rebuild. |
| `submit_job.py` | The one thing the service did that the runner cannot: turn a JSON `TrainRequest` into a `JobStore` record. |
| `smoke.sbatch` | Phase 1: reproduce the Thun run on 1× RTX 4090, free QoS. |
| `report.py` | Print a finished job's status / metrics / error. |
| `specs/*.json` | Job requests, the same body `POST /train/jobs` takes. |

## Setup (once)

```bash
ssh ubelix
git clone https://github.com/thodel/serving-atr-inference.git ~/serving-atr-inference
mkdir -p ~/ubelix/logs && cp -r ~/serving-atr-inference/ubelix/* ~/ubelix/
export APPTAINER_TMPDIR=/scratch/network/users/$USER/apptainer/tmp
export APPTAINER_CACHEDIR=/scratch/network/users/$USER/apptainer/cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# Build from the REPO ROOT: the def's %files path is relative to the build CWD.
cd ~/serving-atr-inference
apptainer build ~/ubelix/vlm-train.sif ubelix/vlm-train.def   # ~10 min, 4.1 GB
```

Build on a **login node**: compute nodes may lack internet, login nodes have it.
The `.sif` belongs in `$HOME` — 1 TB private quota, snapshotted. Not the share
(group quota, 88 % full), not scratch (30-day purge).

## Standard procedure: two stages

**Always split the run.** `prepare` and `compile` are network, CPU and disk —
streaming pages, parsing PageXML, cutting crops, writing JSONL — and on a real
corpus that is hours. None of it touches a GPU. Doing it inside a GPU allocation
wastes the scarce half of the machine, and on UBELIX doubly so: the GPU job may
wait two days to start and would then spend its first three hours not using the
GPU it waited for.

```bash
# 1. plan the corpus — never hand-pick projects (VLM_TRAINING.md §2)
apptainer exec --bind /storage/research --bind /scratch --bind /rs_scratch \
  --env HF_HOME=$HF_HOME --env PYTHONPATH=$REPO/src:$REPO/engines \
  ~/ubelix/vlm-train.sif python scripts/plan_corpus.py \
    --org dh-unibe --period 1300 1600 --max-share 0.45 \
    --eval-repo <repo> --eval-project <p> --exclude-project <p> \
    --json ~/ubelix/specs/<name>.json --engine vllm --model-id <id>

# 2. add the measured training params to that spec (see the table below)

# 3. STAGE 1 — build the corpus on a free CPU node, no GPU
sbatch --export=ALL,SPEC=$HOME/ubelix/specs/<name>.json ubelix/prepare.sbatch

# 4. STAGE 2 — train it, resumable, on a GPU
sbatch --export=ALL,JOB_ID=<id from stage 1> ubelix/train.sbatch
```

Stage 1 leaves the job in `training` — **the same state a preemption leaves it
in** — so stage 2 takes the ordinary resume path and there is no second contract
to keep correct. Nothing is re-streamed, so the seeded split stays the one the
corpus was built with; re-preparing would rebuild it and quietly invalidate the
CER.

Compute nodes have internet (verified: `huggingface.co → 200` from `bnode009`),
so corpora need not be pre-cached on the share.

### The HF token (do this once)

Without one, `prepare` is rate-limited as an anonymous IP and dies mid-stream on
any corpus with many project directories:

```
429 … We had to rate limit your IP (130.92.232.126) … make sure you pass a HF_TOKEN
```

The `hf` CLI is inside the container, not on the login node. But **do not run
`hf auth login`**: it writes to `$HF_HOME/token`, and `HF_HOME` here points at the
**group-readable research share**, so the token would be exposed to everyone in
`wbkolleg_dh_1`. Put it in a private file in `$HOME` instead — the jobs read it
from there:

```bash
umask 077
read -rs -p 'HF token: ' T && printf '%s' "$T" > ~/.hf_token && unset T && echo
chmod 600 ~/.hf_token
```

Every sbatch here picks it up automatically and reports `HF token: present` or
`ABSENT` in its log.

*Partial mitigation without a token:* a selection that covers **every** project in
a repo collapses to one glob (`collapse_complete_selection`, #89), so
`all_projects: true` avoids the per-project calls. It only helps when taking the
whole repo is what you want — `plan_corpus` deduplicates, which makes selections
incomplete and re-exposes the problem.

### The measured configuration

Every value comes from an experiment in `docs/UBELIX_PLAN.md` §9, not from taste:

| param | value | why |
|---|---|---|
| `base_model` | `Qwen/Qwen3-VL-4B-Instruct` | 4B matches 8B at half the size (§9.3-B) |
| `load_in_4bit` | `false` | bf16 is +23 % on a card we own (§9.3-A) |
| `batch_size` | `16` | **the** bottleneck: 51 % → 86 % GPU utilization (§9.3-G) |
| `max_pixels` | `262144` | 256 visual tokens; 128 ties, 512 is worse (§9.3-D) |
| `workers` | `8` | 4/8/16 indistinguishable (§9.3-F) |
| `save_steps` | `200` | bounds the work a preemption costs |
| `epochs` | `1` | the runbook's rule at corpus scale |

**Do not stage crops to `/scratch/local`.** It is measurably neutral (−0.5 %) and
costs 10–31 h of copying at full scale (§9.2).

## Run

Use `submit.sh`, not `sbatch` — it validates the spec on the login node first.
`submit_job.py` does validate, but it runs *inside* the batch job, so a bad spec
costs a queue wait and a GPU allocation before anything says so (job 14431367
died 13 s in over a capital letter in a `model_id`).

```bash
./submit.sh ~/ubelix/experiment_a.sbatch
sbatch ~/ubelix/smoke.sbatch
squeue -u $USER
tail -f ~/ubelix/logs/vlm-smoke-<jobid>.out
```

## Checking progress from the laptop

```bash
./ubelix/status.sh            # queue, recent jobs, quota, newest log, metrics
./ubelix/status.sh -f         # follow the newest job's log
./ubelix/status.sh -j 14108981 -n 60
```

Read-only, and it goes through the `ubelix` ssh alias (ProxyJump via asterAIx), so it
needs no VPN.

If asterAIx is down, that route dies with it. On the UniBE VPN, `submit02` is reachable
directly — use the `ubelix-direct` alias, or `UBELIX_HOST=ubelix-direct ./ubelix/status.sh`.

## Paying for GPUs — short version: don't, for H100s

Checked 2026-08-27 against the internal price page and `sacctmgr show qos`:

* **H100 CHF 0.60/h, RTX 4090 CHF 0.10/h**, per-minute billing.
* **Preemptable and debug jobs are free**, project or no project.
* The free `job_gpu_preemptable` QoS grants **h100=4**. The paid `job_gpu` QoS
  grants **h100=1**. Paying gets you *fewer* H100s, not more — more than 4
  needs an investment, and GPU investments are closed until the 2026 DC expansion.

So the free 4× H100 preemptable path is both cheaper and ~3× faster than anything
purchasable. A project is still worth having for the **F2 free tier** (up to
CHF 1000/year refunded per cost centre — the whole campaign is ~CHF 510) and so that
a billed RTX 4090 fallback is a header edit rather than a procurement. Ordered by the
institute's IT-responsible person at `iamportal.unibe.ch` → "HPC - Order new Project
Space". See [`docs/UBELIX_PLAN.md`](../docs/UBELIX_PLAN.md) §4.3.

```bash
#SBATCH --account=gratis              # the primary path: free, 4x H100, killable
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:4
```

## Long runs: preemption and resume

`train_resumable.sbatch` is the template for anything that will not finish inside
one allocation. Three things have to line up, and all three are in the script:

* `--requeue`, so Slurm puts the job back rather than ending it;
* a **stable training job id** across attempts, written once and read back from
  `$SCRATCH/runs/slurm-$SLURM_JOB_ID.jobid` — without it each attempt creates a
  new job, compiles a new corpus with a **new seeded split**, and starts at zero;
* `ATR_TRAIN_PREEMPTABLE=1`, so the runner treats SIGTERM as "resume later"
  rather than "cancelled" and leaves the record in `training`.

Set `save_steps` in the spec. The default (0) checkpoints once per epoch, which
is right when an epoch is minutes and useless when it is days.

**Two interruption modes, and they behave differently.** On preemption or
`scontrol requeue`, Slurm kills the step promptly — there is no time to shut
down, and what keeps the job resumable is that nothing writes a terminal status
(`JobStore.save` is tmp-then-`os.replace`, so a hard kill cannot corrupt the
record). At **walltime**, `--signal=B:TERM@120` gives 120 seconds and the
graceful path runs. Both end in the same place; only the second logs about it.

Verify a resume by three lines: `resuming from …/checkpoint-N` (trainer log),
`re-entered while \`training\`` (runner log), and the same job id on both
attempts (batch log).

## Four things that differ from asterAIx

All of them are set in `smoke.sbatch`; copy that header for any new job.

1. `--mem` is rejected without `--nodes`.
2. `/scratch/network` symlinks to `/rs_scratch`; Apptainer must bind **both**.
3. `ATR_TRAIN_VENVS_ROOT=/opt` — the venv is `/opt/vlm-train` in the container.
4. `ATR_TRAIN_GPU=0` — Slurm's allocated GPU is index 0, not asterAIx's 1.

## Where things go

Read the dataset and base models from the share; write everything else to scratch.
See [`docs/UBELIX_PLAN.md`](../docs/UBELIX_PLAN.md) §5.1 for the full policy — the
short version is that scratch holds only what we can rebuild, and the split manifest
is what makes rebuilding possible.
