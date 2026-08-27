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

## Run

```bash
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
