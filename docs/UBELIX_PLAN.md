# Running a large VLM training on UBELIX

A plan and a cost estimate for fine-tuning Qwen3-VL on the **full** medieval
dataset (`dh-unibe/image-text_medieval-scripts_xiv-xv-xvi`), on the University
of Bern cluster rather than on asterAIx.

Everything marked **[measured]** is a number we have. Everything marked
**[estimate]** is arithmetic from those numbers plus published hardware specs,
and Phase 3 below exists to replace the estimates with measurements before any
money is spent.

---

## 0. Status — verified on the cluster, 2026-08-27

Logged in and probed. Three of this document's assumptions turned out to be wrong,
all in our favour except the last:

| what | assumed | **actual** |
|---|---|---|
| getting the data there | ticket + possibly a 1 TB transfer | **already there** — `/storage/research/wbkolleg_dh_1` is mounted, and the full dataset is cached |
| free-tier GPU ceiling | 1× H100 | **4× H100** preemptable (24 h), or 1× H100 / 2× RTX 4090 on the 96 h `job_gratis` |
| share headroom | ≥5 TB workspace | **88 % full — 1.4 TB and 3.2 M inodes free** |

### Login

Works with a key, from this laptop, **no VPN and no password**. asterAIx sits inside
the UniBE network, so it relays the connection and the private key never leaves the
laptop:

```
Host ubelix
    HostName submit02.unibe.ch
    User th19c587
    ProxyJump srv-train
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

Host keys for all four submit nodes were scanned and matched against the published
fingerprint table (8/8 OK) before being written to `known_hosts` on asterAIx.

### What is already on the cluster

`/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub` — 1.7 TB of HF cache,
the same one `lassberg` built, readable from the login node:

* **`image-text_medieval-scripts_xiv-xv-xvi` — 1.3 TB, 691 parquet files, 0 `.incomplete`**
* `Qwen/Qwen3-VL-8B-Instruct` — 17 GB
* `Qwen/Qwen3-VL-30B-A3B-Instruct`
* ~20 other dh-unibe ground-truth sets

So **phase 0 of §6 is already done** and phase 2 can start whenever the code is ready.
(The export lists 694 parquet files against 691 here — worth a count check before a
production run, but no partial downloads are pending.)

### Phase 1 is done — the smoke run reproduced

`vlm-train.sif` (4.1 GB, Ubuntu 24.04 + python3.12 + torch 2.8.0+cu128) is built and
lives in `$HOME/ubelix/`. Job **14108981**, 1× RTX 4090 on the free `job_gratis` QoS,
**9 min 34 s wall, zero cost**:

| | asterAIx (A40) | **UBELIX (RTX 4090)** |
|---|---:|---:|
| selection | 52 pages → 783 crops (594/189) | **identical — 594 / 189** |
| **CER** | **0.466** | **0.4662** |
| WER | 0.816 | 0.8195 |

Same split, same numbers. The container is correct, and the seeded split is
reproducible across machines — which is what makes §5.1's manifest-as-anchor
argument work in practice.

Stage timings (a tiny run, so read them as plumbing, not throughput):
prepare 13 s · compile 4 s · train+eval 9 min. Crucially the prepare log says
`hub cache …: present` — **streaming read the parquet from the share, not the
network.** The `cache_datasets: false` default is safe here.

### Four portability fixes the smoke run found

Each was a silent assumption about asterAIx baked into the code or the job:

1. **`--mem` requires `--nodes`** on UBELIX's Slurm. Submission is rejected outright.
2. **`/scratch/network` is a symlink to `/rs_scratch`**, which Apptainer does not
   resolve. Bind *both* or every scratch path inside the container dangles.
3. **`ATR_TRAIN_VENVS_ROOT=/opt`** — the runner spawns the trainer with
   `<venvs_root>/vlm-train/bin/python`, which is `$REPO/.venvs` on asterAIx and
   `/opt` in the container.
4. **`ATR_TRAIN_GPU=0`** — the asterAIx default of `1` exists to dodge the shared RAG
   card. Under Slurm, the allocated GPU is always index 0 inside the job.

All four are environment, not code: nothing in the repo had to change.

### Slurm reality (`sqos`, this user)

| QoS | walltime | per-user GPU ceiling |
|---|---|---|
| `job_gratis` | 96 h | h100=1, rtx4090=2, **gpu=3** |
| **`job_gpu_preemptable`** | **24 h** | **h100=4**, rtx3090=18, rtx4090=4, **gpu=29** |
| `job_debug` | 20 min | h100=1, rtx4090=1 |

`sacctmgr` shows only the **`gratis`** and **`teaching`** accounts, and `swckeys`
returns `noop` — **there is no PAYGO project**, so nothing can be billed yet even if
we wanted to. That makes the preemptable path the *primary* plan, not the fallback.

Cluster at the time of probing: 5× 8 H100 nodes all `mix` (partially free), one 8×
H200 node fully `idle` (but our QoS has `h200=0`), plus an undocumented
`rtx_pro_6000_blackwell` node. Free capacity exists.

---

## 1. What "the full medieval data set" actually is

| | |
|---|---|
| page samples | **548,322** (dataset card) / 497.4 K rows in the Parquet export |
| projects | 151+ — Itinera Nova, the SAL series, Thuner Missiven, Königsfelden charters |
| period / scope | 1350–1550, mostly State Archives Leuven, Flemish + German |
| schema | `image` (full page scan), `xml_content` (PageXML), `filename`, `project_name` |
| Parquet export | **1,064.6 GB** across 694 files |
| card's stated total | ~6.6 TB |
| **line crops** | **~8 M** [estimate] |

The line count is the number that drives everything, and it is not published.
Our own measurement is the only anchor: the Thun run turned **52 pages into 783
line crops** [measured], i.e. ~15 lines/page. Protocol registers run denser, so
the honest range is **7–12 M line crops**, and this document uses **8 M**.

For scale: `thun-kurrent-v2`, our best kraken model, was trained on **1,898
lines**. The full medieval set is roughly **4,000× that**.

## 2. What UBELIX gives us

**GPU hardware** (~20 nodes, all 8 GPUs per node except the A100 box):

| GPUs/node | Type | VRAM | CPUs/GPU | RAM/GPU | Nodes | `--gres` |
|---:|---|---:|---:|---:|---:|---|
| 8 | RTX 3090 | 24 GB | 4 | 60 GB | 6 | `gpu:rtx3090:N` |
| 8 | RTX 4090 | 24 GB | 16 | 90 GB | 8 | `gpu:rtx4090:N` |
| 6 | A100 | 80 GB | 20 | 80 GB | 1 | `gpu:a100:N` |
| 8 | H100 | 96 GB | 16 | 90 GB | 5 | `gpu:h100:N` |
| 8 | H200 | 141 GB | 16 | 90 GB | 2 | `gpu:h200:N` |

Every GPU node has **1.92 TB of local NVMe** (`/scratch/local`) and 100 Gb/s
Infiniband. The CPU/memory-per-GPU column is a hard limit — asking for more CPUs
than that per GPU gets the job **rejected**, not queued.

**Accounts, partitions, walltime** — this is the part that shapes the job design:

| account | partition | QoS | walltime | what we get |
|---|---|---|---:|---|
| `gratis` | `gpu` | `job_gratis` | **96 h** | max **2× RTX 4090** or **1× H100** |
| `gratis` | `gpu-invest` | `job_gpu_preemptable` | 24 h | idle investor GPUs, **killed without warning** |
| `paygo` | `gpu` | `job_gpu` | **24 h** | anything, billed, needs a wckey |
| `paygo` | `epyc2` | `job_cpu_long` | 16 days | CPU only — matters for data prep |
| `invest` | `gpu-invest` | investor QoS | — | **closed**: no new GPU investments until the 2026 DC expansion |

The published free-tier page understates what we actually have — see §0 for the
`sqos` output. Two consequences, both load-bearing:

1. **Free means preemptable, and preemptable means 4× H100 for 24 h.** That is the
   working configuration, because we have no PAYGO project (`swckeys` → `noop`).
   A PAYGO project would buy *non*-preemptable time and is created in the IAM portal
   by an institute technology manager; worth starting, not worth waiting for.
2. **Every GPU job we can run dies at 24 h**, whether by the wall or by preemption.
   Checkpoint-and-requeue is the entire job design, not a nicety.

**Storage**:

| area | path | quota | files | notes |
|---|---|---:|---:|---|
| home | `/storage/homefs/$USER` | 1 TB | **1 M** | snapshots; never put the dataset here |
| workspace | `/storage/research/wbkolleg_dh_1` | 12 TB, **88 % used** | 15.7 M, **79 % used** | our share, already mounted |
| capacity | `/storage/capacity` | ≥50 TB | 100 K/TB | no snapshots, **submit nodes only** |
| net scratch | `/scratch/network` | **15 TB, 0 used** | 10 M | **purged after 30 days unaccessed** — the right home for crops |
| local scratch | `/scratch/local` | 1.92 TB | — | node-local, job lifetime only |

## 3. The constraint that actually bites

### 3.1 The data is already there — resolved

`smb://resstore.unibe.ch/wbkolleg_dh_1` and `/storage/research/wbkolleg_dh_1` are the
same share on the University's Research Storage service, and it is **already mounted
on UBELIX** with our group (`rs_wbkolleg_dh_1`) on it. No ticket, no transfer, no
second copy. The 1.3 TB medieval dataset and both Qwen3-VL checkpoints are sitting in
`Textrecognition_Training/hf_hub` (§0).

The only setup left is pointing the environment at it:

```bash
export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
```

### 3.2 Eight million crops cannot be eight million files — and now it is worse

The share is at **88 % of its 12 TB and 79 % of its 15.7 M inodes**: about **1.4 TB
and 3.2 M files free**, shared with everyone else in the DH group.

8 M loose line-crop JPEGs would therefore **not fit at all** — they exceed the free
inode budget by 2.5×, before anyone else writes a byte. This is no longer a
performance argument, it is a hard stop.

So the crop stage must write **sharded** output — WebDataset `.tar` shards or
Parquet, ~2 GB each, ~160 files total. This is a change to `vlm_dataset.py`,
which currently writes `crops/*.jpg` + `train.jsonl`. It is the single most
important code change in this plan, and it also happens to be what makes the
data stage to `/scratch/local` cheaply (see §5).

**And the shards belong on scratch, not the share.** Our personal `SCR_usr` quota is
**15 TB with 0 used**, on a filesystem with 129 TB free. Read the dataset from the
share, write ~320 GB of shards to `/scratch/network/users/$USER`, and leave the share
alone. The 30-day purge is a real cost — anything untouched for a month is deleted —
so the shards get regenerated per campaign rather than archived. At a few CPU-node-hours
to rebuild, that is the cheaper side of the trade.

## 4. Compute estimate

**The anchor [measured]**, from `20260808T080206Z-qwen3vl-thun-smoke` on asterAIx:
Qwen3-VL-8B, QLoRA NF4, line granularity, effective batch 16, **1.9 samples/s on
one A40**. That run also logged `bitsandbytes: inner dimension (4304) is not
aligned ... falling back to slower implementation` — Qwen3-VL's dimensions miss
the fast 4-bit kernel, which is why 1.9 and not more.

Scaling that to UBELIX cards [estimate]:

| GPU | vs A40 | samples/s | why |
|---|---:|---:|---|
| RTX 4090 | 1.5–2× | 3–4 | 2× the bf16 throughput, but 24 GB forces 4-bit and small batches |
| A100 80 GB | ~3× | ~6 | fits bf16 LoRA, 2 TB/s |
| **H100 96 GB** | **6–8×** | **~12** | 5.5× compute **and** 96 GB lets us drop 4-bit entirely — no slow dequant path |
| H200 141 GB | 7–9× | ~14 | same compute, 4.8 TB/s, bigger batches |

Dropping NF4 on the H100 is worth calling out separately: 4-bit exists on asterAIx
because the card is *shared with the serving engines*. On a dedicated 96 GB H100
an 8B model trains in bf16 with room for a real batch, which removes both the
misaligned-kernel penalty and the quantization noise.

**Per epoch over 8 M line crops:**

| configuration | available to us? | GPU-hours/epoch | wall time/epoch |
|---|---|---:|---|
| 1× H100, `job_gratis`, 96 h | **yes** | ~200–350 | 8–15 days |
| 2× RTX 4090, `job_gratis`, 96 h | **yes** | ~700 | 15 days |
| **4× H100, `job_gpu_preemptable`, 24 h** | **yes** | ~200–350 | **~2.5–4 days**, in 24 h chunks |
| 8× H100, one whole node | needs PAYGO | ~200–350 | ~37 h (85 % DDP scaling) |
| 8× H200 | no — QoS has `h200=0` | ~180–300 | ~30 h |

**Three epochs on the full set ≈ 600–1,000 H100-GPU-hours.** On the 4× H100
preemptable QoS that is **8–12 days of wall clock** spread over ten-odd requeued
24 h chunks — and it costs nothing. The full run is therefore *affordable*; what it
costs is calendar time and a restart path that genuinely works.

### 4.1 The recommendation: don't train on all 8 M lines first

The cost above is linear in lines and the benefit is not. A **stratified
subsample — every one of the 151 projects represented, capped at N lines per
project, ~500 K–1 M lines total** — costs about a tenth as much:

| | full 8 M | subsample 800 K |
|---|---:|---:|
| GPU-h/epoch (H100) | 200–350 | 20–35 |
| 3 epochs on 4× H100 preemptable | 8–12 days, ~10 requeues | **~24 h — one chunk, possibly zero requeues** |
| script/language coverage | complete | complete (that is the point of stratifying) |

The whole campaign then becomes: run the subsample, look at where the CER curve
is when it ends, and buy the full run **only if the curve is still falling**. Our
own controlled chain in `TRAINING_PLAN.md` §9–9c is the precedent — every gain so
far came from changing the base model and the batch size, not from more data.

### 4.2 Money

**As things stand, nothing costs anything**: there is no PAYGO project, and the
`gratis` account never bills. The figures below apply only if we later buy
non-preemptable time. UBELIX PAYGO prices are behind the internal calculator (UniBE
VPN), so this plan still cannot state CHF. On GPU nodes **only the GPU is billed** —
CPU and memory on a GPU node are free — so the budget is exactly `GPU-hours × rate`:

* subsample, 3 epochs: **~90 H100-GPU-hours**
* full set, 3 epochs: **~750 H100-GPU-hours**
* data prep: **0 GPU-hours** (CPU partition, and CPU billing is `max(cpu, mem)`)

## 4.3 Getting a PAYGO project (4× H100, non-preemptable)

**What paying actually buys.** The free `job_gpu_preemptable` QoS already grants
**h100=4**. A PAYGO project at the same 4 GPUs therefore buys *no extra capacity* —
it buys **freedom from preemption** and a shorter queue. That is a real thing to want
for a 10-day run, but it is worth naming, because the intuition "we need to pay to get
4 H100s" is not correct here.

| | free (`job_gpu_preemptable`) | PAYGO (`job_gpu`) |
|---|---|---|
| GPUs | h100=4 | h100=4 (confirm with `sqos` once the project exists) |
| walltime | 24 h | 24 h — **same** |
| interrupted? | **yes, any time** | no |
| cost | none | GPU-hours × rate |

Both cap at 24 h, so **checkpoint-and-requeue is required either way**. Paying does
not remove the need for a working restart path; it only removes the unplanned restarts.

**What it costs.** Only the GPU is billed on GPU nodes — CPU and RAM there are free.
So the budget is `GPU-hours × rate`, and our figures are:

| run | H100-GPU-hours |
|---|---:|
| phase 3 scaling test | ~10 |
| phase 4, subsample 800 K lines, 3 epochs | ~90 |
| phase 5, full 8 M lines, 3 epochs | ~750 |
| **all three** | **~850** |

The rate is only visible on the internal UBELIX calculator (UniBE VPN), so **check it
before setting a cost limit** — 850 GPU-hours is the number to multiply.

**How to request one.** This cannot be done from the command line, and it needs
authority and a budget that a job script does not have:

1. A **technology manager of the institute** creates the PAYGO project in the **IAM
   portal** (`serviceportal.unibe.ch`). Only they can; they may appoint delegates.
2. It requires a **credit number** (cost centre) and a **cost limit** at creation.
3. Add `th19c587` as a project member.
4. The project gets a **wckey**; verify with `swckeys` (today it returns `noop`) and
   check the granted ceilings with `sqos`.

Then the job header changes, and nothing else does:

```bash
#SBATCH --account=paygo
#SBATCH --wckey=<PROJECT>
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu
#SBATCH --gres=gpu:h100:4
```

`sbatch` prints `This job generates no costs!` on the gratis account — its absence is
the confirmation that a job is billing.

**Recommendation: do phases 2–4 free first.** They cost ~100 GPU-hours we do not have
to pay for, and they produce the two numbers that would otherwise make the cost limit a
guess — the real line count and the real samples/s. Phase 5 is the only stage where
preemption genuinely hurts, and by then we will know what it costs.

## 5. Pipeline

```
  Research Storage  (wbkolleg_dh_1, exposed at /storage/research/...)
        │  no copy, if §3.1 is granted
        ▼
  [CPU job, epyc2, job_cpu_long]   crop_and_shard.py
        │  548 K pages → ~8 M crops → ~160 WebDataset shards (~320 GB)
        ▼
  workspace: shards/  +  manifest.json  (the seeded, page-disjoint split)
        │  staged at job start, 100 Gb/s
        ▼
  [GPU job, 8× H100, 24 h chunks]  torchrun --nproc_per_node=8 train_qlora.py
        │  checkpoint every N steps → workspace; SIGTERM trap; --requeue
        ▼
  adapter → eval → publish_to_hub.py   (unchanged from what we have)
```

**Data prep** [estimate]: 548 K pages, JPEG decode + PageXML parse + ~15 crops
each. One epyc2 node (2×96 cores) does this in **3–6 h** dominated by I/O, or an
hour as a 4-way job array. Effectively free, and it runs on the 96 h/16-day CPU
QoS so walltime is not a concern.

**Staging**: copy the ~320 GB of shards to `/scratch/local` in the job prologue
(~5–10 min over 100 Gb/s). Every epoch after the first then reads from node-local
NVMe instead of the shared filesystem. This is why sharding matters twice.

**Software**: build one Apptainer image on the submit node from the existing
`vlm-train` venv pins (torch 2.8.0+cu128, transformers ≥4.57, peft/trl/
bitsandbytes) and keep the `.sif` in the workspace. This sidesteps the Lmod
module stack, makes the run reproducible, and means the UBELIX environment and
asterAIx run identical code. `APPTAINER_TMPDIR`/`APPTAINER_CACHEDIR` must point
at scratch — the build does not fit in `$HOME`.

**Multi-GPU**: DDP via `torchrun` on a single node, 8 ranks. Do **not** start
multi-node. An 8B LoRA does not need it, and it adds a failure mode per node.

### 5.1 Using scratch

`/scratch/network/users/th19c587` — **15 TB personal quota, 0 used**, on a filesystem
with 129 TB free. This is where the campaign lives. The share is 88 % full and shared
with the whole DH group; it is a **read-only source** for this work plus a home for a
few small promoted artifacts, nothing else.

```
/scratch/network/users/th19c587/
├── apptainer/{tmp,cache}   build scratch — deletable at any moment
├── shards/                 crop shards, ~320 GB, ~160 files   ← the big one
├── runs/<jobid>/           per-job working dir: logs, live checkpoints
├── smoke/                  phase 1 outputs
└── .campaign_marker        creation date, for purge accounting
```

**The rule that makes this safe: scratch holds only things we can rebuild.**

| artifact | lives on | why |
|---|---|---|
| `vlm-train.sif` | **`$HOME/ubelix/`** | 1 TB private quota, snapshotted, 8.6 GB used. Not the share (group quota), not scratch (purged). |
| the dataset | share, read-only | already there, 1.3 TB, never copied |
| **split manifest** | **share** (small) | *the* reproducibility anchor — see below |
| crop shards | **scratch** | ~320 GB, regenerable in a few free CPU-node-hours |
| live checkpoints | scratch, `runs/<jobid>/` | churn; only the last few matter |
| **final adapter + eval report** | **share** | ~1 GB, few files, must outlive everything |

The **split manifest** — which page went to train vs val, under which seed — is a few MB
of JSON and belongs on the share, because it is what makes the shards a *cache*. Given
`dataset + manifest + code version`, the shards are reproducible byte-for-byte. Without
it, a purge costs a re-run that is not comparable to the previous one. Write it before
the shards, not after.

**The 30-day purge.** Files unaccessed for 30 days are deleted. During an active
campaign every epoch reads every shard, so the shards keep themselves alive; the risk
window is idle time between phases. `.campaign_marker` records when the tree was
created. If a gap longer than three weeks is coming, either `find $S -exec touch {} +`
or — better — accept the purge and rebuild from the manifest.

### 5.2 The 31 August outage

**Research Storage and UBELIX go offline Monday 31 August 2026 for several days**
(new infrastructure). Four days of notice, of which today and tomorrow are working time.

What this changes:

* **Do phase 1 now.** The `.sif` lands in `$HOME`, which is snapshotted and survives.
  Once it exists, everything after the outage is unblocked.
* **Do *not* start phase 2 before the outage.** The crop campaign writes ~320 GB to a
  filesystem that is about to be physically worked on, and scratch has no backup and no
  snapshots. Three to six CPU-hours spent Friday could simply be gone Thursday next week.
  It costs nothing to run it afterwards.
* **Nothing valuable may be on scratch on Sunday evening.** Anything worth keeping goes
  to `$HOME` or the share before then. Scratch should contain only the Apptainer build
  cache, which is disposable by design.
* **Re-verify after the cluster returns**, before trusting any path: `quota` (the share
  was at 88 % — new infrastructure may or may not change that), the mount at
  `/storage/research/wbkolleg_dh_1`, the 691-file dataset count, and `sqos` (QoS ceilings
  are exactly the kind of thing that gets re-provisioned).

The outage is also the reason not to be tempted by a long preemptable run this weekend:
a 24 h chunk started Saturday would be killed mid-flight by the shutdown, and the
restart path has not been tested yet.

### 5.3 sbatch skeleton

```bash
#!/bin/bash
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable  # free, 24 h, h100=4 — and killable at any moment
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64         # 16/GPU × 4 — at the limit, not over it
#SBATCH --mem=360G                 # 90G/GPU × 4
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300        # 5 min warning before the wall
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out

trap 'kill -USR1 $PID' USR1        # trainer catches it, writes a checkpoint, exits 0

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export SHARDS=/scratch/network/users/$USER/medieval-shards  # not the share: 88% full
export APPTAINER_TMPDIR=/scratch/network/users/$USER

srun --ntasks=1 cp -r "$SHARDS" /scratch/local/$SLURM_JOB_ID/

apptainer exec --nv "$SIF" torchrun --standalone --nproc_per_node=4 \
    train_qlora.py --data /scratch/local/$SLURM_JOB_ID \
                   --resume-from-latest \
                   --checkpoint-every 500 &
PID=$!; wait $PID
```

`--resume-from-latest` doing the right thing on an empty checkpoint directory is
the whole restart contract: UBELIX's documented recipe is *look for a state file,
restore it if present, otherwise start from scratch, and save periodically*. The
same code path covers both the 24 h wall and preemption on `gpu-invest`.

## 6. Phases

| # | phase | needs | output |
|---|---|---|---|
| 0 | ~~Access & storage~~ | — | **done** — login works, share is mounted, dataset and base model already cached |
| 1 | ~~Port & smoke~~ | free, 9 min | **done** — job 14108981, CER 0.4662 vs 0.466. `vlm-train.sif` in `$HOME/ubelix/` |
| 2 | **Crop & shard** — `crop_and_shard.py` on epyc2, full 548 K pages, output to `/scratch/network` | CPU only, free | ~160 shards, **the real line count** |
| 3 | **Scaling + restart test** — 1 → 2 → 4 H100, fixed 50 K lines, bf16 vs NF4, and a deliberate `scancel --signal=USR1` to prove resume | ~10 GPU-h, free | **real samples/s**, and a restart path we trust |
| 4 | **Production (subsample)** — stratified 800 K lines, 3 epochs, 4× H100 preemptable | ~90 GPU-h, free | the model we actually ship |
| 5 | **Full set** — only if phase 4's curve is still falling | ~750 GPU-h, free but ~10 days | — |
| — | *(parallel, optional)* **PAYGO project** via the institute technology manager | admin | non-preemptable 8× H100, if calendar time starts to hurt |

Phase 2 produces the first real answer to "how many lines is this dataset", which is
now the largest single uncertainty in this document. Phase 3 produces the second.
Both are free, and neither needs anything we do not already have.

## 7. Risks

* **We may be scaling an unmeasured thing.** #52 is open: the smoke run's CER
  0.466 vs 1.837 is mostly the model learning to *stop at the line*, not to read.
  Spending 750 GPU-hours before we can tell literacy from output discipline buys
  a number we cannot interpret. Resolve #52 first — it is a scoring change, not a
  compute problem.
* **Preemption is now the main plan, so checkpointing is the main risk.** The 4×
  H100 QoS is free precisely because investor jobs may kill ours at any moment. Every
  hour of the full run rides on resume working. Test it deliberately with
  `scancel --signal=USR1` in phase 3, and do not start phase 5 until a job has
  survived a real preemption.
* **Queue time is not walltime.** 5 H100 nodes serve the whole university and all
  were `mix` when probed. Asking for 4 H100s will wait sometimes; asking for 8 on one
  node would wait a lot. Estimate wall clock as compute + queue + preemption restarts,
  and prefer the shorter subsample job for that reason too.
* **The share has ~1.4 TB and 3.2 M inodes left**, and other people write to it.
  Nothing this campaign produces should land there — crops to `/scratch/network`,
  checkpoints to a single small directory on the share or to scratch. Check `quota`
  before and after every stage.
* **GPU investment is closed** until the 2026 datacenter expansion, so the
  investment model is not an option for this campaign regardless of budget.
* **30-day scratch purge.** Anything on `/scratch/network` that has not been read
  in 30 days is deleted. Shards live in the workspace; scratch is a staging area
  only.
* **The 30B MoE variant.** `lassberg` targeted Qwen3-VL-30B-A3B. It fits on an
  H100 and would train roughly at 8B's speed (3B active params), but nothing we
  operate could then serve it — vLLM 0.11 would want the whole card. Train the 8B
  unless the serving story changes.

## 8. Open questions to resolve before phase 4

1. ~~Can `wbkolleg_dh_1` be exposed as a UBELIX workspace?~~ **Already mounted.**
2. Is the cache complete — 691 parquet files present against 694 in the export?
3. What is the actual line count and crops-per-page distribution? → phase 2
4. What is the actual samples/s at bf16 on H100, and does DDP scale to 4? → phase 3
5. How often does `job_gpu_preemptable` actually get preempted? Only a real run tells
   us, and it sets the wall-clock estimate for phase 5.
6. Do we want a PAYGO project at all, given the free path works?
7. Is `modules_to_save: ["lm_head"]` worth it here? On a dedicated card it is
   affordable (~620 M extra trainable params) and the medieval character
   repertoire is exactly the case `VLM_TRAINING.md` says it helps.
