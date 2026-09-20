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
| free-tier GPU ceiling | 1× H100 | **4× H100** preemptable (24 h) — and **paying gets only 1**, see §4.3 |
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

**`job_gratis` also carries `MaxTRESRunMinsPU cpu=11520`** — the F1 free tier's
"8 cores for 24 h", enforced as CPUs × minutes. At the 16 CPUs an H100 allocation
wants, that is a **12-hour ceiling**, and a longer job is rejected at submit with
`MaxCpuRunMinsPerUser`, not queued. **`job_gpu_preemptable` has no such limit** —
only the GPU ceilings and the 24 h wall. So the preemptable QoS is not merely the
bigger one, it is the only one that can run a long job at full CPU width, which
matters whenever dataloader workers are part of what is being measured.

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
| **line crops** | **~10.4 M** [measured-derived] |

The line count is the number that drives everything, and it is not published.
**Measured on this corpus, 2026-09-09** (experiment C's prepare stage): 515 pages
materialized → 115 skipped for having no usable transcription → **400 pages
yielding 9,785 transcribed lines**. That is **19.0 lines per materialized page**,
and a **22 % page-attrition rate** worth budgeting for.

548,322 × 19.0 ≈ **10.4 M line crops**.

This supersedes the earlier ~8 M, which came from the 52-page Thun demo at ~15
lines/page — a demo project, not the target corpus. Every schedule below scales
linearly with it, so the correction is ~30 % more work than this document
originally assumed.

For scale: `thun-kurrent-v2`, our best kraken model, was trained on **1,898
lines**. The full medieval set is roughly **5,500× that**.

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

**As things stand, nothing costs anything**: there is no PAYGO project, the `gratis`
account never bills, and preemptable jobs are free even under a project. Prices are now
known (§4.3): **H100 CHF 0.60/h, RTX 4090 CHF 0.10/h**, billed per minute. On GPU nodes
only the GPU is billed — CPU and memory there are free — so the budget is exactly
`GPU-hours × rate`:

* subsample, 3 epochs: ~90 H100-GPU-hours → **CHF 54**
* full set, 3 epochs: ~750 H100-GPU-hours → **CHF 450**
* data prep: **0 GPU-hours** (CPU partition, and CPU billing is `max(cpu, mem)`)

## 4.3 What paying would buy — checked, 2026-08-27

**Published prices** (intern.unibe.ch, UniBE VPN; per-minute billing):

| resource | CHF/hour |
|---|---:|
| 1 CPU | 0.002 |
| 1 GPU — RTX 4090 | **0.10** |
| 1 GPU+ — H100 | **0.60** |

and, decisively: **"debug/preemptable jobs are free."**

### The ceiling: 4 H100s, and only for free

Every QoS on the cluster, checked with `sacctmgr show qos`:

| QoS | H100/user | walltime | cost | ours? |
|---|---:|---|---|---|
| **`job_gpu_preemptable`** | **4** | 24 h | **free** | **yes** |
| `job_gratis` | 1 | 96 h | free | yes |
| `job_gpu` (**paygo**) | **1** | 24 h | 0.60/h | needs a project |
| `job_interactive` | 1 | 12 h | 0.60/h | needs a project |
| `job_debug` | 1 | 20 min | free | yes |
| `job_gpu_<investor>` | **no cap** | up to **7 days** | CAPEX | **closed until the 2026 DC expansion** |

**Paying does not get us 4 H100s — it gets us one.** The only non-investor QoS on
UBELIX that grants more than a single H100 is the free preemptable one. 40 H100s exist
across five nodes, but no QoS we can obtain releases more than four of them.

So the answer to "how many H100s can we reasonably put together" is **four**, and that
is both the administrative ceiling and roughly the technical one: an 8B LoRA is
single-card work, DDP across 4 cards on one node is the sweet spot, and multi-node
NCCL for a model that fits in 24 GB buys complexity, not speed.

### Which means paying is strictly worse here

Three epochs over 8 M lines, using §4's throughput estimates:

| configuration | samples/s | wall clock | CHF/h | **total** |
|---|---:|---:|---:|---:|
| **4× H100, preemptable** | ~41 | **6.8 days** | — | **free** |
| 1× H100, paygo | ~12 | 23 days | 0.60 | 333 |
| 4× RTX 4090, paygo | ~12 | 23 days | 0.40 | 224 |
| 1× RTX 4090, paygo | ~3.5 | 79 days | 0.10 | 190 |

The free option is **both cheaper and 3.4× faster** than the best paid one. Paying buys
freedom from preemption, but at one quarter of the parallelism — so a paid run finishes
*later* than a preempted free run would, even with no interruptions at all.

**Recommendation: do not set up a paid plan to get H100s. It is not on offer.**

### When a project is still worth creating

Two reasons that have nothing to do with H100 count:

1. **The F2 free tier refunds up to CHF 1000 per cost centre per year**, applied at
   the end of the fiscal year by internal transfer. Our entire campaign — phases 3, 4
   and 5, ~850 H100-GPU-hours — would be **CHF 510**, comfortably inside that. Paid
   fallback capacity is therefore effectively free up to the ceiling.
2. **Preemptable jobs stay free under a project.** Having a wckey costs nothing and
   changes nothing about the primary plan; it just means that when preemption starts
   thrashing a run, switching to a billed 4× RTX 4090 job is a header edit rather than
   a two-week procurement.

If you want it: the IT-responsible person of the institute orders it at
`iamportal.unibe.ch` → **"HPC - Order new Project Space"**, giving organisational unit,
project name (this becomes the wckey), **cost-centre number without the REF prefix**,
administrators, members, and a **monthly** cost ceiling. The cost-centre owner must
confirm before it activates. Prices are re-quoted twice a year and may move — GPU
pricing is volatile, and the page says so.

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

---

## 9. Proposal, 2026-09-09: what to run on UBELIX next

Written after the corpus-scale runbook (`VLM_TRAINING.md` §5, four failed attempts on
asterAIx) and after checking what the Qwen3.5/3.8 line would actually require.

### 9.1 The recent tests move the goalposts — my §4 estimates were ~3× optimistic

`VLM_TRAINING.md` §5 measures a **325 K-line** corpus on asterAIx at **0.67 samples/s**
— three times worse than the 1.94 samples/s of the Thun smoke test that every estimate
in §4 of this document was extrapolated from. One epoch took **6.4 days**.

The runbook names two candidate causes and states plainly that **they have not been
separated**:

1. **IO** — `compile` writes 337,623 individual JPEGs to the CIFS share and `train`
   reads them back one at a time.
2. **Longer lines** — median aspect 9.9 against Thun's much squarer crops, so more
   visual tokens per sample at the same budget.

Corrected, this is what the full medieval set costs on 4× H100:

| anchored on | samples/s (4× H100) | one epoch over 8 M lines | 3 epochs |
|---|---:|---:|---:|
| Thun smoke test (§4) | ~41 | 2.3 days | **6.8 days** |
| **corpus-scale run (§5)** | **~14** | **6.6 days** | **20 days** |

**That gap is the whole decision, and one cheap experiment resolves it.**

### 9.2 Experiment A — ANSWERED, 2026-09-09: compute-bound. Do not stage.

Jobs 14443285 + 14455754, one H100 NVL, free preemptable QoS, zero cost.
Corpus: the medieval set, `all_projects` capped at 400 pages → **8,668 train
samples, 9,785 crop files, 712 MB**. Three passes, same compiled JSONL, same
seed, same node, same argv — only `--data-root` differed. Timed by HF Trainer's
own `train_runtime`.

| pass | crops on | train_runtime | samples/s |
|---|---|---:|---:|
| A1-cold | `/scratch/network` (GPFS), first read | 1058.4 s | **8.19** |
| A2 | `/scratch/local` (node NVMe) | 1042.4 s | **8.32** |
| A1-warm | `/scratch/network`, page-cached | 1037.0 s | **8.36** |

* page cache (A1-cold − A1-warm): **+2.0 %**
* NVMe vs GPFS, both warm (A1-warm − A2): **−0.5 %**

**All three are within 2 % of each other. This workload is compute-bound on
UBELIX, and local staging is not the lever.** The control did its job: the
page-cache effect is 2 %, so the A2/A1 comparison is not an artefact of caching.

**Staging is worse than neutral — it costs.** Copying 9,785 crops to node NVMe
took **43 s and 136 s** on the two runs. Scaled to 8 M crops that is **10–31
hours of pure copying** per job, to buy a measured −0.5 %. The §5 pipeline's
"stage to `/scratch/local` in the prologue" step should be **dropped**.

**What this does and does not settle.** It rules out *shared vs local disk* as
the mechanism, because on UBELIX all three storage arms are indistinguishable.
It does not prove the asterAIx 0.67 samples/s was the CIFS mount rather than
that corpus's longer lines — the corpora differ (medieval here, German there),
so those two remain unseparated. What is certain is that **the 3× corpus-scale
penalty asterAIx saw does not reproduce here**: 8.2 samples/s on a corpus-scale
selection against 1.94 on asterAIx's 52-page smoke test, on a card roughly 4×
faster.

### 9.2-bis The schedule, re-anchored on measurement

8.2 samples/s per H100 [measured], 4 GPUs at 85 % DDP → **~28 samples/s**:

| anchored on | 4× H100 | one epoch, 8 M lines | 3 epochs |
|---|---:|---:|---:|
| Thun smoke test (§4) | ~41 | 2.3 days | 6.8 days |
| corpus-scale asterAIx (§9.1) | ~14 | 6.6 days | 20 days |
| **measured on UBELIX** | **~28** | **3.3 days** | **~10 days** |

Ten days of free preemptable time for three epochs over the whole medieval set.
That is a real campaign, not a hopeful one.

One lever is still unspent: this ran **4-bit NF4**, inherited from a box that
shares its card. A 96 GB H100 has no such constraint, and bf16 removes the
misaligned-bitsandbytes path the 2026-08-08 run documented. That is the next
cheap measurement, and it is a `--no-load-in-4bit` flag.

### 9.3-A Experiment B — bf16 beats 4-bit by 23 %, 2026-09-09

Job 14478418, one H100 NVL (100 GB), same compiled corpus, same argv, one
variable. B1 re-measured 4-bit inside the job rather than trusting experiment A's
number across nodes.

| pass | train_runtime | samples/s | peak VRAM |
|---|---:|---:|---:|
| B1 — NF4 4-bit | 1035.2 s | 8.37 | **21.6 GB** |
| B2 — **bf16** | **798.6 s** | **10.85** | **39.7 GB** |

**bf16 is 22.9 % faster and uses 40 GB of a 94 GB card.** The 4-bit default is
inherited from a box that shares its A40 with the serving engines; on a card we
own outright it buys nothing and costs a quarter of the throughput — the
misaligned-bitsandbytes kernel the 2026-08-08 run logged, paid for a memory
saving nobody needs. **`--no-load-in-4bit` is the UBELIX default from here.**

*(Instrumentation caveat: the summary table this job printed shows 39.7 GB for
both rows. The `kill` of the first pass's `nvidia-smi` sampler did not take — it
captured the wrapper's PID — so it kept writing into B1's file during B2. The
per-pass figures above are the inline readings taken at the end of each pass,
which are correct; the end-of-job table's VRAM column was not.)*

### 9.3-B Experiment C — 4B matches 8B; size is not the lever, 2026-09-09

Job 14479259, three full runner pipelines, same corpus, same val split, same
scorer, 200 eval samples. All three at 4-bit.

| model | train_runtime | samples/s | CER | WER |
|---|---:|---:|---:|---:|
| Qwen3-VL-2B | 808.0 s | 10.73 | 0.4386 | 0.7528 |
| **Qwen3-VL-4B** | 1019.4 s | 8.50 | **0.3344** | **0.6553** |
| Qwen3-VL-8B | 1036.1 s | 8.37 | 0.3411 | 0.6605 |

Two findings, and the second is the more important one.

**4B matches 8B — actually edges it** (0.3344 vs 0.3411, a difference too small to
call). Half the parameters, no measurable cost in quality on this corpus. **2B is
a real drop**: 31 % worse CER than 4B for 26 % more speed, which is a bad trade.

**But shrinking the model barely buys throughput: 4B is 1.5 % faster than 8B.**
Halving the language model changed almost nothing, so the language model is not
what the time is going into — the **vision tower and the visual tokens are**. That
is the same conclusion experiment A reached from the opposite direction, and the
two now agree: this workload is bound by image processing, not by weights and not
by storage.

**So the remaining lever is `max_pixels` and the aspect cap**, not model size and
not hardware. That is the next experiment, and it is a one-flag sweep.

### 9.3-D Experiment D — the visual-token sweep, and the hypothesis it kills

Job 14566642, Qwen3-VL-4B at bf16, four arms, one compiled corpus (8,668/1,117
in every arm — the startup budget line was captured per arm, so the knob is
proven applied, not assumed).

| visual tokens | train_runtime | samples/s | CER | WER |
|---:|---:|---:|---:|---:|
| 64 | 764.5 s | 11.34 | 0.4883 | 0.8168 |
| **128** | 767.7 s | 11.29 | **0.3480** | 0.6705 |
| **256** (default) | 802.7 s | 10.80 | **0.3449** | 0.6832 |
| 512 | 785.7 s | 11.03 | 0.3845 | 0.7169 |

**On quality:** 128 and 256 are a tie (0.3480 vs 0.3449 — under 1 %). 64 is badly
worse (+42 %). And **512 is worse than 256** (+11 %): more pixels do not buy
reading, they cost it. Keep 256; 128 is free if memory ever matters.

**On throughput: nothing happened.** An **8× range of visual tokens moved
throughput by 5 %** (10.80–11.34 samples/s). §9.3-B predicted `max_pixels` was
"the remaining lever". **It is not, and this experiment says so plainly.**

### 9.3-E What A, C and D together actually show

Line up everything measured on this corpus:

| change | expected | **measured** |
|---|---|---|
| GPFS → node NVMe (A) | large if IO-bound | **−0.5 %** |
| 8B → 4B language model (C) | ~2× | **+1.5 %** |
| 8B → 4B **at bf16** (C vs D) | ~2× | **10.85 → 10.80 = 0 %** |
| 512 → 64 visual tokens (D) | large if vision-bound | **+5 %** |
| bf16 → 4-bit NF4 (B) | — | **−23 %** |

Every change that *removes* GPU work buys nothing. The one change that *adds* GPU
work — 4-bit's dequantization — costs 23 %. **That is the signature of a GPU that
is not the bottleneck: it is waiting, and only when you make it slower than the
thing feeding it does the time move.**

Everything lands at **10.8–11.3 samples/s regardless of model size, precision,
visual-token budget or filesystem.** That ceiling is the finding.

**The prime suspect is the input pipeline.** `--workers 4` is passed on an
allocation that holds **16 CPUs**, and each worker decodes a JPEG and runs the
image processor per sample. Four workers producing ~11 samples/s is ~2.8
samples/s each, which is a plausible rate for PIL decode plus preprocessing —
and it would explain all five rows above at once.

**Experiment E, and it is cheap:** sweep `--workers` (4 → 8 → 16) and record GPU
utilization with `nvidia-smi dmon` alongside. If utilization is well under 100 %
at `workers 4`, the ceiling is confirmed as starvation and the lever has been
found. If utilization is already pinned, the ceiling is something else and this
document should stop guessing and profile.

Until that resolves, **every schedule below is a lower bound on speed** — the
10.4 M-crop campaign may be considerably cheaper than 9.8 days.

### 9.3-F Experiment E — the GPU *is* starved, and workers are not why

Job 14584141, Qwen3-VL-4B bf16, 256 visual tokens, three worker counts, GPU
utilization sampled every 5 s throughout each arm.

| workers | samples/s | CER | GPU util (median) | p90 | max |
|---:|---:|---:|---:|---:|---:|
| 4 | 11.07 | 0.3449 | **54 %** | 75 % | 87 % |
| 8 | 11.10 | 0.3449 | **54 %** | 77 % | 85 % |
| 16 | 10.97 | 0.3449 | **52 %** | 75 % | 87 % |

Identical CER across all three is a useful side-check: the runs are
deterministic, so these arms differ only in what was intended.

**Half of the answer is confirmed and half of it is refuted.**

**Confirmed: the GPU is idle about half the time.** And this is worth reading
precisely — `utilization.gpu` is the fraction of time *any kernel was
executing*, not how efficiently it ran. 54 % does not mean "poor occupancy"; it
means the card had **literally nothing to run for 46 % of the wall clock**. That
is a stall, and it explains the 10.8–11.3 ceiling that A, B, C and D all ran
into from different directions.

**Refuted: it is not the dataloader.** Quadrupling workers from 4 to 16, on an
allocation with 16 CPUs, changed throughput by 1 % and utilization by 2 %. §9.3-E
named `--workers 4` as the prime suspect. It is not guilty.

**Two hypotheses have now died this way** — `max_pixels` in D, workers in E — and
the honest reading is that guessing the mechanism from throughput numbers has
stopped being productive. What E adds is that the *shape* of the problem is now
known (a stall, not slow work), which is much narrower than what D left.

**What can stall a training loop while the dataloader has spare capacity:**

* **`paged_adamw_8bit`.** This is bitsandbytes' *paged* optimizer — designed to
  page state between GPU and CPU when VRAM is tight. We have **54 GB free**, so
  it is paying for a service nobody needs, and every optimizer step (one per 4
  micro-batches, with `accumulate_grad_batches: 4`) is a synchronous host
  transfer during which the GPU has nothing to do.
* **`batch_size: 4` with `accumulate_grad_batches: 4`.** Small steps mean more
  optimizer steps per sample, multiplying whatever the per-step overhead is.

Both are testable as a **2×2** — optimizer (`paged_adamw_8bit` / `adamw_torch`)
× batch size (4 / 16) — which discriminates between them instead of confounding
them, and costs about 90 minutes. If the 2×2 also comes back flat, the next step
is `torch.profiler` on one arm, **not** a seventh sweep.

Note this changes nothing about quality: CER 0.3449 throughout, and the schedule
in §9.3-C stands as a lower bound until the stall is understood.

### 9.3-G Experiment F — found it: the micro-batch was starving the card

Job 14613930, Qwen3-VL-4B bf16, 2×2, **effective batch 16 in every arm** so the
number of optimizer steps is constant and only the two factors move.

| arm | optim | bs | runtime | samples/s | CER | util (med) | peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| paged-b4 | `paged_adamw_8bit` | 4 | 785.9 s | 11.03 | 0.3449 | 51 % | 31.0 GB |
| torch-b4 | `adamw_torch` | 4 | 773.7 s | 11.20 | 0.3486 | 51 % | 31.4 GB |
| paged-b16 | `paged_adamw_8bit` | 16 | 499.8 s | 17.34 | 0.3386 | 65 % | 86.1 GB |
| **torch-b16** | **`adamw_torch`** | **16** | **488.1 s** | **17.76** | **0.3209** | **86 %** | 86.6 GB |

| main effect | |
|---|---:|
| optimizer, paged → torch | **1.02×** |
| **micro-batch, 4 → 16** | **1.58×** |

**The micro-batch was the answer, and the paged optimizer was innocent.** `bs: 4`
is a default inherited from a box that shares its A40 with the serving engines —
the same provenance as `load_in_4bit`, and wrong here for the same reason. On a
dedicated 94 GB H100 it left the card idle half the time. At `bs: 16`,
utilization goes **51 % → 86 %** and throughput **1.58×**.

This closes the arc A→F. Every earlier experiment was varying something the GPU
was not waiting on, which is why they all returned the same 11 samples/s: the
card was blocked on having too little work per step, and neither faster storage,
fewer parameters, fewer visual tokens nor more dataloader workers changes that.

**Two caveats before this becomes the production config.**

1. **86.6 GB of 94 GB is a thin margin.** Batches pad to their widest member and
   this corpus has a worst aspect of 61:1, so a single wide line can spike. For a
   multi-day run, either drop to `bs: 12` or keep `paged_adamw_8bit` — it costs
   2 % and its entire purpose is surviving exactly this. An OOM at hour 40 of a
   six-day run is far more expensive than 2 %.
2. **The CER spread (0.3209–0.3486) is suggestive, not established.** Effective
   batch was held constant, so these should be near-identical optimizations; the
   spread is more likely padding numerics and single-epoch variance than a real
   quality gain. Do not claim `bs: 16` improves CER on this evidence.

### 9.3-H The schedule, with the stall removed

17.76 samples/s per H100 [measured], 4 GPUs at 85 % DDP → **~60 samples/s**:

| anchored on | 3 epochs over 10.4 M crops |
|---|---:|
| corpus-scale asterAIx (§9.1) | 26 days |
| measured, 4-bit, bs 4 | ~13 days |
| measured, bf16, bs 4 | ~9.8 days |
| **measured, bf16, bs 16** | **~6 days** |

**The production configuration, now fully measured:** Qwen3-VL-4B · bf16 ·
`max_pixels` 262144 (256 visual tokens) · `batch_size` 16 · `workers` 8 · crops
read straight off GPFS · 4× H100 preemptable. Six days, free, for three epochs
over the whole medieval set.

### 9.3-I Queue time is not free, and it may dominate

Measured 2026-09-10: a 1× H100 job on `job_gpu_preemptable` was given an
estimated start of **the following morning — a ~14 hour wait**, with **26 other
pending H100 requests** on that queue.

This matters more than it looks. Every schedule in this document counts *compute*
time. The preemptable path runs in 24 h chunks, and if each chunk waits hours to
start, a **6-day compute campaign is a calendar campaign of unknown length**. The
correct way to read §9.3-H is now:

* **~6 days of GPU time** — measured, reliable.
* **calendar time — unknown, and demand-dependent.** It is bounded below by 6
  days and could be two or three times that when the queue looks like this.

Two consequences:

* A run should hold its allocation rather than release it. Requeueing after every
  24 h chunk means re-entering a queue that may be a day deep, so **fewer, longer
  chunks are worth more than the QoS ceiling suggests** — which is an argument for
  the 96 h `job_gratis` QoS (1× H100, no preemption) over 4× H100 preemptable
  whenever the queue is congested, despite the 4× fewer GPUs.
* Queue depth should be checked *before* choosing a QoS for a long run, not
  assumed. `squeue -p gpu-invest -h -t PENDING -o "%b" | grep -c h100`.

### 9.4 Preemption-resume — implemented and proven on GPU (#111)

Phase 3's gate, and the thing that had to work before any multi-day run. Before
this, **nothing here survived an interruption**: SIGTERM meant `cancelled`,
`cancelled` is terminal, and the trainer checkpointed once per epoch — so a job
stopped at hour 23 of a two-day epoch resumed from nothing.

#### What was built

| piece | what it does |
|---|---|
| `Preempted` ≠ `Cancelled` | a cancellation is a decision, a preemption an interruption. SIGINT still cancels either way — a person pressing Ctrl-C means stop. |
| exit **75** (`EX_TEMPFAIL`) | a requeue is distinguishable from a real failure. Requeueing genuine failures forever is how a broken job burns a week of GPU. |
| `training` self-edge in `TRANSITIONS` | the only loosened edge. `completed`/`failed`/`cancelled` stay terminal. |
| `VlmTrainParams.save_steps` | step checkpointing, because an epoch at corpus scale is days |
| `_resume_artifacts` | a **different question** from `_reuse_artefact` |
| `ubelix/train_resumable.sbatch` | `--requeue`, a stable job id keyed on `SLURM_JOB_ID`, SIGTERM forwarded, exit 75 handled |

#### Two traps the codebase had already documented

1. **Moving `eval_strategy` to steps alongside `save_strategy`** is the obvious
   fix and is wrong. `make_recovery_callback` explains why: the continuation
   callback (#88) counts one evaluation as one epoch, so a steps-based eval ends
   a `max_epochs: 3` run after three evaluations, a few hundred steps in. **Only
   saving moves.** `load_best_model_at_end` goes with it, which is a trade rather
   than a loss in the single-epoch corpus runs this mode exists for.
2. **`_reuse_artefact` is the wrong hook.** It asks whether *another* job's
   corpus can be adopted, and answers None for this backend by design — the VLM
   JSONL names image paths inside its own job directory, so it is not
   relocatable. That same property is what makes resuming trivial: the files are
   still in *this* job's directory, and a requeue does not delete it.

#### Two interruption modes, and they are not the same

This is the part that took two GPU tests to get right.

| | **preemption / requeue** | **walltime expiry** |
|---|---|---|
| what Slurm does | signals, then **kills the step promptly** | `--signal=B:TERM@120` signals **120 s early** |
| is there time to shut down? | **no** | yes |
| what keeps the job resumable | nothing writes a terminal status, and `JobStore.save` is tmp-then-`os.replace`, so a hard kill cannot corrupt the record | the `Preempted` handler, exit 75, the batch script's requeue branch |

**The graceful path is a walltime mechanism, not a preemption one.** An earlier
version of this section claimed otherwise.

#### What the tests actually proved

**Job 14681323 — a false pass.** It resumed and completed, which looked like
success. The runner had exited **143**, killed by SIGTERM: the handler never ran
and the record survived only because a hard kill leaves it alone. The bug was in
the batch script — **bash's `wait` returns as soon as a *trapped* signal arrives,
with 128+signum, without waiting for the child**, so the script read 143 as the
runner's status, skipped the exit-75 branch, and killed the runner mid-shutdown
by exiting. Fixed by re-waiting (`0e724cc`); verified in isolation, where the
first `wait` returns 143 and the second returns the child's 75.

**Job 14687032 — the real pass.** Same job id across a genuine `scontrol
requeue`:

```
attempt 1 …  train.log: "no checkpoint found; starting from scratch"
== SIGTERM received, forwarding to 490117
attempt 2 …  train.log: "resuming from …/checkpoint-160"
             runner.log: "re-entered while `training` — resuming after preemption"
== runner exited 0 · status: completed
```

Attempt 2 picked up at step 160 and ran the epoch out to 1101 steps. **No
orphaned trainer, no held GPU, record coherent throughout.** Attempt 1's runner
log ends at the trainer invocation — no `Preempted`, no exit 75 — which is how
the preemption column above was established rather than assumed.

#### Status

| | |
|---|---|
| implementation | **done**, on `main` |
| unit coverage | **1017 tests**, 14 new; verified on a clean checkout of HEAD |
| resume across **`scontrol requeue`** | **proven on GPU** (14687032) |
| resume after a **hard kill** | **proven on GPU** (14681323) |
| graceful **walltime** shutdown → exit 75 | **does not fire** — see below |
| walltime chunking without a human | **proven** (14701151): three attempts, `no checkpoint` → `checkpoint-280` → `checkpoint-680`, requeued automatically |
| resuming from a **half-written** checkpoint | **was fatal**; fixed (`last_complete_checkpoint`) |

#### The bug that mattered: a kill during a save ended the run

Job 14701151 chunked correctly through three attempts and then died:

```
FileNotFoundError: …/checkpoint-680/trainer_state.json
StageFailed in train  →  status: failed
```

`transformers.trainer_utils.get_last_checkpoint` returns the highest-numbered
`checkpoint-N` directory that **exists**. On a preemptable queue that is not the
same as one that can be **resumed from**: the directory is populated
progressively, so a job killed mid-save leaves a partial one,
`get_last_checkpoint` hands it back, and the resume dies — which the runner
records as a failed stage. **`failed` is terminal**, so a kill that happened to
land during a save turned a resumable six-day run into a dead one.

`last_complete_checkpoint` walks the checkpoints newest-first and takes the first
with a `trainer_state.json` — the file the Trainer writes **last**, so its
presence means the rest of the directory is already there. Falling back one
checkpoint costs at most `save_steps` of redone work, which is exactly what that
setting exists to bound.

**This is the failure only a real preemption could have found.** Every unit test
passed before it, both earlier GPU tests passed, and the design was wrong in a way
that would have surfaced days into the production run.

#### The graceful path does not fire, and nothing depends on it

Job 14697771, an 8-minute wall with `--signal=B:TERM@120`: Slurm signalled twice,
the trap forwarded twice, and **the runner did not act** — no `Preempted`, no
exit 75, nothing in `runner.log`. The job ran to `TIMEOUT`. In an isolated test
the same signal through the same container *does* reach Python and exits 75, so
the handler is correct in principle and something about the real runner — most
likely being blocked waiting on the detached trainer — swallows it.

**This does not endanger a long run, and it is worth being clear why.** What keeps
a job resumable is not the handler:

* a killed runner **writes no terminal status**, so the record stays `training`;
* `JobStore.save` is **tmp-file-then-`os.replace`**, so even a kill mid-write
  cannot corrupt it;
* the trainer checkpoints every `save_steps`, so at most that much work is lost.

All three were confirmed on GPU. The record in 14697771 was `training` after the
timeout, exactly as required.

#### But a walltime expiry does not requeue itself

That *was* a real gap. A preemption requeues via `--requeue`; a **TIMEOUT simply
ends the job**, leaving it resumable but waiting for a human. A six-day run in
24 h chunks would stall at every wall. Fixed by calling `scontrol requeue` **from
the trap**, 120 s before the wall — which does not depend on the runner handling
the signal at all.

---

## 10. What the campaign established, in one table

Every row measured on the same 400-page medieval selection (8,668 train / 1,117
val crops), Qwen3-VL unless stated.

| # | question | answer |
|---|---|---|
| A | is it IO-bound? | **no** — GPFS, node NVMe and page cache within 2 %. Staging *costs*: 10–31 h of copying at full scale for −0.5 %. |
| B | is 4-bit worth it? | **no** — bf16 is **+23 %** and uses 40 GB of a 94 GB card. 4-bit is an asterAIx inheritance. |
| C | does a smaller model do? | **4B matches 8B** (CER 0.334 vs 0.341) at half the size. 2B is 31 % worse. |
| D | are visual tokens the lever? | **no** — an 8× range moved throughput 5 %. 128 and 256 tie on CER; 512 is *worse*. |
| E | is the GPU starved? | **yes — idle 46 % of wall clock.** But not by the dataloader: 4→16 workers changed 1 %. |
| F | what was it waiting for? | **the micro-batch.** bs 4→16 is **+58 %**, utilization 51 %→86 %. The optimizer was innocent (1.02×). |

**Production configuration:** Qwen3-VL-4B · bf16 · 256 visual tokens ·
`batch_size` 16 · `workers` 8 · straight off GPFS · `save_steps` set ·
4× H100 preemptable.

**Cost:** ~6 days of GPU time for 3 epochs over the measured ~10.4 M crops, free.
Calendar time is demand-dependent and can be far longer — see §9.3-I.

**The through-line.** A, C, D and E each varied something the GPU was not waiting
on, which is exactly why they all returned the same ~11 samples/s. Two defaults
inherited from a box that *shares* its A40 with the serving engines —
`load_in_4bit` and `batch_size: 4` — were together costing a factor of ~2 on
hardware we own outright. Neither was a bug; both were right where they came from.

### 9.3-C The schedule, re-anchored again

bf16 at 10.85 samples/s per H100 [measured], 4 GPUs at 85 % DDP → **~37 samples/s**:

| anchored on | 3 epochs, 8 M lines |
|---|---:|
| Thun smoke test (§4) | 6.8 days |
| corpus-scale asterAIx (§9.1) | 20 days |
| measured, 4-bit (§9.2-bis) | ~10 days |
| **measured, bf16** (8 M) | ~7.5 days |
| **measured bf16, at the measured 10.4 M line count** | **~9.8 days** |

The production configuration these three experiments point at: **Qwen3-VL-4B,
bf16, crops read straight off GPFS, 4× H100 preemptable.** Untested as a
combination — B and C each moved one variable, and 4B+bf16 together is the
obvious confirmation run before committing ten days of anything.

### 9.2-ter (superseded) The original NVMe proposal

The runbook says copying the crops to local disk "is the obvious experiment and has not
been run". On asterAIx it is awkward; **on UBELIX it is free and native**: every GPU
node has **1.92 TB of local NVMe at `/scratch/local`**, and §5 of this document already
specifies staging there.

Same corpus, same model, same seed, two jobs on one RTX 4090 or H100, a few hours each:

| arm | crops live on | measures |
|---|---|---|
| A1 | `/scratch/network` (shared, like the CIFS baseline) | the IO-bound hypothesis |
| A2 | `/scratch/local` (node NVMe, staged in the prologue) | the compute-bound floor |

If A2 ≈ A1, the cost is visual tokens and the answer is to cut the budget or the
aspect cap. If A2 ≫ A1, the cost is IO and the sharding work in §3.2 is worth far more
than any hyperparameter. **Either result changes the plan; the run is free; nothing
else should be started before it.** It is also model-independent, so it is not wasted
if the base model changes.

### 9.3 Qwen3.5 / Qwen3.8 — possible, but it breaks a pin the repo defends

The Qwen3.5 line **folded vision into the base models**: there is no separate `-VL`
variant any more, and every checkpoint is `image-text-to-text`. Qwen3.8-27B is the
newest member and shares the architecture id, so it is the *same* code path.

| model | arch id | safetensors | released | note |
|---|---|---:|---|---|
| Qwen3-VL-8B-Instruct | `qwen3_vl` | 17 GB | Oct 2025 | **what we train today** |
| **Qwen3.5-9B** | `qwen3_5` | 19.3 GB | Feb 2026 | like-for-like successor |
| Qwen3.5-4B | `qwen3_5` | 9.3 GB | Feb 2026 | cheap, for the A/B |
| Qwen3.5-27B | `qwen3_5` | 55.6 GB | Feb 2026 | |
| **Qwen3.8-27B** | `qwen3_5` | 55.6 GB | Aug 2026 | newest, most popular |
| Qwen3.5-35B-A3B | `qwen3_5_moe` | 71.9 GB | Feb 2026 | 3 B active — cheap to train, **unservable here** |

**The blocker, measured not guessed.** Our container's transformers 4.57.6 raises
`does not recognize this architecture: qwen3_5`. And **the 4.x line ended at 4.57.6 on
2026-01-16, before Qwen3.5 shipped** — so `qwen3_5` exists *only* in transformers 5.x.
That collides head-on with `engines/vlm_train_svc/requirements.txt`, which pins
`transformers>=4.57,<5` and says the cap is there because the trainer is written
against the 4.x surface: *"Lift the cap only together with a run that actually trains
on 5.x."*

Verified on the cluster: **transformers 5.16.1 resolves `qwen3_5`, `qwen3_5_moe` **and**
still `qwen3_vl`.** One rebuilt container can therefore run the old and new base models,
which is what makes an honest A/B possible instead of a leap.

The port surface is small and fully enumerated — `AutoModelForImageTextToText`,
`AutoProcessor`, `Trainer`, `TrainingArguments`, `BitsAndBytesConfig`, `TrainerCallback`,
`set_seed`. Notably `Trainer(...)` never passes `tokenizer=`, so the rename that breaks
most 5.x ports does not apply here.

### 9.4 The thinking-mode trap

**Qwen3.8 runs in thinking mode by default**, emitting `<think>…</think>` before the
answer. For one line of handwriting that is pure cost, and it collides directly with
#92 — the generation budget that must scale with granularity. It has to be switched off
at the four `apply_chat_template` call sites (`train_qlora.py` 123/125/150,
`evaluate_qlora.py` 128) with `enable_thinking: False`, and a smoke run has to **confirm
no `<think>` token reaches the reference text**. A thinking model that reasons its way
to a transcription would also make CER incomparable with every number we have.

### 9.5 Recommendation

**Target Qwen3.5-9B, not the 27B.** It is the like-for-like replacement for
Qwen3-VL-8B, so a CER against the same eval set means something next to 0.466; it fits
**bf16 on one 96 GB H100**, which drops NF4 and with it the misaligned-bitsandbytes
penalty documented in the 2026-08-08 run; and it is ~3× cheaper per sample than a 27B.
At corpus scale that is the difference between a feasible campaign and a 60-day one.

Keep **Qwen3.8-27B for a subsample probe only** — a quality ceiling on ~50 K lines, not
the corpus run. And note the MoE option honestly: **Qwen3.5-35B-A3B** trains at roughly
a 3 B dense cost for 35 B of capacity, which is tempting, but nothing we operate can
serve it (`VLM_TRAINING.md` already records this for the 30B MoE).

### 9.6 Order of work

| # | step | GPU cost | gate |
|---|---|---|---|
| A | **NVMe vs share throughput**, current model, current container | ~4 h, free | decides whether the corpus run is 7 or 20 days |
| B | Rebuild the container on **transformers 5.x**; re-run the Thun smoke on `qwen3_vl` | ~10 min, free | **CER must land at 0.466 again** — proves 5.x changed nothing |
| C | Same smoke on **Qwen3.5-9B**, thinking disabled, `<think>` asserted absent | ~15 min, free | first honest old-vs-new number |
| D | Corpus chosen by `scripts/plan_corpus.py`, **not by hand** (§2 of the runbook: 21 hand-picked projects yielded 291 usable pages) | — | a submittable request |
| E | Subsample run, 4× H100 preemptable, checkpoint-and-requeue proven first | ~90 GPU-h, free | the model we ship |

B is the one people skip. Changing the base model and the library in the same step
means a worse CER has two possible causes and no way to tell them apart — which is the
mistake `TRAINING_PLAN.md` §9 exists to prevent.

---

## 11. With CHF 1000/year free: what money can and cannot buy

The F2 free tier refunds up to **CHF 1000 per cost centre per year**, and
activating a PAYGO project is straightforward. So what does that change?

**First, a naming correction.** UBELIX's *Investment Model* — CAPEX over 12–60
months for a pseudo-exclusive allocation — is **closed for GPUs until the 2026
datacenter expansion** (§4.3). It is not an option for this campaign whatever the
budget. What CHF 1000 activates is **PAYGO**, billed per minute against the
refund.

### The arithmetic

The measured campaign — 3 epochs over ~10.4 M crops at 17.76 samples/s — is
**488 H100-GPU-hours**.

| route | GPUs | wall clock | cost |
|---|---:|---:|---:|
| **free `job_gpu_preemptable`** | **4** | **6.0 days** | **0** |
| free `job_gratis` | 1 | 20.3 days | 0 |
| PAYGO `job_gpu` | 1 | 20.3 days | **CHF 293** |

* CHF 1000 buys **1,667 H100-hours/year**.
* The entire campaign is **488 h — 29 % of the allowance**, leaving ~1,179 hours:
  roughly **two and a half more full campaigns**, or the Qwen3.5 comparison, or
  re-runs after a mistake.

### The conclusion is counter-intuitive and worth stating plainly

**Money cannot make this faster.** PAYGO's `job_gpu` QoS caps at **h100=1**,
against the free preemptable QoS's **h100=4** (§4.3). Paying moves the campaign
from 6 days to 20. The CHF 293 buys *reliability* — no preemption — at 3.4× the
calendar time.

So the budget is not the constraint, and never was. **The constraints are the QoS
GPU ceiling and queue depth**, neither of which responds to money.

### What to do anyway

1. **Activate the PAYGO project.** It costs nothing: preemptable and debug jobs
   are free *even under a project*, so the primary path is unaffected. What it
   buys is optionality — a billed, non-preemptable fallback becomes a four-line
   header change instead of a procurement.
2. **Keep the free 4× H100 preemptable path as primary.** 6 days, and now
   genuinely safe to interrupt (§9.4).
3. **Use the paid path as a reliability fallback**, not a speed one: when
   preemption is thrashing a run, or a deadline makes 20 predictable days better
   than 6 unpredictable ones.
4. **Spend the headroom on questions, not on the same run.** ~1,179 hours is
   enough to answer whether Qwen3.5-9B beats Qwen3-VL-4B on real material, which
   is worth more than finishing this campaign three days sooner.

### The cheaper win is not money

`gnode36` is **8× H200, 141 GB each, and was sitting completely idle** while all
40 H100s were allocated — and our QoS has `h200=0` (§9.3-I). One support request
to add H200 to the QoS is free and would plausibly beat anything on this page:
experiment F showed **batch size is the whole bottleneck**, and at `bs: 16` we are
already at 86 GB of the H100's 94. A 141 GB card is the one piece of hardware here
that could take the next step up.

---

## 12. The German medieval run (`qwen3vl-medieval-german-v1`)

Started 2026-09-10, job **14709381**, spec `ubelix/specs/german-medieval.json`.

### The corpus is not the one this document has been measuring

`scripts/plan_corpus.py` was given `--period 1300 1600` and it **excluded
`image-text_medieval-scripts_xiv-xv-xvi` outright**:

```
image-text_medieval-scripts_xiv-xv-xvi   no target language in ('flemish',)
```

That is the dataset every experiment A–F ran on, and it is **Flemish**. Its German
content is ~291 usable pages. The German 14th–16th-century material lives in
other dh-unibe repos, and the planner selected four:

| repo | train projects | pages |
|---|---:|---:|
| `image-text_rats-und-richtebuecher_xv-xvi` | 35 | 9,885 |
| `image-text_bullinger-autoren` | 256 | 8,022 |
| `image-text_koenigsfelden-charters-post-1500` | 1,185 | 3,222 |
| `image-text_aaeb-xiv-xvii` | 349 | 2,566 |
| **total** | **1,825** | **~23,700** |

Held out for evaluation, on both `--eval-project` and `--exclude-project` as
`TRAINING.md` requires: `escript_test`, `escript_test_2`.

**So the schedule in §9.3-H does not apply to this run.** ~23,700 pages at the
measured 19 lines/page is **~450 K crops**, not 10.4 M — **23× smaller**. One
epoch is roughly **7 hours on a single H100**, which fits inside one preemptable
allocation.

### Configuration — every value measured, not chosen

| setting | value | why |
|---|---|---|
| `base_model` | Qwen3-VL-4B | §9.3-B: matches 8B at half the size |
| `load_in_4bit` | **false** | §9.3-A: bf16 is +23 % on a card we own |
| `batch_size` | **16** | §9.3-G: the whole bottleneck; 51 % → 86 % utilization |
| `max_pixels` | 262144 (256 tokens) | §9.3-D: 128 ties, 512 is *worse* |
| `workers` | 8 | §9.3-F: 4/8/16 indistinguishable |
| `optim` | `paged_adamw_8bit` | costs 2 %, insurance at 86 GB of 94 |
| `save_steps` | 200 | ~11 min of redone work if preempted |
| `epochs` | 1 | the runbook's rule at corpus scale |

### A correction this run forces: there is no multi-GPU

`train_qlora.py` pins `device_map={"": 0}` and the runner launches a plain
`python -m`, not `torchrun`. **DDP is not implemented.** Every measurement in
§9 is single-GPU, and so is this run.

Earlier sections quoted "4× H100 → 6 days" by dividing single-GPU throughput by
four. **That was arithmetic, not a measurement, and the code cannot do it today.**
The honest figures for the full Flemish set are the single-GPU ones: ~488
GPU-hours, i.e. ~20 days, not 6. Multi-GPU is a real piece of work — the trainer
would have to be launched under `torchrun` and the `device_map` pin removed — and
it is worth doing before anything at 10 M-crop scale, but it is not done.

For **this** run it does not matter: 450 K crops on one GPU is hours.

### Known risks going in

* **`429` during prepare.** `datasets` makes one hub tree call per project glob,
  and `koenigsfelden-charters-post-1500` contributes **1,185** of them — the exact
  number the runbook names as still costing 1,185 requests against a quota of
  1,000 per five minutes (#89). This is the most likely way the run fails, and it
  fails in `prepare`, cheaply.
* **These corpora are not cached on the share**, unlike the Flemish set, so
  prepare must fetch rather than read locally.

### What actually happened

**Prepare, attempt 1 (14710443): the predicted 429.** It died 28 minutes in on
`koenigsfelden-charters-post-1500` — rate-limited as an anonymous IP. Fixed two
ways: an **HF token** read from a private `~/.hf_token` (not `$HF_HOME/token`,
which is the group-readable share), and **`all_projects: true`** on that repo,
which makes the selection repo-complete so `collapse_complete_selection` emits one
glob instead of 1,185. The planner had deduplicated against three other
Königsfelden repos — all of which it then dropped from the corpus — so taking the
whole repo adds no duplicates here.

**Prepare, attempt 3 (14715226): completed**, 2 h 07 on a free CPU node, **zero
429s**, left in `training` by `--stop-after compile`.

| repo | pages | lines | note |
|---|---:|---:|---|
| `rats-und-richtebuecher_xv-xvi` | 3,902 | 139,114 | 35.7 lines/page |
| `bullinger-autoren` | 3,638 | 73,502 | 1,128 pages had no transcription |
| `koenigsfelden-charters-post-1500` | 2,679 | 50,024 | **3,057 over-wide lines dropped (6.1 %)** |
| `aaeb-xiv-xvii` | 2,068 | 62,534 | |
| **compiled** | | **306,582 train / 19,069 val** | 325,651 crops, 28 GB |

So the corpus is **~326 K crops**, below the ~450 K first estimated — the
estimate did not allow for pages without transcriptions or the geometry guard.
One epoch at the measured throughput is **~4.8 h** on one H100.

**Read the validation set carefully.** Only **594** of the 19,069 val lines come
from the held-out `escript_test`/`escript_test_2`. The rest are the 10 %
`partition` split of the three training repos: disjoint *pages*, but the *same
projects and hands* the model trains on. A CER from this job is therefore
**mostly an in-domain number**, as `VLM_TRAINING.md` §8 warns, and must not be
reported as a held-out result. The honest held-out figure needs the 594-line
subset scored on its own.

## 13. The Qwen3.5 comparison — four arms in parallel

### The transformers 5.x gate

Before any Qwen3.5 run, the **old** model under the **new** library, on the Thun
pair that has reproduced CER 0.466 twice. It took three attempts, and the two
failures are the reason the step exists:

| attempt | result | cause |
|---|---|---|
| 14717192 | **refused** | `size` became a `SizeDict` object in 5.x; the #86 budget guard found no dict and **refused rather than train at 16,384 visual tokens**. Fixed with an attribute-style branch. |
| 14718208 | failed | `TrainingArguments` no longer accepts `warmup_ratio`. Checked all 27 kwargs against the 5.17 signature: it is the **only** one removed. Converted to `warmup_steps`. |
| **14719049** | **CER 0.4720** | vs 0.4662 under 4.57.6 — **passes** the ±0.02 gate set in advance |

The pass is not bit-identity: 5.x moves this model by **+0.006** on 25 samples.
So a Qwen3.5-vs-Qwen3-VL difference smaller than ~0.01 should not be claimed as a
model effect. Either break, met with a new model in the frame, would have been
attributed to that model — and the first would not have crashed at all, only
trained the wrong thing.

### The thinking-mode trap

The Qwen3.5 templates **disagree about their default**. 0.8B and 2B treat an unset
`enable_thinking` as *off*; **4B inverts the test and treats it as on**, opening
`<think>` before every answer. Left alone, the 4B arm would have produced a
reasoning trace in front of each line transcription — not a worse CER but a
meaningless one, and the kind that reads as "4B is worse". `CHAT_TEMPLATE_KWARGS
= {"enable_thinking": False}` now goes to all four `apply_chat_template` calls.

Verified empirically for all four base models through the trainer's own code
path: no open `<think>` at generation, the assistant header is found inside the
training text so the loss mask starts at the answer, and the masked target is
exactly the transcription.

### The arms

`ubelix/fanout.py` cloned the prepared job once per base model. Each clone's
`data/` is its **own** directory of symlinks to the shared inputs — not a symlink
to the whole `data/`, because the test stage writes `data/eval_report.json` and
four arms would read back whichever CER landed last. **One corpus, one seeded
split, one library, four models.**

| job | base | params | GPU |
|---|---|---:|---|
| 14764645 | Qwen3-VL-4B (anchor) | 4 B | H100 |
| 14764646 | Qwen3.5-4B | 4 B | H100 |
| 14764647 | Qwen3.5-2B | 2 B | H100 |
| 14764648 | Qwen3.5-0.8B | 0.8 B | **A100** |

The 0.8B went to the A100 because all 40 H100s were allocated with 20 requests
queued, and it is the only arm that fits 80 GB at `bs: 16`. Its CER is comparable
to the others; its throughput is not.

**The 0.8B confirmed the whole stack on real hardware** within minutes: the 5.x
budget, the resume path through a fan-out clone, 25.6 M trainable of 878.5 M
parameters, and **~18 samples/s** — the same as Qwen3-VL-4B on an H100 despite
being five times smaller, the same "the language model is not the bottleneck"
signature experiment C found.

### Results (three of four arms)

Same corpus, same seeded split, same val set, same library (transformers 5.x).
All CERs over 200 validation samples.

| model | params | **CER** | WER | length_ratio | first score |
|---|---:|---:|---:|---:|---:|
| **Qwen3-VL-4B** (anchor) | 4 B | **0.532** | — | 0.528 | — |
| Qwen3.5-2B | 2 B | 0.588 | 0.681 | 0.485 | ~~6.158~~ |
| Qwen3.5-0.8B | 0.8 B | 0.698 | 0.775 | 0.371 | ~~6.984~~ |
| Qwen3.5-4B | 4 B | *training* | | | |

**Within Qwen3.5 the trend is clean and monotonic**: smaller is worse, and — more
interesting — smaller **under-transcribes more** (`length_ratio` 0.485 → 0.371).
The smaller models end the turn early on lines they cannot finish.

**Qwen3.5-2B does not match Qwen3-VL-4B** on this corpus: 0.588 vs 0.532, a gap of
0.056 against the ~0.01 the library alone can move a number. Whether that is the
generation or the halving of parameters is what the Qwen3.5-4B arm will separate.

### The third Qwen3.5 trap: no stop token

Both Qwen3.5 arms first scored CER **6.16** and **6.98** — not bad models, broken
scoring. The predictions showed it:

```
REF 'Judicatum est'   →  HYP 'Judicatum est\n\n\n\nJudicatum est\n\n\nJudicatum est …'
```

Qwen3-VL ships a `generation_config.json` with `eos_token_id = [151645, 151643]`,
so `generate()` stopped at `<|im_end|>` without being told. **Qwen3.5 ships no
generation config at all**, and `evaluate_qlora` passed no stop token — so every
line ran to the 256-token cap even though the model had learned to end its turn
(the blank lines are the special tokens stripped on decode). The report's own
length-controlled metric had already said so: `truncated_cer` 0.48 against
`length_ratio` 6.75.

Fixed by looking the end-of-turn tokens up **by name** in each model's tokenizer —
the families share no ids (`<|im_end|>` is 151645 in Qwen3-VL, 248046 in Qwen3.5) —
and passing them to `generate()` explicitly. **The adapters were fine**, so
`ubelix/rescore.sbatch` re-scored them in ~2 minutes each rather than retraining
for hours, writing `eval_report_v2.json` beside the original so the broken
numbers stay on disk.

Three Qwen3.5-specific traps in a row — the `SizeDict` budget, the inverted
thinking default, the missing stop token — none of which Qwen3-VL has. The gate
caught the first; the other two surfaced only with a Qwen3.5 model in the loop.

### Two cautions before quoting any of this

* **`length_ratio` ≈ 0.5 for every arm, anchor included.** The models transcribe
  about half the reference characters on this corpus, so these CERs are dominated
  by missing text. Because the anchor does it too, it is not caused by the
  stop-token fix or by Qwen3.5, and the *comparison* is fair — but the absolute
  numbers need that explanation, and its cause (long lines cut short? crops whose
  reference exceeds what is visible?) has not been established.
* **Mostly in-domain.** Only 594 of 19,069 validation lines are the held-out
  `escript_test` projects; the rest share hands with training. The held-out subset
  still needs scoring on its own.

---

## 14. The German 19th-century run (`qwen3vl-german-xix-v1`)

Queued 2026-09-11 as one dependency chain: prepare **14797054** (CPU) →
`fanout_submit` **14797055** (after prepare succeeds) → three H100 arms
(Qwen3-VL-4B anchor, Qwen3.5-4B, Qwen3.5-2B). The 0.8B is left out: it was the
clear loser on the medieval corpus (§13).

### The planner would have trained on machine output

Asked for 1800–1900, `plan_corpus.py` put
**`handwritten-bundesratsprotokolle_xix-xx` at 45 % of the corpus**, scoring it
0.95. Its own card:

```
--- Data has been automatically created, using ATR models ---
These are ''automatically'' transcribed pages.
!!!This data set does not contain Ground Truth!!!
```

The scorer read only period, language and script, so a good card for bad data
scored well. Training on it teaches a model another model's errors; scoring
against it measures agreement with that model. **Fixed in the planner** (`253cc7d`):
a card that declares machine output is vetoed outright, outside the geometric mean
where no weighting can outvote it. The detector is deliberately specific — every
pagexml-hf card also says "the Hub automatically merges all parquet files", so a
bare match on "automatically" would have vetoed the whole org.

**It caught a second one on the next run**: `historisches-grundbuch-basel_xix-xx`,
the Basel land register, rejected for the same reason.

### Two things the cards could not say

* **`kurrent-xix` scored 0.5399**, below threshold only because its card is empty
  (unknown metadata is penalised to 0.5, by design). It is the other large
  19th-century German ground-truth set, and without it the 50 % share cap collapsed
  the corpus to ~1,200 pages. Included by lowering the threshold to 0.53 and
  excluding the other low scorers by name with the new `--exclude-repo`.
* **The two big datasets overlap.** `zh-regierungsratsprotokolle` has 262 project
  directories including `MM_1_001…`, and `kurrent-xix` carries 12 `MM_` projects
  of the same names — the same Zurich volumes twice. The planner could not
  deduplicate them because zh's *card* lists no projects. `kurrent-xix`'s `MM_*`
  are excluded; zh keeps them.

### A real held-out benchmark this time

`kurrent-xix` ships the CITlab/READ split: 21 `TRAIN_CITlab_*` and 21 matched
`TEST_CITlab_*` projects. The `TEST_*` projects are held out on both
`--eval-project` and `--exclude-project` — **genuinely unseen hands**, unlike the
medieval run's validation set, which was 97 % partition split.

The validation set is still mixed, though: zh and the two small repos have no
eval projects, so they contribute a partition split. **The `TEST_CITlab` subset
must be scored on its own** to get the held-out number; that is now the second run
that needs it, and it should become a feature of the evaluator rather than a
one-off.

### Corpus

| repo | pages | note |
|---|---:|---|
| `zh-regierungsratsprotokolle` | 20,000 | capped from 152,786 at the 50 % share |
| `kurrent-xix` | 19,808 | 25 train projects; 21 `TEST_*` held out, 12 `MM_*` dropped as duplicates |
| `parlamentsdienste-protokolle` | 138 | |
| `nr-sr-vereinigte-bundesversammlung-xix` | 52 | |
| **total** | **39,998** | ~940 K lines *estimated* — the medieval estimate ran ~30 % high |

---

## 15. Moving trained models to asterAIx

**Transfer is trivial; serving is not.** UBELIX `/storage/research/wbkolleg_dh_1` and
asterAIx `/mnt/wbkolleg_dh_1` are the same storage, and asterAIx's convention is to
leave adapters on the share and point `local_path` at them. So the adapters went to
`Textrecognition_Training/trained-ubelix/`, with a `README.md` beside them.

| model | CER | on asterAIx |
|---|---:|---|
| `qwen3vl-medieval-german-v1` | 0.532 | registered, **merged (8.3 GB), verified in vLLM**, disabled pending the gate |
| `qwen3.5-2b-medieval-german-v1` | 0.588 | stored; **not servable** (vLLM 0.11 / transformers 4.57 has no `qwen3_5`) |
| `qwen3.5-0.8b-medieval-german-v1` | 0.698 | stored; not servable |

The Qwen3.5 adapters are deliberately **not** in the overlay: `merge_loras.py` with
no `--only` merges every vLLM LoRA it finds, and a `qwen3_5` entry would make it
fail for everyone.

### Four obstacles, in the order they appeared

1. **The merge failed halfway.** All four medieval arms ran under the transformers
   5.x image, so their adapter directories carry 5.x processor files, which
   asterAIx's 4.57 cannot parse (`'list' object has no attribute 'keys'`). The
   weights had merged; the base model's processor was saved instead — valid only
   because the adapter has no `modules_to_save` and adds no tokens, which was
   checked (vocab 151,936 ≥ tokenizer 151,669) rather than assumed.
2. **The gate used the wrong endpoint.** `/ocr` serves only kraken and TrOCR
   (`400: use /recognize for 'vllm'`). `training/promote.py`'s gate also posts to
   `/ocr`, so it can never pass for a VLM.
3. **The gateway could not load it.** Every serving unit pins
   `CUDA_VISIBLE_DEVICES=1`; the always-on engines hold ~14 GB there and the
   asterAIx run `qwen3vl-german-pages-v3` holds ~30 GB, so vLLM started with
   **0.59 GB free** and died. This blocks *every* gateway VLM while that training
   runs, not just this one. Verified the model independently on GPU 0 instead:

   ```
   REF 'hoc, nunc illud imparatus agere cogar. Ad eam relationem'
   HYP 'hoc, nunc illud impareatur, agere cogor. Ad eum relationem'
   REF 'mains, dans laquelle peu auparavant elle avoit tenu de son'
   HYP 'mais,'
   ```

   The second pair is the corpus-wide `length_ratio ≈ 0.5` caught in vLLM itself —
   the model genuinely ends its turn early; it is not an evaluator artefact.
4. **agentic_historian cannot use gateway VLMs as written.** `orchestrator.py`
   sends `engine == "vlm"` to GPUStack and everything else to `/ocr`. The client
   already has `recognize()` — added when `party` hit the same 400 — so routing
   gateway-vLLM picks through it is the whole fix.

Every enable/restart was reversed on failure; production ended each time at its
original 50 models with the gateway healthy.

### HF (private)

Three private repos under `dh-unibe` are planned and their cards verified — the
Qwen3.5 cards carry the corrected CER, the re-score explanation and the in-domain
caveat. **Blocked**: `~/.hf_token` on UBELIX held zero bytes, so no job had ever
been authenticated despite logging "present" (fixed in `cbb7858`).

---

## 16. Results of the 19th-century arms — and what they reveal about medieval

All three 19th-century arms finished 2026-09-12. The four medieval arms had
finished 2026-09-11. Every adapter is on the research share under
`Textrecognition_Training/trained-ubelix/`, with `metadata.json` (the full
`job.json`) beside it.

### The 19th-century numbers

| model | base | CER | WER | H100 time |
|---|---|---:|---:|---:|
| `qwen3vl-german-xix-v1` | Qwen3-VL-4B | **0.0100** | 0.0378 | 10 h 03 |
| `qwen3.5-4b-german-xix-v1` | Qwen3.5-4B | 0.0107 | 0.0435 | 15 h 46 |
| `qwen3.5-2b-german-xix-v1` | Qwen3.5-2B | 0.0141 | 0.0527 | 7 h 45 |

881,542 train / 85,206 validation lines, one epoch, 200 lines scored.
`length_ratio` 1.00 for all three.

Two readings, and the caveats belong with them. **"Smaller but newer" holds up**:
Qwen3.5-2B reaches 1.41 % on a 2B backbone in 7 ¾ hours. **Newer at the same size
does not pay here**: Qwen3.5-4B is within noise of Qwen3-VL-4B (63 vs 59 character
errors out of 5,898) and took 57 % longer. On 200 lines that difference is not
resolvable; the wall-clock difference is. Qwen3-VL-4B stays the anchor.

The 21 `TEST_CITlab_*` projects are still not scored separately — the validation
set mixes them with the partition split from `zh`. The 1.0 % is therefore a
*mostly in-domain* number, the same caveat as §14 flagged in advance.

### The medieval `length_ratio ≈ 0.5` is a data problem, and now it is identified

This had been open since §12 across every medieval run. The 19th-century results
solved it by contrast: identical code, prompt, evaluator and hardware, and
`length_ratio` 1.00 instead of 0.48. So it is not the trainer.

The medieval models stop after the first word:

```
REF (35): Hanns pfister Jacob slossers knecht     HYP (5): Hanns
REF (13): Judicatum est                           HYP (9): Judicatum
REF  (3): xvj                                     HYP (3): xvj
```

Output length over reference length, by how long the true line is — 1.70 at 1–3
chars, 0.67 at 4–15, **0.14 at 16–40**. The longer the real line, the less it
writes. `truncated_at_cap` is 0, so this is the model emitting end-of-turn, not
`max_new_tokens`.

The cause is the training set's line lengths:

| set | median line | ≤3 chars |
|---|---:|---:|
| medieval **train** | 12 chars | **20.9 %** |
| medieval **val** | 44 chars | 8.3 % |
| 19th c. train | 30 chars | 0.8 % |
| 19th c. val | 30 chars | 1.3 % |

A fifth of medieval training samples are 1–3 character fragments — folio numbers,
column figures, marginalia (`dat`, `16`, `B VI`, `190`). At one epoch the model
learns an aggressive end-of-turn from them and carries it onto the long lines it is
scored on. Train and val differ because the partition is by page, and the
fragment-heavy pages are not spread evenly.

**This means the four medieval CERs (0.53–0.70) measure the corpus, not the
models**, and the medieval/19th-century gap is not a statement about how hard
medieval script is. All of the medieval sweeps varied hyperparameters; none
touched this.

**Next medieval run:** filter short lines out of *training* (a minimum reference
length, or a cap on their share), leaving validation as it is so the numbers stay
comparable. That is one prepare-stage change, and it should be tried before any
further hyperparameter work on this corpus.

---

## 17. The medieval CER was a parser bug, not a model or a corpus

`min_train_chars` (§16) was a fix for a symptom. It helped — CER 0.5322 → 0.4995,
`length_ratio` 0.479 → 0.586, one variable changed against the same seed, the same
split and the same `val.jsonl` by symlink — but nothing like the amount §16
predicted. Chasing the remainder found the actual cause, and it was neither the
model nor the data.

### What the validation images showed

The first four validation examples split into two unrelated failures:

* `REF "a"` → the model wrote `berlin`; `REF "i"` → it wrote `der Statt`. Those
  crops are not lines at all but near-square blocks holding fragments of three
  lines, and the model read them correctly. Bad crops, bad references.
* `REF "Hanns pfister Jacob slossers knecht"` → the model wrote `Hanns`. That crop
  is a **clean, full-width line**, completely legible, with a correct reference.

The second kind dominates the error (3,587 of 8,266 characters missing), and it
cannot be blamed on crop geometry.

### The measurement that pointed at the references

Sampling crop aspect ratios against reference length: medieval samples with 4-9
character references sit on crops of median aspect **7.24 : 1** — a full-width
line that should hold ~30 characters. The 19th-century corpus's 4-9 character
references sit on crops of **1.63 : 1**, genuinely small images. Restricted to
proper line crops (aspect ≥ 5:1), **36.3 %** of medieval training samples carry
less than a third of the median transcription density.

Wide line, almost no text. So: truncated transcriptions on intact images.

### The cause, in our own code

Transkribus exports word segmentation as `<Word>` children, each with its own
`TextEquiv/Unicode`, and those come **before** the line's own `TextEquiv` in
document order. Taking the first `Unicode` under a `TextLine` therefore returns
word 1 and drops the rest of the line.

On one medieval page, `000002_1610663_0114_60532792.xml`:

```
OLD 'un'      -> NEW 'un Bud du xv ß iij der'
OLD 'hützer'  -> NEW 'hützer artz hers wip Eid xvj ß'
OLD 'tor'     -> NEW 'tor Basmatter von wungen der altt Eid ij lb'
page total: 246 chars -> 1,760 chars (7.2x)
```

`'hützer'` and `'tor'` are exactly the references seen on the wide crops above.

### The control that settles it

60 random pages from each corpus, old parser against new:

| corpus | old | new | ratio |
|---|---:|---:|---:|
| medieval | 50,673 chars | 74,408 chars | **1.47×** |
| 19th century | 62,913 chars | 62,913 chars | **1.00×** |

> **Corrected in §21.** The next paragraph is wrong. The 60-page sample was
> dominated by `zh-regierungsratsprotokolle` (931,173 of ~1.19 M lines) and never
> reached the two small federal-protocol sources, which *were* truncated — 59.7 %
> and 30.2 % of their lines. The 1.00 % CER was not evidence of anything: it was
> measured on five pages from the unaffected source.

The 19th-century corpus has no word-level `TextEquiv` and was never affected —
which is precisely why it reached 1.00 % CER on identical code, and why the
medieval/19th-century gap was never about the difficulty of medieval script.

The medieval corpus lost roughly a third of its characters, concentrated in the
lines that carry word segmentation. A model trained on that learns to write the
first word and stop, which is the behaviour every medieval run has shown.

### Status

Fixed in `33f55fc` ("A transcribed line is the line, not its first word", #125)
with `24ec2c5` bumping the artefact cache key so nothing compiled before it is
reused. Both landed 2026-09-14, after the medieval corpus was prepared on
2026-09-10 — so **every medieval run so far, v2 included, trained and scored on
truncated references**. The four published medieval CERs measure this bug.

**What follows:** the medieval corpus must be prepared again from scratch — the
clone trick in §16 cannot help, because the damage is in the compiled JSONL.
`min_train_chars` should go back to 0 for that run until the fixed corpus has been
measured: it was built against a distribution that no longer exists, and the short
lines it removes may be legitimate once the references are whole. It stays in the
codebase as an option, defaulting to off.

---

## 18. `qwen3vl-medieval-german-v3` — what the parser was costing

Corpus prepared again from scratch with the fixed parser (§17), `min_train_chars`
back to 0, everything else identical to v1: same four repositories, same seed,
same 0.9 page-level partition.

### The corpus

Same 12,301 pages, same 325,768 lines, same 325,651 samples — **four million more
characters**. Nothing was added; what had been thrown away came back.

| repo | v1 chars/line | v3 chars/line | gain |
|---|---:|---:|---:|
| `rats-und-richtebuecher_xv-xvi` | 12.7 | 34.5 | **2.71×** |
| `aaeb-xiv-xvii` | 24.7 | 36.6 | **1.49×** |
| `bullinger-autoren` | 49.9 | 52.6 | 1.05× |
| `koenigsfelden-charters-post-1500` | 76.4 | 77.8 | 1.02× |
| **total** | **33.2** | **45.7** | **1.38×** |

Concentrated exactly where word-level `TextEquiv` exists. `aaeb-xiv-xvii` was also
affected, which §17 did not predict — only the Zurich books were on the list.

Training-set line length, median **7 → 42 characters**; samples of ≤3 characters,
**25.6 % → 4.1 %**. The 19th-century corpus sits at median 30 and 0.8 %. So the
"medieval corpus is full of fragments" reading in §16 was wrong: four fifths of
those fragments were truncated transcriptions, and `min_train_chars` was a filter
against an artefact of our own parser.

### The result

| | v1 | v2 (`min_train_chars=4`) | **v3 (parser fixed)** |
|---|---:|---:|---:|
| CER | 0.5322 | 0.4995 | **0.1120** |
| WER | 0.6199 | 0.6046 | **0.2787** |
| `length_ratio` | 0.479 | 0.586 | **1.0012** |
| missing chars | 4,389 | 3,587 | **185** |
| substitutions | 325 | 380 | 546 |

4.75× better than v1, in 4 h 48 on one H100, no preemption.

**The comparison is exact.** The first 200 validation lines are byte-identical
between v1 and v3 — they come from pages with no word-level `TextEquiv` and were
never affected — so all three CERs are measured against the same reference
strings. The gain is entirely in the training data.

The error profile inverted, which is the part that matters: v1's errors were
missing text (4,389 of 4,799), v3's are substitutions (546 of 926). Misreadings
instead of truncation — the profile an HTR model is supposed to have.

```
REF: Hanns pfister Jacob slossers knecht
v1:  Hanns
v3:  Hanns Pfister Jacob glogsters knecht
```

### Still open

The near-square crops remain: blocks holding fragments of three lines, carrying a
one-character reference (`REF "a"`, `REF "i"`). 7.9 % of validation by aspect
ratio, and now a measurable share of the remaining 0.112, because they are simply
wrong ground truth. That is the next lever — but measure how much it actually
costs before building a filter for it. §16 is the cautionary tale.

Published privately as `dh-unibe/qwen3vl-medieval-german-v3` (544 MB, 15 files)
and copied to the share.

---

## 19. The block crops are not worth filtering — measured, not assumed

§18 named the near-square crops as the next lever. They are not. Measuring first
was the point of §16's lesson, and this time the measurement came back negative.

### Method

No GPU. The evaluator already writes `eval_report.raw.jsonl` — every prediction,
not the ten kept for eyeballing — so the 200 scored samples were re-joined to
their references and their crop dimensions, and re-scored with `score_pairs`, the
evaluator's own scorer. The decomposition reproduces the reported **CER 0.1120**
on the full set exactly, which is what says the right 200 samples were recovered.

### Result

| subset (aspect = width / height) | n | ref chars | errors | CER |
|---|---:|---:|---:|---:|
| all | 200 | 8,266 | 926 | **0.1120** |
| block-like, w/h < 3 | 32 | 154 | 60 | 0.3896 |
| proper lines, w/h ≥ 3 | 168 | 8,112 | 866 | **0.1068** |

The block crops are read badly — CER 0.39 against 0.11 — and that is what made
them look worth fixing. But they hold **1.9 % of the reference characters** while
carrying 6.5 % of the errors. Removing them entirely moves the CER from 0.1120 to
**0.1068**: a gain of 0.005, under 5 % relative.

At the most generous threshold (w/h < 4, 40 samples) it is 0.1054 — still under
6 %.

**That 6.5 % is a ceiling, not an estimate.** Even a perfect fix — correct
geometry and correct ground truth for every block crop — cannot recover more than
the errors they carry, and the small subset's own CER being noisy does not change
it. So the ceiling holds however the 32 samples fall.

### What this means

The remaining error is on **proper line crops**: 866 errors over 8,112 characters,
CER 0.1068. That is ordinary handwriting difficulty in 14th–16th century German,
not a data defect with a cheap fix. Further gains have to come from training —
more epochs, more corpus, a larger base — or from ground truth that is wrong in
ways this measurement does not see.

A crop-geometry filter would be real work for under 5 %, and it would also delete
whatever legitimate short lines fall under the threshold. Not worth it. The
finding is recorded so the idea does not get proposed again from the same
plausible-sounding reasoning that produced §16.

---

## 20. What the reported CERs were actually measured on

Scoring the held-out subset separately (open since §14) turned up something
larger: **until `4785410` the test stage scored `val.jsonl[:eval_samples]` — the
head of the file** — and `_prepare_multi` writes that file one dataset after
another. So every CER in §12–§19 describes whichever material happens to sit at
the front of its corpus, not the corpus.

The fix landed 2026-09-15 21:37. The v3 training job started 00:02 the next
morning, but the UBELIX checkout was last pulled at 07:38 the previous day — the
fix was committed and not deployed. Worth remembering: on UBELIX the container
takes the code from `~/serving-atr-inference` via `PYTHONPATH`, so a job runs
whatever that checkout was when it started, not what is on `main`.

It cuts both ways.

**Medieval v3 was better than reported.** The first 594 lines of its `val.jsonl`
are `escript_test` and `escript_test_2` — the held-out projects — so all 200
scored samples fell inside them, where a random draw would have given about six.
The 0.1120 was already a held-out number.

**The 19th-century runs are weaker than reported.** Their 200 scored lines come
from **five pages**, every one of them from a document that also appears in
training. The 1.00 % is one source, five pages, same hands.

### v3 measured properly, on two disjoint subsets

| subset | n | ref chars | CER | WER | ratio |
|---|---:|---:|---:|---:|---:|
| held-out (`escript_test`, all 594 lines) | 594 | 24,543 | **0.1109** | 0.2747 | 1.001 |
| in-domain (seeded 200 of 18,475) | 200 | 11,629 | **0.1427** | 0.3601 | 0.989 |

The held-out number rests on all 594 lines rather than 200, and lands at 0.1109
against the 0.1120 the 200 gave — so that sample was representative.

**The held-out set scores *better* than the in-domain set, and the reason is the
source mix, not the split.** `escript_test` is Rats- und Richtebücher material,
of which the model saw 139,708 lines; it is an unseen *project* in a very
familiar hand. The in-domain draw is spread over all four repositories, including
the harder ones. Per source, on that draw:

| source | n | CER |
|---|---:|---:|
| `aaeb-xiv-xvii` | 51 | 0.0979 |
| `bullinger-autoren` | 66 | 0.1301 |
| `koenigsfelden-charters-post-1500` | 27 | 0.1574 |
| unattributed (mostly rats-und-richtebuecher) | 56 | 0.1695 |

So **0.14 is the number that describes the model on this corpus**, and 0.11 the
number on one familiar-looking held-out project. Neither is wrong; quoting only
the second would be.

### A gap in the #120 fix

`plan_eval_subset` attributes a page by the pool index in its **image** name. At
`granularity: line` the image is a crop, whose index carries no source, so
attribution finds nothing and the planner falls back to an unstratified draw —
which is what it did here (`0 source(s) could be attributed`). The `page` field on
each row does carry the index, and reading that instead makes it work: the table
above was produced by calling `attribute` on `page` rather than `image`. Every
line-granularity run is affected, which is all of the medieval and 19th-century
work.

Attribution reached 119 of 200 pages, the rest falling in the index ranges that
skipped pages make ambiguous — the module's docstring explains why it declines to
guess, and leaving them as "unattributed" is the right behaviour.

---

## 21. The 19th-century models on a published held-out benchmark

§20 found that the 19th-century CERs (1.00 / 1.07 / 1.41 %) came from five pages
of documents that also appear in training. This section is the number that
replaces them.

### The test set

*Handwritten Text Recognition Test Set: Minutes of the Swiss Federal Council
(1848-1903)*, Hodel & Schoch 2021, https://doi.org/10.5281/zenodo.4746342,
CC BY 4.0. 2,751 ground-truth line images drawn at random from ~150,000 pages of
BAR E1004.1#1000/9#1-215. Every page carries `status="GT"`.

Mirrored privately as `dh-unibe/image-text_federal-minutes-testset` in
`pagexml-hf` shape (one row per line, `project_name` `TEST_federal_minutes`), with
the text read by `_own_text`. The card cites the Zenodo record.

**Zero document overlap**: its 111 `docId`s checked against all 115 `docId`s on
the train and validation sides of `qwen3vl-german-xix-v1`. Same period (1848–1903)
and script as the training corpus, different documents and hands — the clean case,
where a drop cannot be blamed on a change of era.

Lines were cut with the service's own `samples_for` + `write_crops`, so they are
shaped the way the models were trained. 2,751 lines, 114,960 characters, the same
count the dataset build produced.

### Result, all 2,751 lines

| model | CER | WER | ratio | collapsed | CER on the rest |
|---|---:|---:|---:|---:|---:|
| `qwen3vl-german-xix-v1` | **0.2551** | 0.3917 | 0.815 | 18.4 % | 0.0921 |
| `qwen3.5-2b-german-xix-v1` | 0.2937 | 0.4223 | 0.771 | 22.2 % | 0.0988 |
| `qwen3.5-4b-german-xix-v1` | 0.3596 | 0.4733 | 0.685 | 29.3 % | 0.0884 |

"Collapsed" = the prediction is under a third of the reference.

A factor of 25–35 against the reported figures. Two things make up the gap:

* **Reading.** On the lines each model does transcribe, CER is 0.09–0.10. That is
  the held-out reading quality of these models on unseen 19th-century hands.
* **Stopping.** On a fifth to a third of lines they write the first word and
  stop. The crops are clean — one checked by eye is a complete, legible 54-char
  line (`Gerichten auf den Prozeß einlassen wolle. Auf Anregen,` → `Zürchen`).

The collapse is **not** explained by line length: lines over 70 characters
collapse *less* (7 %) than 45–55 (28 %). It is tied to the images: **289 lines
collapse in all three models, against 33 if collapse were independent**, and on
those lines all three write the same first word — `verwaltung.`, `ner`, `seiner`,
`dem`, `Protokoll`.

### The 19th-century corpus was affected by the parser bug after all

That signature is the §17 bug's. So the old and fixed readers were compared over
**every** page of the 19th-century corpus rather than a sample:

| source | lines | truncated | char gain |
|---|---:|---:|---:|
| `nr-sr-vereinigte-bundesversammlung-xix` | 7,606 | **59.7 %** | **3.00×** |
| `parlamentsdienste-protokolle` | 5,100 | **30.2 %** | **1.74×** |
| `kurrent-xix` | 127,957 | 1.1 % | 1.02× |
| `zh-regierungsratsprotokolle` | 931,173 | 0.0 % | 1.00× |

About 7,500 truncated lines, some 0.6 % of the corpus — and concentrated in the
two **federal-protocol** sources, the material closest to a Federal Council test
set. That fits a model that learned "on this kind of page, write the first word
and stop", and never saw the behaviour contradicted because the cantonal Zurich
volumes, nearly 80 % of the data, look different and were not truncated.

It fits; it is not proven. The proof is a retrained model: re-prepare the
19th-century corpus with the fixed reader and score it on this set. Nothing in
this section justifies predicting the number that run will produce.

### Lesson, again

§17's "1.00× — never affected" came from 60 random pages of a corpus in which one
source holds 78 % of the lines. A uniform sample of a skewed corpus measures its
largest source. Per-source checks over the whole corpus cost minutes; the wrong
conclusion cost three published model cards.

The three cards now carry the held-out result, name the benchmark by DOI, state
that the headline CER above the note is not held-out, and record the training-data
defect.

### 21a. A 16-hour prepare, then a chain that died in one second

`qwen3vl-german-xix-v2` was queued as prepare → chain → train → score. The
prepare took **16 h 34** instead of v1's 4 h: compile finished in 2 h 35, and the
remaining 14 h were the artefact cache (#109) copying 966,748 crop files — 87.5 GB
— one by one onto GPFS at ~57,000 files an hour. That step did not exist when v1
ran. It fitted the 20 h walltime, but not by much; a larger corpus will not.

The chain then failed in one second: `MaxCpuRunMinsPerUser`. `job_gratis` caps
**CPUs × walltime at 11,520 minutes**, and that applies to GPU jobs as much as CPU
ones. The train job asked for 16 × 24 h = 23,040 and was rejected at submission.
The v3 medieval run had fitted only because it asked for 16 × 8 h.

Resubmitted as 12 CPUs × 15 h = 10,800. The chain script now uses the same, with
the reason next to it, and lives in the repo as `ubelix/chain_train_score.sbatch`
beside `ubelix/score_federal_minutes.sbatch`.

### 21b. Seven seconds: a stale checkout, the second time

A note on names first: #136 established that the serving box at 130.92.59.240 is
**idhefix**, not asterAIx; asterAIx is 130.92.59.242 and becomes the training
host. Sections up to §21a use the old name for .240. From here on it is idhefix.

Resubmitted after §21a, the training job `15449955` failed after **7 seconds**:

```
File ".../runner_base.py", line 1009, in _finish
    self._release_gpu()
File ".../gpu_release.py", line 55, in release_gpu
    import httpx  # trainer venv only
ModuleNotFoundError: No module named 'httpx'
```

`_finish` asked the idhefix gateway (`127.0.0.1:8200`) to free its GPU before
training (#129). UBELIX has no such gateway and the container has no `httpx`. `main`
had made that call non-fatal at 13:56 the day before (`02fd666`) and removed it
entirely at 22:08 (`4f3ab24`, #139) — but the UBELIX checkout had last been pulled
that morning, before the prepare was submitted, and the train job runs whatever the
checkout holds when it **starts**.

That is §20's failure again. The first time it cost a mis-scored evaluation; this
time a failed job and a manual repair.

**Repair.** `failed` is terminal in `jobstore.py` — nothing leaves it. No training
had happened (no checkpoint directory) and the compiled corpus was intact, so the
record was set back to `training` by hand, the failed `train` stage record dropped,
and the failed state kept beside it as `job.json.failed-15449955`. Resubmitted as
`15450030` (train, 12 CPUs × 15 h) and `15450031` (score, after it).

**Mitigation** (`b53d990`). `train.sbatch` and `prepare.sbatch` now log the commit
they run and fetch `origin/main`, printing a loud warning when the checkout is
behind. They do not pull: a job changing its own code while it waits in the queue
is worse than a visible warning. The warning is after-the-fact. What actually
prevents this is pulling on the login node **before every** `sbatch` — and job
records do not say which code produced them, which is the underlying gap (§22).

---

## 22. Open problems, and which of them need an issue

Everything that went wrong between §16 and §21b, sorted by whether it needs to be
tracked. The test for "needs an issue": the problem will recur without a code
change, or its consequences outlive this document (published models, other
people's numbers). An operational habit already fixed in a script does not.

| # | problem | existing | recommendation |
|---|---|---|---|
| A | parser bug also truncated the federal-protocol sources; models trained on them | #125 (open) | **comment on #125** — posted |
| B | artefact-cache store takes 14 h on GPFS | #109 (open) | **#148**, linked from #109 |
| C | #120's attribution missed line granularity | #120 (closed) | **comment on #120** — posted, not reopened |
| D | jobs run whatever the checkout holds; records carry no code commit | — | **#147** — part 1 done (`14087a8`; training-atr-models#18) |
| E | `job_gratis` CPU-minute cap applies to GPU jobs | — | no issue |
| F | the recorded CER is never a held-out number | — | **#146** (the most important) |
| G | first-word collapse on the Federal Council test set | — | wait for v2 |
| H | near-square block crops | — | no issue (§19) |
| I | no way out of `failed` after an environmental failure | — | no issue, for now |

### A — the 19th-century damage belongs in #125

*For tracking:* the consequences outlive the fix. Three published 19th-century
cards, the Qwen3.5 variants (#132) and the four medieval v1 adapters were trained
on truncated text; #125 today only knows about the medieval sources.
`nr-sr-vereinigte-bundesversammlung-xix` (6.35× the characters after the fix) and
`parlamentsdienste-protokolle` (4.54×) are missing from it, and so is the
downstream evidence: first-word collapse on a held-out benchmark.

*Against a new issue:* same root cause, same fix; a second issue splits the record.

*So:* a comment on #125 with the per-source table, the list of models trained
before `33f55fc`, and a closing criterion — #125 closes when each of those is
retrained or marked superseded.

### B — the store step of the artefact cache

*For:* measured, and it nearly failed a job. `_store_artefact` copied 966,748
crop files (87.5 GB) one at a time — 14 of the prepare's 16.5 hours, against a 20 h
walltime. A larger corpus will time out *after* its corpus is built, leaving the job
at `compiling`. It also doubles scratch use. The obvious fixes are cheap: move or
hard-link instead of copy when source and cache share a filesystem, or store a tar
of the crops, or an opt-out for runs that will never be reused.

*Against:* specific to network filesystems with many small files; on local disk the
same copy may be fast. And `xix-v2`'s corpus is unlikely to be reused, so the cache
bought nothing here.

*So:* a new, focused issue — it is a defect with a clear fix, and folded into #109
it would disappear when #109 is closed as implemented.

### C — traceability for #120

*For a comment:* anyone investigating an odd stratified draw will open #120. The
hole (crop names carry no source; `page` does) and its fix `e829028` should be
findable from there.

*Against reopening:* it is fixed and tested.

### D — a job does not know what code it ran

*For:* it has cost two runs (§20, §21b), and the mitigation only warns after the
fact. The deeper problem is reproducibility: `job.json` records the request, the
data and the metrics, but not the commit that produced them — so a CER cannot be
tied to the evaluator that measured it, which is exactly what §20 had to
reconstruct by hand. A fix: record `git rev-parse HEAD` at submission, carry it in
the job record, and run from that commit (a worktree per job) or refuse to start
when HEAD differs.

*Against:* one operator, and "pull before `sbatch`" is a habit, not a system.
Worktree-per-job adds moving parts on a shared home directory.

*So:* a new issue, framed as reproducibility rather than as UBELIX housekeeping.
Recording the commit is small and valuable on its own; pinning can follow.

### E — `job_gratis` and GPU jobs

*Against an issue:* one incident, rejected at submission (so it cost queue time,
not compute), documented in §21a and handled in `chain_train_score.sbatch`.

*For:* `submit.sh` could compute CPUs × walltime and refuse early. Worth a line in
D's issue if that one grows a "submission checks" section; not an issue of its own.

### F — the recorded CER is not a held-out number

*For:* this is the thread running through §16–§21. The number in `job.json`, on
the share and on every published card comes from the run's own validation split —
seeded partition, mostly the same documents as training — and for two campaigns it
was read as model quality. Getting the real number took a hand-built evaluation
directory, a one-off sbatch and hand-written notes on four cards. Now a published
benchmark exists in the right shape (`dh-unibe/image-text_federal-minutes-testset`,
doi:10.5281/zenodo.4746342). The test stage should take optional named benchmarks
(`params.benchmarks: [hf_repo, …]`), score them beside the split, record both, and
`publish.py` should put the benchmark first on the card and the split second,
labelled as such.

*Against:* benchmarks exist only for some periods (federal minutes for the 19th
century; `escript_test` for one medieval source), and scoring 2,751 extra lines
adds about an hour per run.

*So:* a new issue, and the most important of these — it removes the conditions
under which A, C and the medieval misreadings went unnoticed.

### G — the first-word collapse

*Wait.* `qwen3vl-german-xix-v2` is training on the repaired corpus. If its
collapse rate on the test set falls from v1's 18.4 % to near zero, this is A's
consequence and needs no issue of its own. If it does not, it is a new problem and
gets one — with the v2 numbers, not a hypothesis.

### H — block crops

*No.* Measured in §19: at most 6.5 % of the errors can be recovered, and a filter
would also drop legitimate short lines.

### I — leaving `failed`

*Against, for now:* `failed` being terminal is a deliberate invariant, and a
"retry" transition would make it easy to resume jobs that should not be resumed.
One manual repair, with the failed record kept beside it, is acceptable.

*For, later:* if environmental failures before any training become common (D would
make them rarer), a guarded `failed → training` for jobs with no checkpoint and an
intact corpus would replace hand-edited JSON.

---

## 23. `qwen3vl-german-xix-v2`: the retrained model settles §21

§21 said the first-word collapse *fit* the truncated federal-protocol sources but
was not proven by them, and that the proof would be a retrained model scored on
the same benchmark. It was retrained on the corpus rebuilt with the fixed reader
(#125) — same four repositories, same seed, same page split — and scored on the
same 2,751 lines of the Federal Council test set.

| | v1 | **v2** |
|---|---:|---:|
| CER | 0.2551 | **0.0765** |
| WER | 0.3917 | **0.2458** |
| `length_ratio` | 0.815 | **1.0019** |
| missing characters | 22,831 | **1,538** |
| **collapsed lines** (under a third of the reference) | **507 (18.4 %)** | **0 (0.0 %)** |

The CER is 3.3× better, but the last row is the finding: the collapse does not
shrink, it **disappears**. Nothing in the run addressed it except the corpus, so
the truncated transcriptions in `nr-sr-vereinigte-bundesversammlung-xix` (6.35×
the characters after the fix) and `parlamentsdienste-protokolle` (4.54×) were the
cause. Those two are 0.8 % of the corpus and the material closest to the
benchmark; the model had learned "on this kind of page, write the first word".

The remaining errors are 5,499 substitutions against 1,538 missing characters —
misreadings, which is the profile §18 arrived at for medieval v3.

**Do not compare v2's own split CER (0.0533) with v1's 0.0100.** v2 was scored by
the stratified draw (#120, fixed after v1 ran); v1's figure is five in-domain
pages from the head of `val.jsonl` (§20). The benchmark row above is the
comparison.

Trained in 10 h 11 on one H100 (`gpu`, `job_gratis`, 12 CPUs × 15 h — see §21a
for why not 16 × 24 h), from job `20260916T090417Z-qwen3vl-german-xix-v2`.

### What the UBELIX tooling did during this

`ubelix/` moved to `thodel/training-atr-models` (#7 there), together with the
part-2 pinning of #147: `submit.sh` records the commit and every batch file runs
a git worktree of it. A peer session's review found two defects in that work — a
Slurm job still wrote the registry through the disable-before-replace and the
promotion gate, and `pin_code` accepted a half-made worktree — both fixed before
the merge, and the acceptance smoke on UBELIX then ran end to end from the merged
tree: `completed`, registry untouched, the commit recorded in every stage and in
`metadata.json`.

## 24. The other three sizes: what four retrained models say together

§23 proved the collapse on one model. The remaining three arms of the 19th-century
grid — Qwen3.5 at 4B, 2B and 0.8B — were retrained on the same corrected corpus
(UBELIX `15560727/28/29`) and scored on the same 2,751 benchmark lines on
2026-09-19 (`15696149/50/51`).

| model | base | CER v1 | **CER v2** | WER v2 | `length_ratio` v2 | collapsed v1 → v2 |
|---|---|---:|---:|---:|---:|---|
| `qwen3.5-4b-german-xix-v2` | Qwen3.5-4B | 0.3596 | **0.0680** | 0.2342 | 0.9996 | 807 (29.3 %) → **0** |
| `qwen3vl-german-xix-v2` | Qwen3-VL-4B | 0.2551 | **0.0765** | 0.2458 | 1.0019 | 507 (18.4 %) → **0** |
| `qwen3.5-2b-german-xix-v2` | Qwen3.5-2B | 0.2937 | **0.0895** | 0.2624 | 1.0005 | 611 (22.2 %) → **0** |
| `qwen3.5-0.8b-german-xix-v2` | Qwen3.5-0.8B | — | **0.1115** | 0.3065 | 0.9975 | — → **0** |

Three things this adds to §23, none of which one model could have shown:

**The collapse was never architectural.** It disappears in all four models, across
two different model families and a 5× spread in parameters, with nothing changed
but the corpus. A single model going from 507 collapses to 0 left room for a lucky
run; four independent runs going to exactly 0 do not.

**v1 inverted the ranking.** Qwen3.5-4B was the *worst* of the three v1 models
(0.3596, and 29.3 % collapsed — the highest of any arm) and is the *best* v2 model
(0.0680, ahead of Qwen3-VL's 0.0765). The truncated corpus hurt it hardest, so its
v1 number was a statement about the data and not about the architecture. Any
model-selection decision taken on the v1 grid would have picked the wrong family.
This is the concrete cost of §21's defect, and the reason the v1 cards name their
successor rather than merely carrying a warning.

**Scaling is monotone again**: 0.8B 0.1115 → 2B 0.0895 → 4B 0.0680. On the v1 grid
it was not (4B worse than 2B worse than Qwen3-VL), which in hindsight was a second,
quieter symptom of the same defect — a corpus that truncates a third of its
characters rewards a model for stopping early, and the larger the model, the more
reliably it learns to. The 0.8B size, left out in 2026-09 because it lost on the
medieval corpus, beats every v1 model by a wide margin and is the cheap option for
bulk runs.

**What is not solved.** WER stays at 0.23–0.31 against CERs of 0.07–0.11.
`convention_normalized` moves it by about 0.005, so this is not the
whitespace/convention question of §20 — it is genuine word-level error spread thin
across many lines (4B: 4,995 substitutions against 1,437 missing characters). For
reading and for semantic search that is fine; for word-level search on these
transcriptions it is not, and nothing in this grid addresses it.

All four are on the share and published privately to `dh-unibe/…`. Two corrections
made while publishing, both found by reading the result back rather than trusting
the upload:

* `dh-unibe/qwen3vl-german-xix-v2` had been **public** since 2026-09-18 — created
  that way by the upload, not flipped afterwards — against the standing rule that
  these repos are private. The check I ran at the time printed `private=False` and
  I did not act on it. Now private.
* The three new directories landed on the share as `drwxr-----`: `rsync --no-perms`
  carried the umask, not the siblings' `drwxr-sr-x`, so the group — i.e. asteraix —
  could not have read them. Now `2755`.

`scan_trained` returns an empty scan, not an error, when its root is not a
directory, and in the container `/scratch/network/…` needs the `/rs_scratch` bind
to resolve. Publishing without that bind therefore reports "0 models" and exits 0.
With `--only` it raises instead; without it, it succeeds silently.
