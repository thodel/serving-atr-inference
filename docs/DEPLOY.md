# Deploying the serving side on idhefix

This is the provisioning runbook for idhefix (130.92.59.240, `srv`, 2× A40), the serving
box. It covers the gateway, the recognition engines and vLLM, and nothing else.

- How the two machines fit together, and which values must agree across them:
  [`INFRASTRUCTURE.md`](INFRASTRUCTURE.md).
- The host probe of June 2026, and the reasoning behind these choices:
  [`idhefix-environment.md`](idhefix-environment.md).
- **Training** runs on asteraix since 16.09.2026, from
  [training-atr-models](https://github.com/thodel/training-atr-models), and is deployed
  there. See [§8](#8-training-is-not-on-this-box).

Everything runs as **`systemctl --user` units**, so no root is needed. Every unit binds
to `127.0.0.1` except the gateway. vLLM is **not** a unit: the ModelManager spawns it as
a subprocess (see IMPLEMENTATION_PLAN.md §8).

## Host baseline

This is the state measured on 16.09.2026. The June probe
([`scripts/probe_host.sh`](../scripts/probe_host.sh)) recorded the same box. Re-run the
probe if the box changes.

- Ubuntu 24.04.3, NVIDIA driver 565.57.01, 2× A40 (46068 MiB each)
- **Python 3.12 only**
- no passwordless sudo; linger is on
- **GPU 0 belongs to the neighbours' RAG service.** Everything here runs on GPU 1.

## 1. Clone

```bash
mkdir -p ~/Repo && cd ~/Repo
git clone https://github.com/thodel/serving-atr-inference.git
cd serving-atr-inference
```

The unit files assume the checkout is at `%h/Repo/serving-atr-inference`. If you clone
elsewhere, edit `deploy/systemd/*.service` to match.

## 2. Build the per-engine venvs (Python 3.12)

Build the serving venvs by name:

```bash
bash scripts/make_venvs.sh gateway kraken party trocr vllm
```

Without arguments, the script builds **every** venv it knows, including the three
training venvs. No service on this box uses those any more, but two scripts still do:

- `scripts/publish_to_hub.py` runs from `.venvs/kraken-train` until it moves to
  training-atr-models (training-atr-models#6).
- `scripts/merge_loras.py` stays in this repository, and for an adapter trained on
  asteraix it usually needs `.venvs/vlm-train` (§8).

#143 removes the three venvs after the v5 acceptance run. It does not yet say which
venv merges our adapters after that. **Never run `make_venvs.sh` without arguments on
a live box.** Several requirement files specify ranges rather than pins, so a blanket
run silently upgrades a serving engine under a running service.

`scripts/spike_engine_installs.sh` checked on 2026-06-29 that the engine stacks install
on Python 3.12: all passed (results in
[`idhefix-environment.md`](idhefix-environment.md#engine-install-results-spike-2026-06-29--all-pass-on-python-312)).
If `kraken` or `party` ever fail on 3.12, ask an admin for a `python3.11` (deadsnakes)
and set `PYTHON=python3.11` for that engine.

## 3. Configure `.env`

```bash
cp .env.example .env && chmod 600 .env
python -c "import secrets; print('ATR_API_KEY=' + secrets.token_urlsafe(32))" >> .env  # then dedupe
```

Set these in `.env`. The values that must match asteraix are listed in
[INFRASTRUCTURE.md § Shared values](INFRASTRUCTURE.md#shared-values).

- `ATR_API_KEY`: a strong secret. **The same value goes to every client**
  (agentic_historian on tei sends it as `X-API-Key`) **and into asteraix's `.env` as
  `ATR_TRAIN_GATEWAY_API_KEY`**, where the promotion gate uses it.
- `ATR_TRAIN_URL=http://130.92.59.242:8204`: the trainer, on asteraix.
- `ATR_TRAIN_API_KEY`: the same value as in asteraix's `.env`, and **not** the same as
  `ATR_API_KEY`.
- `ATR_REGISTRY_ROOT=/mnt/wbkolleg_dh_1/Textrecognition_Training/registry`: the shared
  registry. This is how models trained on asteraix get served here.
- **Do NOT set `HF_HOME`.** Older revisions of this file told you to point it at
  `~/atr-cache/hf`, which put about 26 GB of weights on the root partition that later
  filled up. `~/.cache/huggingface/hub` is a symlink to
  `/mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub`, so the standard path already
  resolves to the research share. That cache is shared with asteraix and with
  `lassberg/vlm_training`.

Before any later edit, back up `.env` to `~/atr-cache/env-backups/` at mode 600. Never
commit it, or any copy of it: `.gitignore` covers `.env.*` because a backup once sat
unignored in this public repository.

## 3b. Post-provisioning verification

Run the smoke test right after `make_venvs.sh`, and again whenever a venv is rebuilt:

```bash
bash scripts/check_venvs.sh
```

It runs two checks per venv, and **the second one is the point**:

1. **An import smoke test** of what the code in this repository actually imports from
   that venv. A broken or incomplete dependency tree fails here, not at the first
   request.
2. **A version check** against the venv's own `requirements.txt`
   (`scripts/check_requirements.py`). The installed version has to satisfy the
   requirement the venv was built from.

Imports alone would not have caught the transformers 5.x incident (#48):
`import transformers` worked on 5.14.1, and so did the code built on it. The mismatch
was found by printing `transformers.__version__`. The failed repair had the same shape:
the downgrade died with `EPERM`, and the venv silently kept 5.14.1. When a `MISMATCH`
line appears, the usual cause of a version that refuses to change is a `TMPDIR` on the
share (#54).

Every expectation is read from the requirements files themselves, so there is no second
list to drift. Add `-v` to see the satisfied requirements too. The script exits 0 only
when every **present** venv passes. A venv that was never built is reported as `SKIP`,
not as a failure.

## The research share is not POSIX

The share is CIFS, and several parts of the stack broke on it before the rules were
understood. The rules, each with the incident that taught it, are in
[INFRASTRUCTURE.md § CIFS rules](INFRASTRUCTURE.md#cifs-rules-each-one-learned-the-hard-way).
The short version: **data on the share; scratch, caches and anything written
incrementally on local disk.**

One of those limitations applies on this box as well.

### Known limitation: CIFS hub cache symlink

On both machines, `~/.cache/huggingface/hub` is a symlink to a CIFS share:

```
~/.cache/huggingface/hub → /mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub
```

CIFS does not support the `chmod` and `symlink` operations that `huggingface_hub` uses
to deduplicate blobs. Every download is therefore stored in full, even when the blob is
already on the share. This is **harmless but not optimal**: the symlink works, models
load correctly, and the only cost is extra disk I/O. No error is raised, but
`huggingface_hub` prints a warning like this one, which can be ignored:

```
Could not set permissions on [...] Operation not permitted
```

> **It is *not* related to the transformers 5.x incident (#48)**, even though both
> print `Operation not permitted`. That incident had two causes. A requirement without
> an upper bound installed 5.14.1. The later downgrade then failed because `TMPDIR`
> pointed at the share: pip stages the files it replaces in `TMPDIR`. Since `eb3b202`,
> `make_venvs.sh` replaces a network `TMPDIR` with a local one.

## 4. Prefetch model weights and merge vLLM LoRA adapters

```bash
set -a; . ./.env; set +a
python scripts/download_models.py            # HF adapters and bases, into the shared hub cache
```

The vLLM models are **LoRA adapters** (Qwen3-VL, LightOnOCR) whose adaptation includes
the vision tower, and vLLM cannot serve that as a runtime LoRA. Merge each adapter into
its base. This needs the vLLM venv and downloads missing bases:

```bash
.venvs/vllm/bin/python scripts/merge_loras.py    # -> ~/atr-cache/vllm-merged/<id>
```

The gateway's ModelManager serves the merged full model automatically (setting
`vllm_merged_dir`). Note the pinned vLLM setting `max_model_len=16384` in `Settings`:
Qwen3-VL's default of 262k runs the KV cache out of memory. kraken and party download
their Zenodo models on demand through htrmopo.

### How much of the card a model gets

`--gpu-memory-utilization` is **a fraction of the card's total memory**, and vLLM
refuses to start unless that much is *free*. No single constant satisfies both
constraints. On 2026-09-14 a 12 GB model failed to start three times in a row on GPU 1
(45.5 GB total, 19 GB free at the time), because the configured 0.70 asks for 31 GB.

The launcher therefore computes the fraction for each launch, from the registry's
`vram_mb` and `nvidia-smi` (`manager.plan_gpu_budget`):

`min(vram_mb × 1.6, free − 2048 MiB) / total`, rounded down to two decimals

The ×1.6 covers the KV cache, activation scratch space and CUDA graphs, which
`vram_mb` does not include. Both numbers are estimates: the ×1.6 was fitted to
one model, and no `vram_mb` in the registry was ever measured against a card
(#130). `scripts/measure_vram.py` is how one gets measured —

```bash
python scripts/measure_vram.py --model qwen3vl-german-xix-v1 --write
python scripts/measure_vram.py --report      # what is measured so far
```

It warms the model through `POST /recognize`, reads the resident total from
`GET /gpu`, and takes the weights/KV split from vLLM's own memory-profiling line
in the gateway's journal. The split is what matters: `nvidia-smi` alone reports
what vLLM was *granted* by the arithmetic above, not what it needs, so feeding
that back into `vram_mb` would close a circle rather than measure anything. A
measured entry carries `vram_measured:` in `config/models.yaml`, and the launch
line then says so:

```
vLLM qwen3vl-german-xix-v1 gpu budget: 0.42 = 19200 of 45516 MiB
  (12000 MiB weights x 1.6 for KV cache), 31047 MiB free
  [measured 2026-09-26 on idhefix: 8521 MiB of weights, -3479 MiB against the registry's 12000]
``` The 2048 MiB stays with the card. Every launch logs the
number and the arithmetic behind it:

```
vLLM qwen3vl-german-xix-v1 gpu budget: 0.42 = 19200 of 45516 MiB
  (12000 MiB weights x 1.6 for KV cache), 31047 MiB free
```

If the card cannot hold the model even at 1.15×, the request fails **before** the
launch, with a message that names the free and total memory. Otherwise vLLM would fail
only after a minute of loading weights. `GET /gpu` then shows what is holding the
memory.

How many models may be resident at once is a separate budget, derived from the card
at every launch (see
[INFRASTRUCTURE.md § Services](INFRASTRUCTURE.md#idhefix-reposerving-atr-inference)).

| variable | default | |
|---|---|---|
| `ATR_VLLM_AUTOSIZE` | `true` | `false` goes back to the fixed constant |
| `ATR_VLLM_GPU_MEMORY_UTILIZATION` | `0.70` | fallback when there is no `vram_mb` or no readable `nvidia-smi` |
| `ATR_VLLM_VRAM_HEADROOM` | `1.6` | multiplier on `vram_mb` |
| `ATR_VLLM_VRAM_RESERVE_MB` | `2048` | never handed to vLLM |
| `ATR_VLLM_VRAM_BUDGET_MB` | unset | overrides the resident budget read from the card |

## 5. Install and start the user services

```bash
bash scripts/install_user_units.sh
```

This installs `atr-kraken`, `atr-trocr`, `atr-party` and `atr-gateway` as user units,
then enables and starts them, engines first and the gateway last. It prints a warning
if the retired `atr-train` is still enabled or running here (§8). With `--no-start`, it
installs and enables the units without starting them.

Units survive logout only with linger. Linger is on for this user (16.09.2026). On a new
box, it is a one-time admin step:

```bash
sudo loginctl enable-linger tobias
```

## 6. Who may reach the gateway

- **idhefix** (130.92.59.240) runs this server.
- The **clients** are agentic_historian on `tei.dh.unibe.ch` and the ATR-MCP.
- **asteraix** (130.92.59.242) also calls in, for the promotion gate.

According to the code comments, `ufw` opens `:8200` to tei; this was not re-measured on
16.09.2026. asteraix's gate requests do arrive (measured 16.09.2026), so the current
rules also admit 130.92.59.242. A rule scoped to tei alone would break the gate: every
kraken model would stay disabled, while its job still completes. Setting up the
firewall needs an admin:

```bash
CLIENT_IP=$(getent hosts tei.dh.unibe.ch | awk '{print $1}')   # resolve to an IP
sudo ufw allow from "$CLIENT_IP" to any port 8200 proto tcp
sudo ufw allow from 130.92.59.242 to any port 8200 proto tcp    # asteraix: the promotion gate
sudo ufw reload
```

The tei rule is the one the code comments describe. The asteraix rule is reconstructed
from the gate requests that arrive, not read from `ufw status`: an existing rule may
be broader. If asteraix's address changes, this rule must follow it
([INFRASTRUCTURE.md § Shared values](INFRASTRUCTURE.md#shared-values)).

Engines stay on `127.0.0.1` and are never exposed. Authentication is the `X-API-Key`
header. The gateway logs a SECURITY warning if it starts exposed with the default key,
and another if `ATR_TRAIN_URL` points off the box while `ATR_TRAIN_API_KEY` is empty.

> TODO: confirm that `tei.dh.unibe.ch` resolves to the IP that actually reaches idhefix
> (it may leave through a different address), and check `ufw status`.

## 7. Verify

```bash
curl -s localhost:8200/health | python -m json.tool     # engines and the trainer, each with "reachable"
KEY=$(grep ^ATR_API_KEY= .env | cut -d= -f2)
curl -s -H "X-API-Key: $KEY" localhost:8200/models | python -m json.tool | head
curl -s -H "X-API-Key: $KEY" localhost:8200/gpu | python -m json.tool | head        # this box
curl -s -H "X-API-Key: $KEY" localhost:8200/train/gpu | python -m json.tool | head  # asteraix
curl -s -H "X-API-Key: $KEY" localhost:8200/train/jobs | head -c 300                # key and allowlist
journalctl --user -u atr-gateway -f
```

`reachable` in `/health` only means that the service answered with a status below
500. A trainer that refuses this gateway with `401` or `403` still reads as reachable,
so `/train/jobs` is the check that proves the trainer accepts this box. What each
`/train/*` failure means is in
[INFRASTRUCTURE.md § When /train/* fails](INFRASTRUCTURE.md#when-train-fails).

From the agentic_historian host (`tei.dh.unibe.ch`):

```bash
curl -s http://130.92.59.240:8200/health
```

Then point `KRAKEN_SERVICE_URL` (agentic_historian) at `http://130.92.59.240:8200`. Its
existing `KrakenHTTPClient` uses the legacy `/ocr` alias unchanged.

## 8. Training is not on this box

Training moved to asteraix (130.92.59.242) on 16.09.2026 (#137, #139). It is built,
configured and deployed from
[training-atr-models](https://github.com/thodel/training-atr-models). Its job
lifecycle, local and shared paths, and deploy rules are described there.

This box needs only three things for training to work:

- `ATR_TRAIN_URL` and `ATR_TRAIN_API_KEY` in `.env` (§3). With them, callers keep using
  `/train/*` on `:8200`.
- `ATR_REGISTRY_ROOT` (§3), so that trained models are served without a restart.
- `scripts/merge_loras.py` (§4), before a trained VLM adapter can be served.

**Merging an adapter trained on asteraix.** peft writes its own version into the
adapter, and an older peft misreads it with an `AttributeError` that names neither
peft nor a version. Merge in a venv whose peft is at least as new as the one in
asteraix's `.venvs/vlm-train`. On this box that is usually `.venvs/vlm-train`, not
`.venvs/vllm`. The script's preflight compares the two versions and stops before
loading anything if the venv is too old:

```bash
cd ~/Repo/serving-atr-inference              # .env there names ATR_REGISTRY_ROOT, so trained/ is read
.venvs/vlm-train/bin/python scripts/merge_loras.py --list
.venvs/vlm-train/bin/python scripts/merge_loras.py --only ID
```

The merged model is still registered `enabled: false`. Enabling it is described in
[INFRASTRUCTURE.md § When the gate did not promote a model](INFRASTRUCTURE.md#when-the-gate-did-not-promote-a-model).

**The in-repo trainer stays retired.** It is disabled and stopped, and its unit file
was moved to `~/atr-cache/retired-units/`. It is no longer in
`deploy/systemd/` or `install_user_units.sh`. It has none of training-atr-models#15's
job-ownership rules, so on the shared job store it would mark asteraix's runs failed,
start asteraix's queued jobs, and delete registrations in progress. Check that it is
still off:

```bash
systemctl --user status atr-train      # must be disabled or "could not be found"
```

The retired setup is described in this file's history (before 16.09.2026) and in
[`TRAINING.md`](TRAINING.md) and [`VLM_TRAINING.md`](VLM_TRAINING.md), whose measurements
were all taken on this box. Removing the training venvs from this box is part of #143;
removing the in-repo training code is a follow-up named there.

## 9. Deploying an update

```bash
cd ~/Repo/serving-atr-inference
KEY=$(grep ^ATR_API_KEY= .env | cut -d= -f2)
ss -tn state established '( sport = :8200 )'     # any requests in flight? Wait for them.
curl -s -H "X-API-Key: $KEY" localhost:8200/train/jobs | python3 -c \
  'import json,sys; print([j["id"] for j in json.load(sys.stdin)["jobs"] if j["status"] == "registering"])'
                                                  # must print []; otherwise wait
git pull --ff-only
systemctl --user restart atr-gateway
```

- **Wait while a job is in `registering`.** Its promotion gate may be running, and a
  gateway that refuses the connection ends the gate at once, without a retry. The
  model then stays disabled
  ([INFRASTRUCTURE.md § What survives what](INFRASTRUCTURE.md#what-survives-what)).
  `ss` can miss it, because the gate holds no connection between two attempts.
- **A gateway restart drops every resident vLLM model.** The models run as children of
  the unit. The next request for a model reloads it, which takes about 45 s for the 4B
  xix. The registry is read again: the curated models from the checkout, the trained
  ones from the share, and the legacy overlay `config/models.local.yaml`
  ([INFRASTRUCTURE.md § Where a model is registered](INFRASTRUCTURE.md#where-a-model-is-registered)).
- **Engines** are restarted individually:
  `systemctl --user restart atr-kraken` (or `atr-trocr`, `atr-party`).
- **A changed unit file**: run `bash scripts/install_user_units.sh`, then
  `systemctl --user restart` that unit. The script copies the units and reloads
  systemd, but it only starts units that are stopped, so a running service keeps the
  old unit until it is restarted.
- **A changed requirements file** needs a rebuild of that venv, by name (§2), followed
  by `bash scripts/check_venvs.sh`. `git pull` alone never changes a venv.

## Notes and known follow-ups

- Prometheus metrics (latency, VRAM, evictions) are a follow-up. Logs are structured
  through loguru and visible with `journalctl --user`.
