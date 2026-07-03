# Deploying on asterAIx

Provisioning runbook for the DH GPU box (`srv`, user `tobias`, 2× A40). Host facts
and the reasoning behind these choices live in
[`asteraix-environment.md`](asteraix-environment.md).

Everything runs as **`systemctl --user` units** (no root needed) and binds to
`127.0.0.1` except the gateway. vLLM is **not** a unit — the ModelManager spawns it
as a subprocess (see IMPLEMENTATION_PLAN.md §8).

## Host baseline (confirmed by `scripts/probe_host.sh`, 2026-06-26)

Ubuntu 24.04 · NVIDIA driver 565.57.01 / CUDA 12.7 · 2× A40 (~45 GB) · **Python 3.12
only** · no passwordless sudo · `Linger=no` · GPU 0 shared with a RAG service.
Re-run the probe if the box changes.

## 1. Clone

```bash
mkdir -p ~/Repo && cd ~/Repo
git clone https://github.com/thodel/serving-atr-inference.git
cd serving-atr-inference
```

The unit files assume `%h/Repo/serving-atr-inference`. If you clone elsewhere, edit
`deploy/systemd/*.service` accordingly.

## 2. Build the per-engine venvs (Python 3.12)

```bash
bash scripts/make_venvs.sh          # gateway + kraken + party + trocr
```

vLLM's venv is built by #5. First, validate the engines install on 3.12:

```bash
bash scripts/spike_engine_installs.sh
```

If `kraken` or `party` FAIL on 3.12, ask an admin for a `python3.11` (deadsnakes)
venv and set `PYTHON=python3.11` for that engine.

## 3. Configure `.env`

```bash
cp .env.example .env
python -c "import secrets; print('ATR_API_KEY=' + secrets.token_urlsafe(32))" >> .env  # then dedupe
```

Set in `.env`:
- `ATR_API_KEY` — a strong shared secret. **The same value goes on the
  agentic_historian VM** (it sends it as `X-API-Key`).
- `HF_HOME=/home/tobias/atr-cache/hf` — keep weights off the 80%-full root default
  and somewhere you can monitor.

## 4. Prefetch model weights + merge vLLM LoRA adapters

```bash
set -a; . ./.env; set +a                     # HF_HOME
python scripts/download_models.py            # HF adapters + bases; honors HF_HOME
```

The vLLM models are **LoRA adapters** (Qwen3-VL / LightOnOCR) whose adaptation
includes the vision tower, which vLLM can't serve as a runtime LoRA. Bake each into
its base (needs the vLLM venv; also downloads the bases if missing):

```bash
.venvs/vllm/bin/python scripts/merge_loras.py    # -> ~/atr-cache/vllm-merged/<id>
```

The gateway's ModelManager serves the merged full model automatically (config
`vllm_merged_dir`). Note the pinned vLLM knobs in `Settings`: `max_model_len=16384`
(Qwen3-VL's 262k default OOMs the KV cache) and `gpu_memory_utilization≈0.70`.
kraken/party download their Zenodo models on demand via htrmopo.

## 5. Install + start the user services

```bash
bash scripts/install_user_units.sh
```

This installs `atr-kraken`, `atr-trocr`, `atr-party`, `atr-gateway` as user units,
enables and starts them (engines first, gateway last).

**One-time admin step** so services survive logout:

```bash
sudo loginctl enable-linger tobias
```

## 6. Open the gateway to the client host only

Topology: **asterAIx** (`srv`, `130.92.59.240`) runs this server; the client is
**agentic_historian on `tei.dh.unibe.ch`**. asterAIx has a routable IP, so expose
`:8200` **only** to `tei.dh.unibe.ch` (needs admin once):

```bash
CLIENT_IP=$(getent hosts tei.dh.unibe.ch | awk '{print $1}')   # resolve to an IP
sudo ufw allow from "$CLIENT_IP" to any port 8200 proto tcp
sudo ufw reload
```

Engines stay on `127.0.0.1` (never exposed). Auth is the shared `X-API-Key`; the
gateway logs a SECURITY warning if it starts exposed with the default key.

> TODO: confirm `tei.dh.unibe.ch` resolves to the IP that actually reaches asterAIx
> (it may egress via a different address) and check `ufw status`.

## 7. Verify

```bash
curl -s localhost:8200/health | python -m json.tool
curl -s -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)" localhost:8200/models | python -m json.tool | head
journalctl --user -u atr-gateway -f
```

From the agentic_historian host (`tei.dh.unibe.ch`):

```bash
curl -s -H "X-API-Key: <shared-key>" http://130.92.59.240:8200/health
```

Then point `KRAKEN_SERVICE_URL` (agentic_historian) at `http://130.92.59.240:8200`;
its existing `KrakenHTTPClient` uses the legacy `/ocr` alias unchanged.

## Notes / known follow-ups
- vLLM units/subprocess + ModelManager land in #5/#6.
- Prometheus metrics (latency/VRAM/evictions) are a follow-up; logs are structured
  via loguru and visible through `journalctl --user`.
