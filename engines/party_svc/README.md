# Party Engine Service

**Engine ID:** `party`  
**Always-on:** Yes — the party model is pre-loaded and kept resident at startup.

## Model

- **ID:** `10.5281/zenodo.20642057` (Zenodo)
- **Type:** Handwritten Text Recognition (HTR)
- **Powered by:** `kraken` / `mittagessen/party`
- **Cached at:** `~/.kraken/10.5281/zenodo.20642057.mlmodel`

The model is downloaded on first startup if not already cached, then kept
permanently in memory for low-latency inference.

The Zenodo release ships **`model.safetensors`**, and this service used to load it
with `kraken.lib.models.load_any`, which is CoreML-only in kraken 7.0.2 — so it
failed at startup with `KrakenInvalidModelException` and `/health` reported
`degraded` (**#32**, one of the three engines in **#30**). Loading now goes
through `atr_serving.kraken_loader` → `kraken.models.load_models`, which
dispatches on the file. Verify after a restart: `/health` must say `ok`, not
`degraded`.

## Install

```bash
cd ~/Repo/serving-atr-inference
bash scripts/make_venvs.sh party
```

## Run

```bash
# Local development
.venvs/party/bin/python -m uvicorn party_svc.app:app --host 127.0.0.1 --port 8203

# Or run directly
.venvs/party/bin/python -m party_svc.app
```

## Systemd (production)

asterAIx has **no passwordless sudo** and the docker socket is denied, so every
engine runs as a `systemctl --user` unit:

```bash
bash scripts/install_user_units.sh
systemctl --user enable --now atr-party
```

**GPU affinity:** the unit sets `CUDA_VISIBLE_DEVICES=1`, so the service uses
GPU 1 — GPU 0 belongs to the box's RAG service and stays untouched.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + model status |
| `POST` | `/recognize` | HTR inference (multipart image) |
| `POST` | `/ocr` | Alias for `/recognize` |

### `/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "10.5281/zenodo.20642057",
  "resident": true
}
```

### `/recognize` / `/ocr`

- **Form field `file`:** image file (JPEG, PNG, TIFF, …)
- **Form field `model`:** ignored (party is always used)

```json
{
  "model": "10.5281/zenodo.20642057",
  "engine": "party",
  "text": "transcribed text...",
  "lines": [...],
  "confidence": 0.97,
  "timing_ms": 234,
  "segmented_by": null,
  "version": "0.1.0"
}
```