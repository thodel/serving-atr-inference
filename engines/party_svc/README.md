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

## Install

```bash
# One-time venv creation (also handled by scripts/make_venvs.sh)
python3.12 -m venv .venvs/party
.venvs/party/bin/pip install --upgrade pip
.venvs/party/bin/pip install -r engines/party_svc/requirements.txt
```

## Run

```bash
# Local development
.venvs/party/bin/python -m uvicorn party_svc.app:app --host 127.0.0.1 --port 8203

# Or run directly
.venvs/party/bin/python -m party_svc.app
```

## Systemd (production)

```bash
cp deploy/systemd/atr-party.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now atr-party
```

**GPU affinity:** Set `CUDA_VISIBLE_DEVICES=1` in the service environment
(the systemd unit already does this). The service will use GPU 1 for inference.

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