# Kraken Engine Service

Standalone FastAPI service wrapping [kraken](https://github.com/mittagessen/kraken)
for ATR/OCR and layout analysis.

## What it does

- **`POST /segment`** – layout analysis via kraken's `blla` segmenter. Returns line bounding boxes and baselines.
- **`POST /recognize`** – OCR/HTR using a kraken recognition model (downloaded from Zenodo and cached locally on first use).
- **`POST /ocr`** – legacy alias for `/recognize`, compatible with `agentic_historian`'s `KrakenHTTPClient`.

The service is **lazy**: no model is loaded at startup. Recognition models are
downloaded and cached in `engines/kraken_svc/models_cache/` on first request.

## Installing

Use the shared `scripts/make_venvs.sh` script:

```bash
cd /home/dh/serving-atr-inference
./scripts/make_venvs.sh
```

This creates `.venvs/kraken` with all dependencies from `requirements.txt`
(including kraken, torch, torchvision).

## Running

Development:

```bash
.venvs/kraken/bin/python -m uvicorn kraken_svc.app:app --host 127.0.0.1 --port 8201 --reload
```

Or directly:

```bash
.venvs/kraken/bin/python -m kraken_svc.app
```

Production (systemd):

```bash
sudo systemctl link deploy/systemd/atr-kraken.service
sudo systemctl enable --now atr-kraken
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | `{"status": "ok", "model_loaded": bool, "model_id": str|null}` |
| `GET` | `/models` | `{"models": [str, …]}` — IDs cached on disk |
| `POST` | `/segment` | multipart image → `SegmentResponse` |
| `POST` | `/recognize` | multipart image + `model` param → `RecognitionResult` |
| `POST` | `/ocr` | same as `/recognize` (legacy alias) |

## Compatibility

`POST /ocr` is provided for backward compatibility with
`agentic_historian`'s `KrakenHTTPClient`, which calls `/ocr` instead of `/recognize`.

## Ports

- **8201** – kraken engine (configured in `src/atr_serving/config.py`)