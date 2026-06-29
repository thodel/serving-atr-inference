# ATR TrOCR Engine

FastAPI service for medieval / Kurrent / Latin OCR via TrOCR (HuggingFace `VisionEncoderDecoderModel`).

## Requirements

- Python 3.12 (asterAIx only has Python 3.12)
- CUDA (optional; falls back to CPU)

## Install

```bash
cd /home/dh/serving-atr-inference
./scripts/make_venvs.sh
```

This creates `.venvs/trocr/` and installs `requirements.txt`.

## Run

```bash
.venvs/trocr/bin/python -m uvicorn trocr_svc.app:app --host 127.0.0.1 --port 8202
```

Or with systemd (after installing the unit):

```bash
sudo systemctl start atr-trocr
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + loaded model info |
| GET | `/models` | List available TrOCR model IDs |
| POST | `/segment` | Segmentation — best-effort pass-through (no kraken bundled) |
| POST | `/recognize` | Run OCR on an image (`model` form field = HF repo) |
| POST | `/ocr` | Alias for `/recognize` |

## Available Models

- `dh-unibe/trocr-medieval-escriptmask` — Medieval (de, fr, la, nl), 13th–16th c.
- `dh-unibe/trocr-kurrent-XVI-XVII` — Kurrent (de), 16th–17th c.
- `dh-unibe/trocr-essoins-middle-latin` — Medieval Latin (la), 13th–15th c.

## systemd

```bash
sudo cp deploy/systemd/atr-trocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now atr-trocr
```
