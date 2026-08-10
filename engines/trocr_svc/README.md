# ATR TrOCR Engine

FastAPI service for medieval / Kurrent / Latin OCR via TrOCR (HuggingFace `VisionEncoderDecoderModel`).

## Requirements

- Python 3.12 (asterAIx only has Python 3.12)
- CUDA (optional; falls back to CPU)

## Install

```bash
cd ~/Repo/serving-atr-inference
bash scripts/make_venvs.sh trocr
```

This creates `.venvs/trocr/` and installs `requirements.txt`. Name the target:
several requirement files are ranges, so a bare run upgrades other engines under
their running services.

## Run

```bash
.venvs/trocr/bin/python -m uvicorn trocr_svc.app:app --host 127.0.0.1 --port 8202
```

Or with systemd — `systemctl --user`, since the box has no passwordless sudo:

```bash
bash scripts/install_user_units.sh
systemctl --user enable --now atr-trocr
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + loaded model info |
| GET | `/models` | List available TrOCR model IDs |
| POST | `/segment` | Segmentation — best-effort pass-through (no kraken bundled) |
| POST | `/recognize` | Run OCR on an image (`model` form field = HF repo, or a local path for a model trained here — #36) |
| POST | `/ocr` | Alias for `/recognize` |

## Available Models

- `dh-unibe/trocr-medieval-escriptmask` — Medieval (de, fr, la, nl), 13th–16th c.
- `dh-unibe/trocr-kurrent-XVI-XVII` — Kurrent (de), 16th–17th c.
- `dh-unibe/trocr-essoins-middle-latin` — Medieval Latin (la), 13th–15th c.

## Fine-tuning

A TrOCR **training** backend is planned but not built — epic **#41** (#43 argv +
contracts, #44 the service). Until it exists, the models above come from the hub;
nothing here trains one.
