# Kraken Engine Service

Standalone FastAPI service wrapping [kraken](https://github.com/mittagessen/kraken)
for ATR/OCR and layout analysis.

## What it does

- **`POST /segment`** – layout analysis via kraken's `blla` segmenter. Returns line bounding boxes and baselines.
- **`POST /recognize`** – OCR/HTR using a kraken recognition model. The `model`
  form field is either a **Zenodo DOI** (downloaded and cached on first use) or a
  **local path** — a weights file, or a directory the trainer registered under
  `~/atr-cache/trained/<model_id>/`. Models this box trained have no DOI, and this
  service has no registry to look one up in, so the gateway sends their
  `local_path` (#36).
- **`POST /ocr`** – legacy alias for `/recognize`, compatible with `agentic_historian`'s `KrakenHTTPClient`.

The service is **lazy**: no model is loaded at startup. Downloaded models are
cached in `engines/kraken_svc/models_cache/`; local ones are read where they are.

Weights are loaded through `atr_serving.kraken_loader`, which prefers
`kraken.models.load_models` — it dispatches on the file and therefore reads
**safetensors as well as CoreML**. The old `kraken.lib.models.load_any` is
CoreML-only in kraken 7.0.2 while `ketos train` writes safetensors by default,
which is why this service could not serve the trainer's own output and why
`atr-party` could not load its model at all (**#32**).

## Installing

Use the shared `scripts/make_venvs.sh` script:

```bash
cd ~/Repo/serving-atr-inference
bash scripts/make_venvs.sh kraken
```

This creates `.venvs/kraken` with all dependencies from `requirements.txt`
(including kraken, torch, torchvision). Pass the target explicitly: several
requirement files are ranges, so a bare run silently upgrades a serving engine
under a running service.

## Running

Development:

```bash
.venvs/kraken/bin/python -m uvicorn kraken_svc.app:app --host 127.0.0.1 --port 8201 --reload
```

Or directly:

```bash
.venvs/kraken/bin/python -m kraken_svc.app
```

Production (systemd) — asterAIx has **no passwordless sudo**, so these are
`systemctl --user` units, installed by `scripts/install_user_units.sh`:

```bash
bash scripts/install_user_units.sh
systemctl --user enable --now atr-kraken
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