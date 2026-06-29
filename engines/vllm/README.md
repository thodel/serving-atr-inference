# vLLM engine

Serves the page/line VLMs (LightOnOCR + the Qwen3-VL fine-tunes) via vLLM's
OpenAI-compatible server. Unlike the other engines, vLLM is **not** a systemd
unit — the gateway's `ModelManager` (`src/atr_serving/manager.py`) starts each
model as a `vllm serve` **subprocess** on demand and evicts the LRU one under
the VRAM budget (asterAIx: GPU 1 only, one 8B resident at a time).

## Models (from config/models.yaml)
- `lightonocr-catmus-caroline` — pinned, **line-level** (cropped lines)
- `qwen3vl-8b-old-church-slavonic` — lazy, **line-level**
- `qwen3vl-8b-hebrew` — lazy, **page-level**

Line-level models are driven through the gateway pipeline: kraken segments the
page, each line is cropped and sent to the VLM, results are reassembled.

## Setup
```bash
bash scripts/spike_engine_installs.sh vllm   # confirm install + pin
bash scripts/make_venvs.sh                   # builds .venvs/vllm
python scripts/download_models.py --engine vllm
```

## How the manager launches an instance
```
CUDA_VISIBLE_DEVICES=1 .venvs/vllm/bin/vllm serve <hf_repo> \
    --host 127.0.0.1 --port <8210+> --served-model-name <id> \
    --gpu-memory-utilization 0.45 --trust-remote-code
```
Tunables live in `Settings` (`vllm_*`): GPU index, port base, VRAM budget,
memory utilization, max-model-len, startup timeout.

## Use
- `POST /recognize` with a vLLM `model` id (page → one call; line → segmented).
- `POST /v1/chat/completions` — OpenAI passthrough; the manager makes the model
  resident first.
