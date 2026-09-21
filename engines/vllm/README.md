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
- the German 19th-century fine-tunes (`qwen3vl-german-xix-v{1,2}`,
  `qwen3.5-4b-german-xix-v2`) — lazy, **page-level**; see
  `docs/GERMAN_XIX_MODELS.md`

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
PATH=.venvs/<venv>/bin:$PATH CUDA_VISIBLE_DEVICES=1 .venvs/<venv>/bin/vllm serve <merged dir or hf_repo> \
    --host 127.0.0.1 --port <8210+> --served-model-name <id> \
    --gpu-memory-utilization <budget> --trust-remote-code --max-model-len 16384 \
    [--max-num-seqs <n>]
```
`<venv>` is `vllm` unless the registry entry sets `vllm_venv` (see below). The
venv's `bin/` goes first on `PATH` because the venv is never activated and vLLM
0.29 shells out to `ninja` while compiling kernels at start-up. `<budget>` is
computed per launch from `vram_mb` and what is free on the card
(`manager.plan_gpu_budget`). `--max-num-seqs` comes from the entry's
`max_num_seqs`. Other tunables live in `Settings` (`vllm_*`): GPU index, port
base, VRAM budget override, max-model-len, startup timeout.

## Two vLLMs
| venv | vLLM | for | build |
|---|---|---|---|
| `.venvs/vllm` | 0.11.0, torch 2.8.0 cu128, transformers 4.57 | every model without `vllm_venv` | `make_venvs.sh vllm` |
| `.venvs/vllm-next` | 0.29.0+cu129, torch 2.13.0+cu129, transformers 5.17, peft 0.21 | `vllm_venv: vllm-next` — Qwen3.5 (`qwen3.5-4b-german-xix-v2`) | `make_venvs.sh vllm-next` |

vLLM 0.11 does not know Qwen3.5, and its transformers cannot merge a Qwen3.5
adapter; merge those with `.venvs/vllm-next/bin/python scripts/merge_loras.py`.
The newer vLLM's default PyPI wheel is CUDA 13 and does not run on this driver;
the cu129 build does (#132, `docs/idhefix-environment.md`).

Qwen3.5 is a hybrid model: vLLM allocates one Mamba-style state block per running
sequence up front, and at its default of 256 sequences the model does not start on
the share of card 1 it gets ("max_num_seqs (256) exceeds available Mamba cache
blocks (130)"). Its entry sets `max_num_seqs: 64`.

## Locally fine-tuned adapters
The training service can QLoRA-fine-tune a Qwen3-VL base (`engine: "vllm"`), but
vLLM 0.11 **will not serve the adapter**: it refuses a LoRA that touches the
vision tower ("only supports adding LoRA to language model"). `scripts/merge_loras.py`
bakes an adapter into its base, and the manager serves the merged directory if
present. This is why a VLM training job never passes the promotion gate (#36) and
says so on its record, rather than being advertised in `/models` unserved.

## Use
- `POST /recognize` with a vLLM `model` id (page → one call; line → segmented).
- `POST /v1/chat/completions` — OpenAI passthrough; the manager makes the model
  resident first.
