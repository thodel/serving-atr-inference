# asterAIx — environment & derived decisions

Captured from `scripts/probe_host.sh` on **2026-06-26**. This is the authoritative
description of the target host; engine pins and deploy choices derive from it.
Re-run the probe and update this file if the box changes.

## Facts

| Area | Value |
|---|---|
| Host / user | `srv` / `tobias` (no passwordless sudo) |
| IP | `130.92.59.240` (Uni Bern public range — firewall + API key matter) |
| OS | **Ubuntu 24.04.3 LTS** (noble), kernel 6.8 |
| CPU / RAM | Threadripper PRO 5965WX, 48 threads / **251 GB** |
| Disk | single `/` partition, 1.8 T, **80 % used, ~356 G free** |
| GPUs | **2× NVIDIA A40, 46068 MiB (~45 GB) each**, compute **8.6** (bf16 OK) |
| Driver / CUDA | **565.57.01 / CUDA 12.7** capability; toolkits 12.1/12.4/12.6 installed, `nvcc` 12.6 |
| **GPU 0** | **shared** — ~10 GB used by an existing `rag-change/venv` service |
| **GPU 1** | **free** (4 MiB) |
| Python | **3.12.3 only** (`/usr/bin/python3`); no 3.11; `venv` available; pip 24.0 |
| Package mgrs | no conda, no Lmod modules, **no Slurm** (plain box) |
| Containers | docker 28.5 present but **socket denied** (user not in `docker` group); **rootless podman 4.9.3** works; NVIDIA Container Toolkit 1.18 installed |
| systemd | v255; **Linger=no**; user cannot `sudo -n`; can use `systemctl --user` |
| Ports in use | `:8000`(local), `:8080`, `:9000`, `:11434`(Ollama), `:80`(nginx), `:22` |
| Existing svcs | docker, nginx, Ollama (`:11434`), a RAG service on GPU 0 |
| Dev tools | git 2.43, gcc/g++ 13.3, make 4.3 (gh NOT installed on the box) |

## Derived decisions

1. **Python 3.12 everywhere.** All engine venvs use the system `python3.12`. Risk: if
   `kraken` or `party` reject 3.12, install `python3.11` via deadsnakes (needs admin) —
   only then. vLLM, TrOCR/transformers are fine on 3.12.
2. **No system CUDA toolkit dependency.** Driver 565 / CUDA 12.7 covers any cu12x wheel.
   Each venv brings its own `torch` (cu12x wheel). Don't link against `/usr/local/cuda`.
3. **GPU placement: default everything to GPU 1.** GPU 0 is shared with a live RAG
   workload — keep our footprint off it. GPU 1 (45 GB) hosts pinned small engines
   (LightOnOCR + TrOCR + kraken + party ≈ 10–11 GB) plus **one** resident 8 B Qwen3-VL
   (≈ 18 GB + KV cache). A second concurrent 8 B only fits by overflowing to GPU 0 — the
   ModelManager must read **live free VRAM via `nvidia-smi`** (not a static budget) and
   use GPU 0 only when it has room. → affects ISSUE #6.
4. **Process supervision without root.** No passwordless sudo and `Linger=no` mean we
   cannot rely on root `systemctl` for the dynamic `atr-vllm@` units. Options, in order:
   - **(recommended) ModelManager spawns vLLM as child subprocesses** (`subprocess.Popen`,
     track PID, health-check the port, `terminate()` on evict). No sudo, no linger.
     → replaces the `atr-vllm@.service` approach in ISSUE #5/#6.
   - Gateway + pinned engines run as **`systemctl --user`** units (request a one-time
     admin `loginctl enable-linger tobias` so they survive logout), or under
     tmux/`nohup` as a fallback.
   - **rootless podman** (quadlet user units) if containerization is preferred later.
5. **Ports.** `:8000` is taken. Use **gateway `:8200`**, engines `:8201` (kraken),
   `:8202` (trocr), `:8203` (party), vLLM instances `:8210+`. All engines bind
   `127.0.0.1`; only the gateway is reachable off-host.
6. **Exposure / firewall.** The box has a routable IP. Bind the gateway to the host and
   add a `ufw` allow rule **scoped to the agentic_historian VM source IP** for `:8200`
   (needs admin once), or reverse-proxy via the existing nginx. API key is mandatory
   regardless. → ISSUE #9; confirm the VM's IP and `ufw status` (needs sudo).
7. **HF cache / disk.** Set `HF_HOME` explicitly (e.g. `/home/tobias/atr-cache/hf`) and
   monitor — `/` is 80 % full with ~356 G free, enough for the planned models
   (~80 G total incl. the Qwen3-VL base) but shared with everything else.
8. **Coexistence.** Ollama (`:11434`), nginx (`:80`), docker and the RAG service are
   already running. Our stack must not grab their ports or GPU 0 memory.

## Open confirmations (need admin / info)
- `ufw status` and which source IP (the agentic_historian VM) must reach `:8200`.
- Whether a one-time `loginctl enable-linger tobias` (or adding `tobias` to `docker`)
  is acceptable — decides supervision style.
- Whether kraken/party install cleanly on Python 3.12.
