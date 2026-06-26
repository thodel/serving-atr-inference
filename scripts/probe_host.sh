#!/usr/bin/env bash
# probe_host.sh — read-only environment report for the target server (asterAIx @ DH).
#
# Run this ON asterAIx and paste the output back. It changes NOTHING; it only
# reads. The output drives the version pins for each engine venv (torch/CUDA,
# vLLM, kraken, transformers, Python) and the systemd/user setup.
#
#   bash scripts/probe_host.sh            # print to stdout
#   bash scripts/probe_host.sh | tee asteraix-probe.txt   # and save a copy
#
# Safe to run as a normal user. Some sections note where root/sudo would add info.

set -uo pipefail

have() { command -v "$1" >/dev/null 2>&1; }
sec()  { printf '\n══════════════════════════════════════════════════════════════\n## %s\n══════════════════════════════════════════════════════════════\n' "$1"; }
run()  { printf '\n$ %s\n' "$*"; "$@" 2>&1 | sed 's/^/  /'; }
note() { printf '  (note) %s\n' "$*"; }

printf '########## asterAIx host probe — %s ##########\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'host: %s   user: %s\n' "$(hostname 2>/dev/null)" "$(id -un 2>/dev/null)"

# ── OS / kernel ─────────────────────────────────────────────────────────────
sec "OS / kernel"
[ -r /etc/os-release ] && run cat /etc/os-release || note "/etc/os-release missing"
run uname -a
have lsb_release && run lsb_release -a
run bash --version

# ── CPU / RAM / disk ────────────────────────────────────────────────────────
sec "CPU / RAM / disk"
have nproc && run nproc
have lscpu && lscpu 2>/dev/null | grep -E 'Model name|Socket|Core|Thread' | sed 's/^/  /'
have free && run free -h
run df -h /
[ -n "${HOME:-}" ] && run df -h "$HOME"
# Where model weights / HF cache will live — need lots of space (8B models are ~16GB each)
note "HF cache default: ${HF_HOME:-$HOME/.cache/huggingface}  (override with HF_HOME)"
echo "  HF_HOME=${HF_HOME:-<unset>}  HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-<unset>}"

# ── GPU / driver / CUDA ─────────────────────────────────────────────────────
sec "GPU / NVIDIA driver / CUDA"
if have nvidia-smi; then
  run nvidia-smi
  run nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap --format=csv
else
  note "nvidia-smi NOT found — confirm GPU drivers are installed / module loaded"
fi
[ -r /proc/driver/nvidia/version ] && run cat /proc/driver/nvidia/version
if have nvcc; then run nvcc --version; else note "nvcc not on PATH (fine — torch wheels bundle their own CUDA)"; fi
ls -d /usr/local/cuda* 2>/dev/null | sed 's/^/  cuda toolkit: /' || note "no /usr/local/cuda* toolkit"
echo "  CUDA_HOME=${CUDA_HOME:-<unset>}  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

# ── Python interpreters ─────────────────────────────────────────────────────
sec "Python interpreters"
for py in python3 python3.10 python3.11 python3.12 python3.13 python; do
  if have "$py"; then printf '  %-12s -> %s\n' "$py" "$("$py" --version 2>&1)"; fi
done
have python3 && run python3 -c "import sys,platform;print(sys.executable);print(platform.platform())"
have python3 && run python3 -m venv --help >/dev/null 2>&1 && note "python3 -m venv available" || note "venv module may be missing (apt: python3-venv)"
for tool in pip pip3 pipx uv poetry; do have "$tool" && printf '  %-8s -> %s\n' "$tool" "$($tool --version 2>&1 | head -1)"; done

# ── Conda / module systems (common on university servers) ───────────────────
sec "Conda / environment modules / scheduler"
for c in conda mamba micromamba; do have "$c" && run "$c" --version; done
have conda && run conda env list
have module && { note "Lmod/environment-modules present:"; module avail 2>&1 | sed 's/^/  /' | head -60; } || note "no 'module' command"
for s in sinfo squeue srun sbatch; do have "$s" && note "Slurm present ($s) — is this a scheduler-managed node? long-running systemd services may not be allowed"; done

# ── Existing ML stack (any already-installed torch/vllm/kraken) ─────────────
sec "Existing ML packages (system python)"
if have python3; then
  python3 - <<'PY' 2>&1 | sed 's/^/  /'
mods = ["torch","torchvision","transformers","vllm","kraken","accelerate","pillow","fastapi","uvicorn"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:14s} -- not importable ({type(e).__name__})")
try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.version.cuda:", getattr(torch.version,'cuda',None))
    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("  gpu", i, torch.cuda.get_device_name(i))
except Exception as e:
    print("torch cuda check skipped:", e)
PY
fi
have kraken && run kraken --version
have vllm && run vllm --version

# ── systemd / service capability ────────────────────────────────────────────
sec "systemd / service capability"
if have systemctl; then
  run systemctl --version
  note "Can this user manage units? (ModelManager needs to start/stop atr-vllm@ units)"
  systemctl list-units --type=service --state=running 2>/dev/null | grep -iE 'gpustack|vllm|ollama|atr-|nginx|docker' | sed 's/^/  related running: /' || note "no obviously-related services running"
  loginctl show-user "$(id -un)" 2>/dev/null | grep -i linger | sed 's/^/  /' || true
  note "If no root: check 'systemctl --user' (user units) or whether sudo is available."
else
  note "systemctl not found — is this a container? would change the deploy approach"
fi
have sudo && (sudo -n true 2>/dev/null && note "passwordless sudo available" || note "sudo present but needs password / not allowed")

# ── Existing orchestration (GPUStack / Docker / Ollama) ─────────────────────
sec "Existing orchestration"
for t in docker podman nvidia-ctk; do have "$t" && run "$t" --version; done
have docker && run docker ps
have ollama && run ollama list
ss -ltnp 2>/dev/null | sed 's/^/  /' | head -40 || (have netstat && netstat -ltnp 2>/dev/null | sed 's/^/  /' | head -40) || note "no ss/netstat to list listening ports"
note "GPUStack endpoint referenced elsewhere: https://gpustack.unibe.ch/v1 — is asterAIx part of that pool?"

# ── Networking / firewall (for the two-VM API key setup) ────────────────────
sec "Networking / firewall"
run hostname -I
have ip && ip -brief addr 2>/dev/null | sed 's/^/  /'
have ufw && (sudo -n ufw status 2>/dev/null || note "ufw present (status needs sudo)")
have firewall-cmd && (firewall-cmd --state 2>/dev/null | sed 's/^/  /' || note "firewalld present")
note "Confirm: which port can the agentic_historian VM reach on asterAIx?"

# ── Dev tooling ─────────────────────────────────────────────────────────────
sec "Dev tooling"
for t in git gh gcc g++ make cmake rustc; do have "$t" && printf '  %-6s -> %s\n' "$t" "$($t --version 2>&1 | head -1)"; done

printf '\n########## end of probe ##########\n'
