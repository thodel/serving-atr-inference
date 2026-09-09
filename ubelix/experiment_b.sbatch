#!/bin/bash
# Experiment B — does dropping 4-bit NF4 buy throughput on a card we own outright?
#
# The VLM trainer defaults to load_in_4bit because asterAIx shares its A40 with
# the serving engines; a bf16 8B does not fit beside them (VLM_TRAINING.md). That
# constraint does not exist on a 94 GB H100. The 2026-08-08 run also logged
# `bitsandbytes: inner dimension (4304) is not aligned for fast kernel` — Qwen3-VL's
# dimensions miss the fast 4-bit path, so 4-bit here may be paying dequantization
# cost for a memory saving nobody needs.
#
# ONE variable. Both passes reuse experiment A's compiled corpus, same seed, same
# node, same argv — only --load-in-4bit / --no-load-in-4bit differs. B1 re-measures
# the 4-bit baseline inside this job rather than comparing against experiment A's
# number across jobs and nodes; that is the same discipline the A1-warm control
# served, and it costs 20 minutes.
#
# Gradient checkpointing stays ON in both. Turning it off would likely be faster
# in bf16, but it is a second variable and belongs in its own run.
#
#SBATCH --job-name=vlm-expB
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=6:00:00
#SBATCH --no-requeue
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail

S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
SIF=$HOME/ubelix/vlm-train.sif
JOB_ID=${JOB_ID:-20260909T180141Z-expa-io-probe}
JOB=$S/expA/jobs/$JOB_ID

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export PYTHONPATH=$REPO/src:$REPO/engines

run() {
  apptainer exec --nv \
    --bind /storage/research:/storage/research \
    --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
    --env HF_HOME="$HF_HOME" --env PYTHONPATH="$PYTHONPATH" \
    "$SIF" "$@"
}

[ -f "$JOB/data/train.jsonl" ] || { echo "no compiled corpus at $JOB" >&2; exit 1; }
N=$(wc -l < "$JOB/data/train.jsonl")
echo "== node $(hostname) | corpus $N samples | reusing $JOB_ID"
run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"gpu: {p.name} {p.total_memory/1e9:.0f} GB")'

# Verbatim from the runner's train.log, minus the quantization flag.
COMMON=(
  --base-model Qwen/Qwen3-VL-8B-Instruct
  --prompt "Transcribe the handwritten text in this image exactly as written."
  --granularity line --max-pixels 262144 --max-seq-len 512 --seed 42 --device cuda:0
  --epochs 1 --max-epochs 1 --patience 2 --min-delta 0.0001
  --batch-size 4 --accumulate-grad-batches 4
  --lrate 0.0002 --lr-scheduler cosine --warmup-ratio 0.05
  --weight-decay 0.0 --max-grad-norm 1.0 --optim paged_adamw_8bit
  --lora-r 64 --lora-alpha 128 --lora-dropout 0.05
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --workers 4 --gradient-checkpointing
)

pass() {  # pass <label> <quant-flag>
  local label=$1 quant=$2
  echo "== $label ($quant)"
  ( run nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 60 > "$S/expA/$label.vram" 2>/dev/null & echo $! > /tmp/smi.$$ ) || true
  run python -m vlm_train_svc.train_qlora \
      --train-jsonl "$JOB/data/train.jsonl" --val-jsonl "$JOB/data/val.jsonl" \
      --output-dir "$S/expA/out-$label" --data-root "$JOB" \
      "${COMMON[@]}" "$quant" > "$S/expA/$label.log" 2>&1 \
    || echo "   FAILED — see $S/expA/$label.log"
  kill "$(cat /tmp/smi.$$)" 2>/dev/null || true; rm -f /tmp/smi.$$
  echo "   peak VRAM: $(sort -n "$S/expA/$label.vram" 2>/dev/null | tail -1)"
}

pass B1-4bit "--load-in-4bit"
pass B2-bf16 "--no-load-in-4bit"

echo
echo "================ EXPERIMENT B ================"
python3 - "$S/expA" "$N" <<'PY'
import re, sys
from pathlib import Path
out, n = Path(sys.argv[1]), int(sys.argv[2])
PAT = re.compile(r"'train_runtime':\s*([0-9.]+)")
res = {}
print(f"{'pass':<10} {'train_runtime':>14} {'samples/s':>10}   peak VRAM")
for label, what in [("B1-4bit", "NF4 4-bit"), ("B2-bf16", "bf16")]:
    log, vram = out / f"{label}.log", out / f"{label}.vram"
    hits = PAT.findall(log.read_text(errors="replace")) if log.exists() else []
    peak = ""
    if vram.exists():
        vals = [int(x.split()[0]) for x in vram.read_text().split("\n") if x.strip().split()[:1] and x.split()[0].isdigit()]
        peak = f"{max(vals)/1024:.1f} GB" if vals else ""
    if not hits:
        print(f"{label:<10} {'(no runtime)':>14} {'-':>10}   {peak}  <- {what}")
        continue
    rt = float(hits[-1]); res[label] = rt
    print(f"{label:<10} {rt:>13.1f}s {n/rt:>10.2f}   {peak}  <- {what}")
if len(res) == 2:
    d = res["B1-4bit"] - res["B2-bf16"]
    print(f"\nbf16 vs 4-bit: {d:+.1f}s ({100*d/res['B1-4bit']:+.1f}%)")
    print("VERDICT: bf16 is the default for UBELIX." if d > 0.03*res["B1-4bit"]
          else "VERDICT: no material gain - keep 4-bit, the memory is free anyway.")
PY
echo "Experiment A baseline on this corpus: 8.19-8.36 samples/s, 4-bit."
echo "=============================================="
