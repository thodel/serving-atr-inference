#!/bin/bash
# Experiment F — the 2x2. What is the GPU waiting for?
#
# E established the shape: 54 % median GPU utilization means the card had nothing
# to run for 46 % of the wall clock. That is a stall, not slow work, and it is
# not the dataloader (4/8/16 workers moved it 2 %).
#
# Two candidates, tested together so they discriminate instead of confounding:
#
#   optimizer   paged_adamw_8bit is bitsandbytes' PAGED optimizer, which moves
#               state to host memory to free VRAM we are not short of - 54 GB
#               sits idle. Every optimizer step is then a synchronous host
#               transfer with the GPU stalled through it.
#   micro-batch batch_size 4 makes each kernel launch small relative to its
#               fixed overhead.
#
# EFFECTIVE BATCH IS 16 IN EVERY ARM - bs4/accum4 and bs16/accum1. That holds the
# NUMBER of optimizer steps constant across all four. Without it, the bs16 arms
# would take a quarter as many optimizer steps and would look faster for a reason
# that has nothing to do with micro-batch size, which would confound the very
# comparison this job exists to make.
#
# workers is fixed at 8 (E: indistinguishable from 4 and 16, so it is not a
# variable; 8 just leaves headroom for the bs16 arms).
#
# IF THIS COMES BACK FLAT, THE NEXT STEP IS torch.profiler ON ONE ARM, NOT A
# SEVENTH SWEEP. Two hypotheses have already died to throughput-number guessing
# (max_pixels in D, workers in E) and a third would be a pattern, not bad luck.
#
#SBATCH --job-name=vlm-expF
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

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export ATR_TRAIN_JOBS_ROOT=$S/expF/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/expF/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/expF/trained
export ATR_TRAIN_ARTEFACT_CACHE_ROOT=$S/expA/artefacts
export PYTHONPATH=$REPO/src:$REPO/engines
export ATR_TRAIN_VENVS_ROOT=/opt
export ATR_TRAIN_GPU=0
mkdir -p "$ATR_TRAIN_JOBS_ROOT" "$ATR_TRAIN_CHECKPOINT_ROOT" "$ATR_TRAIN_TRAINED_ROOT" "$S/expF"

run() {
  apptainer exec --nv \
    --bind /storage/research:/storage/research \
    --bind /scratch:/scratch --bind /rs_scratch:/rs_scratch \
    --env HF_HOME="$HF_HOME" \
    --env ATR_TRAIN_JOBS_ROOT="$ATR_TRAIN_JOBS_ROOT" \
    --env ATR_TRAIN_CHECKPOINT_ROOT="$ATR_TRAIN_CHECKPOINT_ROOT" \
    --env ATR_TRAIN_TRAINED_ROOT="$ATR_TRAIN_TRAINED_ROOT" \
    --env ATR_TRAIN_ARTEFACT_CACHE_ROOT="$ATR_TRAIN_ARTEFACT_CACHE_ROOT" \
    --env PYTHONPATH="$PYTHONPATH" --env ATR_TRAIN_VENVS_ROOT=/opt --env ATR_TRAIN_GPU=0 \
    "$SIF" "$@"
}

echo "== node $(hostname) | CPUs ${SLURM_CPUS_PER_TASK:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

IDS=()
for tag in paged-b4 torch-b4 paged-b16 torch-b16; do
  echo; echo "######## $tag ########"
  JID=$(run python "$HOME/ubelix/submit_job.py" "$HOME/ubelix/specs/expf-$tag.json")
  echo "job $JID"
  IDS+=("$tag:$JID")
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
      -l 5 > "$S/expF/$tag.util" 2>/dev/null &
  SMI=$!
  run python -m vlm_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" --job-id "$JID" \
    || echo "  ($tag failed — the record will say why)"
  kill "$SMI" 2>/dev/null || true; wait "$SMI" 2>/dev/null || true
done

echo
echo "================ EXPERIMENT F ================"
python3 - "$ATR_TRAIN_JOBS_ROOT" "$S/expF" "${IDS[@]}" <<'PY'
import json, re, statistics as st, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
print(f"{'arm':<11} {'optim':<18} {'bs':>3} {'samp/s':>8} {'CER':>8} {'util med':>9} {'VRAM':>8}")
rows = {}
for item in sys.argv[3:]:
    tag, jid = item.split(":", 1)
    rec = root / jid / "job.json"
    if not rec.exists():
        print(f"{tag:<11} (no record)"); continue
    j = json.loads(rec.read_text()); m = j.get("metrics") or {}
    pr = j["request"]["params"]
    log = root / jid / "logs" / "train.log"
    hits = re.findall(r"'train_runtime':\s*([0-9.]+)", log.read_text(errors="replace")) if log.exists() else []
    tj = root / jid / "data" / "train.jsonl"
    n = sum(1 for _ in tj.open()) if tj.exists() else 0
    rt = float(hits[-1]) if hits else 0.0
    sps = n / rt if rt else 0.0
    uf = out / f"{tag}.util"
    us, vs = [], []
    if uf.exists():
        for line in uf.read_text().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                us.append(int(parts[0])); vs.append(int(parts[1]))
    busy = [x for x in us if x > 0] or [0]
    print(f"{tag:<11} {pr['optim']:<18} {pr['batch_size']:>3} {sps:>8.2f} "
          f"{(f\"{m['cer']:.4f}\" if m.get('cer') is not None else '-'):>8} "
          f"{st.median(busy):>8}% {(max(vs)/1024 if vs else 0):>7.1f}G")
    rows[tag] = (sps, st.median(busy))
if len(rows) == 4:
    b = rows["paged-b4"][0]
    print(f"\nrelative to paged-b4 (E's configuration, {b:.2f} samp/s):")
    for tag in ("torch-b4", "paged-b16", "torch-b16"):
        print(f"  {tag:<11} {rows[tag][0]/b:5.2f}x   util {rows[tag][1]}%")
    opt_eff = (rows["torch-b4"][0]/rows["paged-b4"][0] + rows["torch-b16"][0]/rows["paged-b16"][0]) / 2
    bs_eff  = (rows["paged-b16"][0]/rows["paged-b4"][0] + rows["torch-b16"][0]/rows["torch-b4"][0]) / 2
    print(f"\nmain effect, optimizer (paged -> torch): {opt_eff:5.2f}x")
    print(f"main effect, micro-batch (4 -> 16):      {bs_eff:5.2f}x")
    if opt_eff < 1.05 and bs_eff < 1.05:
        print("\nBOTH FLAT. Stop sweeping. Next step is torch.profiler on one arm.")
PY
echo "E baseline: 4B bf16, 256 tokens, workers 8 = 11.10 samp/s at 54% util."
echo "=============================================="
