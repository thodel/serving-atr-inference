#!/bin/bash
# Experiment E — is the GPU starved? Sweep dataloader workers, watch utilization.
#
# A, B, C and D between them establish a ceiling at 10.8-11.3 samples/s that does
# not move for filesystem, model size, precision or visual-token budget. Every
# change that REMOVES GPU work bought nothing; the one that ADDED it (4-bit
# dequantization) cost 23 %. That is what a waiting GPU looks like.
#
# --workers 4 is the default, on an allocation holding 16 CPUs, with each worker
# decoding a JPEG and running the image processor per sample. ~11 samples/s over
# 4 workers is ~2.8 each, which is a plausible decode-plus-preprocess rate.
#
# This measures the thing directly rather than inferring it again: three worker
# counts, and GPU utilization sampled throughout each. Utilization is the
# discriminator, not throughput - if it sits well under 100 % at workers 4 the
# ceiling is starvation; if it is already pinned, the ceiling is something else
# and the next step is a profiler, not another sweep.
#
# The 4-worker arm reproduces experiment D's 256-token arm (10.80 samples/s), so
# it doubles as a check that this job measures the same thing D did.
#
# NOTE on the sampler: nvidia-smi runs on the HOST here, not through apptainer.
# In experiment B it ran inside the container and $! captured the wrapper's PID,
# so the kill missed and the first arm's sampler wrote through the second arm,
# corrupting that job's VRAM column. Same mistake is not available here.
#
#SBATCH --job-name=vlm-expE
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
export ATR_TRAIN_JOBS_ROOT=$S/expE/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/expE/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/expE/trained
export ATR_TRAIN_ARTEFACT_CACHE_ROOT=$S/expA/artefacts
export PYTHONPATH=$REPO/src:$REPO/engines
export ATR_TRAIN_VENVS_ROOT=/opt
export ATR_TRAIN_GPU=0
mkdir -p "$ATR_TRAIN_JOBS_ROOT" "$ATR_TRAIN_CHECKPOINT_ROOT" "$ATR_TRAIN_TRAINED_ROOT" "$S/expE"

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

echo "== node $(hostname) | CPUs allocated: ${SLURM_CPUS_PER_TASK:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

IDS=()
for w in 4 8 16; do
  echo; echo "######## workers=$w ########"
  JID=$(run python "$HOME/ubelix/submit_job.py" "$HOME/ubelix/specs/expe-w$w.json")
  echo "job $JID"
  IDS+=("$w:$JID")

  nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits \
      -l 5 > "$S/expE/w$w.util" 2>/dev/null &
  SMI=$!
  run python -m vlm_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" --job-id "$JID" \
    || echo "  (workers=$w failed — the record will say why)"
  kill "$SMI" 2>/dev/null || true; wait "$SMI" 2>/dev/null || true
  echo "  samples collected: $(wc -l < "$S/expE/w$w.util")"
done

echo
echo "================ EXPERIMENT E ================"
python3 - "$ATR_TRAIN_JOBS_ROOT" "$S/expE" "${IDS[@]}" <<'PY'
import json, re, statistics as st, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
print(f"{'workers':>7} {'samp/s':>8} {'CER':>8} {'GPU util: median':>17} {'p90':>6} {'max':>5}")
rows = []
for item in sys.argv[3:]:
    w, jid = item.split(":", 1)
    rec = root / jid / "job.json"
    if not rec.exists():
        print(f"{w:>7} (no record)"); continue
    j = json.loads(rec.read_text()); m = j.get("metrics") or {}
    log = root / jid / "logs" / "train.log"
    hits = re.findall(r"'train_runtime':\s*([0-9.]+)", log.read_text(errors="replace")) if log.exists() else []
    tj = root / jid / "data" / "train.jsonl"
    n = sum(1 for _ in tj.open()) if tj.exists() else 0
    rt = float(hits[-1]) if hits else 0.0
    sps = n / rt if rt else 0.0
    # Utilization while the GPU was actually busy: zeros are model load and disk.
    u = [int(l.split(",")[0]) for l in (out / f"w{w}.util").read_text().split("\n")
         if l.strip() and l.split(",")[0].strip().isdigit()]
    busy = [x for x in u if x > 0] or [0]
    med, p90, mx = st.median(busy), sorted(busy)[int(.9*len(busy))-1], max(busy)
    cer = m.get("cer")
    print(f"{w:>7} {sps:>8.2f} {(f'{cer:.4f}' if cer is not None else '-'):>8} "
          f"{med:>16}% {p90:>5}% {mx:>4}%")
    rows.append((int(w), sps, med))
if rows:
    base = rows[0]
    print(f"\nrelative to workers={base[0]}:")
    for w, sps, med in rows:
        print(f"  workers={w:<3} {sps/base[1]:.2f}x throughput, median GPU util {med}%")
    print("\nIf median utilization is well under 100% the GPU is starved and workers")
    print("is the lever. If it is already pinned, the ceiling is elsewhere - stop")
    print("sweeping and profile.")
PY
echo "Experiment D anchor: 4B bf16, 256 tokens, workers 4 = 10.80 samples/s."
echo "=============================================="
