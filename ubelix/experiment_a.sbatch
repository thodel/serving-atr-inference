#!/bin/bash
# Experiment A — is corpus-scale VLM training IO-bound or compute-bound?
#
# VLM_TRAINING.md §5 measures 0.67 samples/s on a 325 K-line corpus against 1.94
# on the 52-page smoke test, names two unseparated causes — 337 K JPEGs read one
# at a time off CIFS, and longer lines costing more visual tokens — and says
# copying the crops to local disk "is the obvious experiment and has not been
# run". Every UBELIX GPU node has 1.92 TB of NVMe at /scratch/local.
#
# DESIGN. Three training passes, same compiled JSONL, same seed, same node, same
# GPU, back to back. Only --data-root differs, because JsonlSamples resolves each
# crop as `root / sample.image` (train_qlora.py:95) — so no repo change is needed
# to move the data.
#
#   A1-cold   /scratch/network (GPFS), first read   <- the runner's own train stage
#   A2        /scratch/local   (node NVMe)
#   A1-warm   /scratch/network again, page-cached
#
# The third pass is the control, and the reason to trust the other two. After
# A1-cold every crop is in page cache, so a naive A1-vs-A2 would partly compare
# cache with cache. (A1-warm - A1-cold) IS the page-cache effect, measured rather
# than assumed; (A2 - A1-warm) is then GPFS against NVMe with both warm.
#
# All three are timed by the SAME clock: HF Trainer's own `train_runtime`, parsed
# from each pass's log. Wall-clocking the passes would fold in model load and
# evaluation, which differ between the runner and a direct invocation.
#
# READ THE RESULT CAREFULLY.
#  * UBELIX shared scratch is parallel GPFS; asterAIx's share is CIFS/SMB. A1
#    here is NOT the asterAIx baseline. If A1-cold already beats 0.67 samples/s
#    by a wide margin, the finding is "the CIFS mount was the problem" and local
#    staging buys little more. That is a good answer, but a different one.
#  * The corpus is the medieval set on purpose: it is the target, so the crop
#    aspect ratios are the ones the real run will see. This measures SECONDS PER
#    SAMPLE only. No CER from this job means anything - the selection is
#    all_projects capped by max_pages, largely Flemish, and is not a quality run.
#
#SBATCH --job-name=vlm-expA
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=24:00:00
# NOT job_gratis: that QoS carries MaxTRESRunMinsPU cpu=11520, so 16 CPUs cap out
# at 12 h and the three passes do not fit. job_gpu_preemptable has no CPU-minute
# limit, and it is the QoS the real campaign will use anyway. The cost is that
# this can be killed at any moment - hence --no-requeue: a silent restart would
# corrupt the timings, and each pass writes its own log so a partial result is
# still readable.
#SBATCH --no-requeue
# 16 CPUs is deliberate and must not be traded away for walltime: dataloader
# workers are part of what this measures.
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail

S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
SIF=$HOME/ubelix/vlm-train.sif
SPEC=${SPEC:-$HOME/ubelix/specs/expA.json}
LOCAL=/scratch/local/$SLURM_JOB_ID

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export ATR_TRAIN_JOBS_ROOT=$S/expA/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/expA/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/expA/trained
export ATR_TRAIN_ARTEFACT_CACHE_ROOT=$S/expA/artefacts
export PYTHONPATH=$REPO/src:$REPO/engines
export ATR_TRAIN_VENVS_ROOT=/opt
export ATR_TRAIN_GPU=0
mkdir -p "$ATR_TRAIN_JOBS_ROOT" "$ATR_TRAIN_CHECKPOINT_ROOT" "$ATR_TRAIN_TRAINED_ROOT" \
         "$ATR_TRAIN_ARTEFACT_CACHE_ROOT" "$S/expA" "$LOCAL"

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

echo "== node $(hostname) $(date -Is)"
run python -c 'import torch; print("gpu:", torch.cuda.get_device_name(0))'
echo "== local scratch:"; df -h /scratch/local | tail -1

JOB_ID=$(run python "$HOME/ubelix/submit_job.py" "$SPEC")
JOB=$ATR_TRAIN_JOBS_ROOT/$JOB_ID
echo "== job $JOB_ID"

# Pass 1: prepare + compile + train. The train stage is A1-cold.
run python -m vlm_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" --job-id "$JOB_ID" || true

N=$(wc -l < "$JOB/data/train.jsonl")
NFILES=$(find "$JOB/data/crops" -type f | wc -l)
BYTES=$(du -sm "$JOB/data/crops" | cut -f1)
echo "== corpus: $N train samples, $NFILES crop files, ${BYTES} MB"

pass() {  # pass <label> <data-root>
  local label=$1 root=$2
  run python -m vlm_train_svc.train_qlora \
      --train-jsonl "$JOB/data/train.jsonl" --val-jsonl "$JOB/data/val.jsonl" \
      --output-dir "$S/expA/out-$label" --data-root "$root" \
      --base-model Qwen/Qwen3-VL-8B-Instruct \
      --prompt "Transcribe the handwritten text in this image exactly as written." \
      --granularity line --max-pixels 262144 --max-seq-len 512 --seed 42 \
      --device cuda:0 --epochs 1 --batch-size 4 --accumulate-grad-batches 4 \
      --load-in-4bit --gradient-checkpointing --workers 4 \
      > "$S/expA/$label.log" 2>&1 || echo "  ($label exited non-zero — see $label.log)"
}

# Staging must reproduce the layout the JSONL encodes: image paths are
# "data/crops/train/N.jpg" RELATIVE TO THE JOB DIR, so the data/ prefix is part
# of the path and copying crops/ alone would resolve to nothing.
echo "== staging $NFILES crops to node NVMe"
t=$SECONDS; mkdir -p "$LOCAL/data"; cp -r "$JOB/data/crops" "$LOCAL/data/crops"
echo "== copy: $((SECONDS-t)) s for ${BYTES} MB"

pass A2      "$LOCAL"
pass A1-warm "$JOB"

echo
echo "================ EXPERIMENT A ================"
python3 "$HOME/ubelix/report_expA.py" "$JOB" "$S/expA" "$N" || true
echo "Compare against VLM_TRAINING.md §5: 0.67 samples/s on asterAIx/CIFS."
echo "=============================================="
rm -rf "$LOCAL"
