#!/bin/bash
# Experiment D — the visual-token sweep. The lever A and C both pointed at.
#
# A found the workload compute-bound (GPFS, NVMe and page cache within 2 %).
# C found halving the language model bought 1.5 % (4B vs 8B). Between them the
# time is not going into storage and not into the weights: it is going into the
# vision tower. max_pixels is the knob on exactly that — VLM_TRAINING.md §86
# calls it "the single most consequential number in a VLM run", and it converts
# as max_pixels / 1024 visual tokens on Qwen3-VL's 32 px cell.
#
# Four arms, 64 / 128 / 256 / 512 visual tokens. 256 is today's default. 512 is
# included on purpose: knowing whether MORE pixels still buys CER is worth as
# much as knowing where less starts to hurt.
#
# CONTROLLED. Compile output does not depend on max_pixels — the over-long filter
# caps on VLM_MAX_SAMPLE_CHARS[granularity], a per-granularity constant (#110,
# runner.py:84) — so all four arms reuse one compiled corpus and score the same
# 8,668/1,117 split. max_seq_len is raised to 1024 for EVERY arm so that even the
# 512-token arm sits well clear of the cap; a sequence over it is reported, never
# truncated (§86), and an arm that silently dropped samples would not be
# comparable to the others.
#
# Model is Qwen3-VL-4B at bf16 — the production candidate C and B point at, and
# the 256 arm doubles as the confirmation that 4B+bf16 works as a combination,
# which neither B nor C tested. Note it is therefore NOT directly comparable to
# C's 4B row: that ran 4-bit at max_seq_len 512.
#
#SBATCH --job-name=vlm-expD
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail

S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
SIF=$HOME/ubelix/vlm-train.sif

export HF_HOME=/storage/research/wbkolleg_dh_1/Textrecognition_Training/hf_hub
export ATR_TRAIN_JOBS_ROOT=$S/expD/jobs
export ATR_TRAIN_CHECKPOINT_ROOT=$S/expD/checkpoints
export ATR_TRAIN_TRAINED_ROOT=$S/expD/trained
export ATR_TRAIN_ARTEFACT_CACHE_ROOT=$S/expA/artefacts
export PYTHONPATH=$REPO/src:$REPO/engines
export ATR_TRAIN_VENVS_ROOT=/opt
export ATR_TRAIN_GPU=0
mkdir -p "$ATR_TRAIN_JOBS_ROOT" "$ATR_TRAIN_CHECKPOINT_ROOT" "$ATR_TRAIN_TRAINED_ROOT" "$S/expD"

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

echo "== node $(hostname)"
run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"gpu: {p.name} {p.total_memory/1e9:.0f} GB")'

IDS=()
for tok in 64 128 256 512; do
  echo; echo "######## ${tok} visual tokens ########"
  JID=$(run python "$HOME/ubelix/submit_job.py" "$HOME/ubelix/specs/expd-$tok.json")
  echo "job $JID"
  IDS+=("$tok:$JID")
  run python -m vlm_train_svc.runner --root "$ATR_TRAIN_JOBS_ROOT" --job-id "$JID" \
    || echo "  (${tok} failed — the record will say why)"
  # The budget line is printed before the first step; surface it so the knob is
  # provably applied rather than assumed (§86: read-back is not proof it is honoured).
  grep -m1 "visual tokens" "$ATR_TRAIN_JOBS_ROOT/$JID/logs/train.log" 2>/dev/null || true
done

echo
echo "================ EXPERIMENT D ================"
python3 - "$ATR_TRAIN_JOBS_ROOT" "${IDS[@]}" <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
print(f"{'v-tokens':>8} {'status':<11} {'train_runtime':>13} {'samp/s':>8} {'CER':>8} {'WER':>8}")
for item in sys.argv[2:]:
    tok, jid = item.split(":", 1)
    rec = root / jid / "job.json"
    if not rec.exists():
        print(f"{tok:>8} (no record)"); continue
    j = json.loads(rec.read_text()); m = j.get("metrics") or {}
    log = root / jid / "logs" / "train.log"
    hits = re.findall(r"'train_runtime':\s*([0-9.]+)", log.read_text(errors="replace")) if log.exists() else []
    tj = root / jid / "data" / "train.jsonl"
    n = sum(1 for _ in tj.open()) if tj.exists() else 0
    rt = float(hits[-1]) if hits else 0.0
    sps = n / rt if rt else 0.0
    cer = m.get("cer")
    print(f"{tok:>8} {j.get('status','?'):<11} {rt:>12.1f}s {sps:>8.2f} "
          f"{(f'{cer:.4f}' if cer is not None else '-'):>8} "
          f"{(f'{m[chr(119)+chr(101)+chr(114)]:.4f}' if m.get('wer') is not None else '-'):>8}")
    if rt and cer is not None: rows.append((int(tok), sps, cer, n))
if len(rows) > 1:
    rows.sort()
    base = [r for r in rows if r[0] == 256] or [rows[-1]]
    b_sps, b_cer = base[0][1], base[0][2]
    print(f"\nrelative to {base[0][0]} visual tokens:")
    for tok, sps, cer, n in rows:
        print(f"  {tok:>4}: speed {sps/b_sps:5.2f}x   CER {cer/b_cer:5.2f}x"
              f"   ({'cheaper and no worse' if sps > b_sps and cer <= b_cer*1.02 else ''})")
    print("\nSame corpus and split in every arm - the over-long filter caps on")
    print("characters, not pixels, so max_pixels cannot have moved the sample set.")
PY
echo "=============================================="
