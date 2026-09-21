#!/bin/bash
# After a prepare: fan the corpus out to several base models and queue their training.
#
# Runs on a CPU node with --dependency=afterok:<prepare>, so a whole
# prepare -> N parallel trainings pipeline can be queued in one go and left
# alone. It needs no GPU: fanout.py only writes job records and symlinks.
#
#   sbatch --dependency=afterok:<P> --export=ALL,PREP_SLURM=<P>,ARMS="id=base id=base" \
#          ubelix/fanout_submit.sbatch
#
# The prepared job itself trains too, as the anchor arm, on its own base_model.
#
#SBATCH --job-name=fanout
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out

set -euo pipefail
S=/scratch/network/users/$USER
REPO=$HOME/serving-atr-inference
TF5=$HOME/ubelix/vlm-train-tf5.sif
: "${PREP_SLURM:?set PREP_SLURM=<slurm id of the prepare job>}"
: "${ARMS:?set ARMS=\"model_id=base_model ...\"}"
PREP=$(cat "$S/runs/prep-$PREP_SLURM.jobid")
echo "== prepared job: $PREP"

run() {
  apptainer exec --bind /storage/research --bind /scratch --bind /rs_scratch \
    --env PYTHONPATH="$REPO/src:$REPO/engines" "$TF5" "$@"
}
# shellcheck disable=SC2086
CLONES=$(run python "$HOME/ubelix/fanout.py" "$S/runs/jobs" "$PREP" $ARMS)
echo "$CLONES" > "$S/runs/fanout-$PREP_SLURM.ids"

sub() {  # sub <job_id>
  local name; name=$(echo "$1" | sed 's/^[0-9TZ]*-//' | cut -c1-14)
  sbatch --export=ALL,JOB_ID="$1",SIF="$TF5" --job-name="$name" \
         --gres=gpu:h100:1 --cpus-per-task=16 --mem=90G \
         "$HOME/ubelix/train.sbatch" | sed "s/^/  $name: /"
}
echo "== queueing training arms"
sub "$PREP"
for c in $CLONES; do sub "$c"; done
