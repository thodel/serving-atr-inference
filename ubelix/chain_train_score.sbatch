#!/bin/bash
# After a prepare: queue its training (gpu/job_gratis) and then the held-out scoring.
#   sbatch --dependency=afterok:<P> --export=ALL,PREP_SLURM=<P>,TAG=<tag> chain_train_score.sbatch
#SBATCH --job-name=chain
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/storage/homefs/%u/ubelix/logs/%x-%j.out
set -euo pipefail
S=/scratch/network/users/$USER
: "${PREP_SLURM:?}" "${TAG:?}"
JOB_ID=$(cat "$S/runs/prep-$PREP_SLURM.jobid")
echo "== prepared: $JOB_ID"
# job_gratis caps CPUs x walltime at MaxTRESRunMinsPU cpu=11520 — for GPU jobs too.
# 16 cpus x 24 h = 23,040 was rejected at submission ("MaxCpuRunMinsPerUser"),
# which killed this chain after a 16 h prepare. 12 x 15 h = 10,800 fits, and 8
# dataloader workers do not need 16 cores.
T=$(sbatch --parsable --partition=gpu --qos=job_gratis --cpus-per-task=12 --time=15:00:00 \
      --job-name=${TRAIN_NAME:-train} --export=ALL,JOB_ID=$JOB_ID "$HOME/ubelix/train.sbatch")
echo "== train: $T"
E=$(sbatch --parsable --dependency=afterok:$T \
      --export=ALL,JOB_ID=$JOB_ID,TAG=$TAG "$HOME/ubelix/score_federal_minutes.sbatch")
echo "== score: $E"
echo "$JOB_ID $T $E" > "$S/runs/chain-$PREP_SLURM.ids"
