#!/usr/bin/env bash
# Validate a job spec on the LOGIN node, then sbatch. Run this instead of sbatch.
#
#   ./submit.sh experiment_a.sbatch [specs/expA.json]
#
# Why: submit_job.py validates the request, but it runs inside the batch job — so
# a bad spec costs a queue wait and a GPU allocation before anything says so.
# Job 14431367 died 13 s in because a model_id had a capital letter. Validation
# needs no GPU and no queue, so it belongs here.
set -euo pipefail

SBATCH_FILE=${1:?usage: submit.sh <file.sbatch> [spec.json]}
SPEC=${2:-}
UB=$HOME/ubelix
REPO=$HOME/serving-atr-inference

# Default to the SPEC= the sbatch file itself names.
if [ -z "$SPEC" ]; then
  SPEC=$(sed -n 's/^SPEC=${SPEC:-\(.*\)}$/\1/p' "$SBATCH_FILE" | head -1)
  SPEC=${SPEC/\$HOME/$HOME}
fi
[ -n "$SPEC" ] && [ -f "$SPEC" ] || { echo "no spec found (pass one explicitly)" >&2; exit 2; }

echo "validating $SPEC"
apptainer exec --bind /storage/research --bind /scratch --bind /rs_scratch \
  --env PYTHONPATH="$REPO/src:$REPO/engines" \
  "$UB/vlm-train.sif" python - "$SPEC" <<'PY'
import json, sys
from atr_serving.training.contracts import TrainRequest
req = TrainRequest.model_validate(json.load(open(sys.argv[1])))
print(f"  OK  {req.model_id}  engine={req.engine}")
for d in req.datasets:
    print(f"      {d.hf_repo}  all_projects={d.all_projects} "
          f"max_pages={d.max_pages} partition={d.partition}")
PY

bash -n "$SBATCH_FILE"
echo "submitting $SBATCH_FILE"
sbatch "$SBATCH_FILE"
