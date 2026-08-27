#!/usr/bin/env bash
# Check UBELIX training progress from the laptop. Read-only.
#
#   ./ubelix/status.sh              one snapshot
#   ./ubelix/status.sh -f           follow the newest job's log
#   ./ubelix/status.sh -n 60        tail 60 log lines instead of 25
#   ./ubelix/status.sh -j 14108981  a specific job
#
# Reaches the cluster through the `ubelix` ssh alias (ProxyJump via asterAIx),
# so it needs no VPN. See docs/UBELIX_PLAN.md §0.
set -uo pipefail

HOST=${UBELIX_HOST:-ubelix}
LINES=25
FOLLOW=0
JOB=""

while getopts "fn:j:h" opt; do
  case $opt in
    f) FOLLOW=1 ;;
    n) LINES=$OPTARG ;;
    j) JOB=$OPTARG ;;
    h) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) exit 2 ;;
  esac
done

if [ "$FOLLOW" = 1 ]; then
  # -t so Ctrl-C reaches the remote tail instead of orphaning it.
  exec ssh -t "$HOST" "L=\$(ls -t \$HOME/ubelix/logs/*-*.out 2>/dev/null | head -1); \
    if [ -z \"\$L\" ]; then echo 'no job logs yet'; exit 1; fi; \
    echo \"== following \$L\"; tail -f \"\$L\""
fi

ssh "$HOST" JOB="$JOB" LINES="$LINES" 'bash -s' <<'REMOTE'
set -uo pipefail
bar() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

bar "queue"
q=$(squeue -u "$USER" -o "%.10i %.14j %.9T %.8M %.10L %.9P %R" 2>/dev/null)
[ "$(echo "$q" | wc -l)" -gt 1 ] && echo "$q" || echo "  (nothing queued or running)"

bar "recent jobs"
sacct -u "$USER" --starttime now-3days \
      --format=JobID%12,JobName%16,State%12,Elapsed%10,End%16 -X 2>/dev/null | head -12

bar "quota"
quota 2>/dev/null | tail -3
printf '  scratch : '
du -sh "/scratch/network/users/$USER" 2>/dev/null | cut -f1

L=""
if [ -n "${JOB:-}" ]; then
  L=$(ls -t "$HOME"/ubelix/logs/*-"$JOB".out 2>/dev/null | head -1)
  [ -z "$L" ] && echo "  (no log for job $JOB)"
else
  L=$(ls -t "$HOME"/ubelix/logs/*-*.out 2>/dev/null | head -1)
fi

if [ -n "$L" ]; then
  bar "log: $(basename "$L")"
  # Drop the per-line cropping spam — hundreds of lines that say nothing.
  grep -v "cropping:write_crops" "$L" | tail -n "${LINES:-25}"

  bar "metrics"
  jid=$(basename "$L" .out); jid=${jid##*-}
  # The runner writes job.json under whichever jobs root the sbatch set.
  found=0
  for root in /scratch/network/users/"$USER"/*/jobs; do
    for j in "$root"/*/job.json; do
      [ -e "$j" ] || continue
      found=1
      python3 - "$j" <<'PY'
import json, sys
from pathlib import Path
j = json.loads(Path(sys.argv[1]).read_text())
m = j.get("metrics") or {}
line = f"  {j['id']}  [{j.get('status')}]"
if m.get("cer") is not None:
    line += f"  CER {m['cer']:.4f}  WER {m.get('wer', float('nan')):.4f}  n={m.get('samples')}"
if j.get("error"):
    line += f"  error: {j['error']}"
print(line)
PY
    done
  done
  [ "$found" = 0 ] && echo "  (no job records yet)"
fi
echo
REMOTE
