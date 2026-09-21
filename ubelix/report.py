#!/usr/bin/env python3
"""Print the outcome of a finished job: status, metrics, error."""
import json
import sys
from pathlib import Path

root, job_id = sys.argv[1], sys.argv[2]
job = json.loads((Path(root) / job_id / "job.json").read_text(encoding="utf-8"))
print("status :", job.get("status"))
print("metrics:", json.dumps(job.get("metrics"), indent=2))
print("error  :", job.get("error"))
