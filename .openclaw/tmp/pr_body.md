## What

Implements issue #46: verify a DatasetSpec against the hub **before** a job is queued, so that misconfigured specs fail fast with HTTP 400 instead of deep in the prepare stage.

## Changes

### `verify_dataset_spec` in `atr_serving/training/hf_source.py`
New function that runs four cheap public checks:
1. **Repo existence** — `huggingface_hub.list_repo_files()` with optional revision
2. **Project directories** — checks that `data/<split>/<project>/*` paths exist in the repo
3. **Parquet presence** — verifies the dataset uses the pagexml-hf converter layout
4. **Disk size estimate** — samples up to 20 parquet files to estimate total download vs `min_free_disk_gb`

Returns `list[str]` of human-readable errors (empty = valid). Raises `DatasetSelectionError` for structural problems (empty projects, ambiguous names — same class used by `data_files_for`).

All network I/O is behind a seam (`list_repo_files_fn` / `hf_hf_file_download_fn`) so tests can inject fakes without touching the network.

### `POST /jobs/verify` in the trainer service
New endpoint that calls `verify_dataset_spec` and returns `{valid: bool, errors: list[str]}`. A `verify_only=true` query param causes HTTP 400 on an invalid spec; without it, failures are included in the normal submit response.

### `POST /train/jobs?verify_only=true` in the gateway proxy
New query param on the existing submit route. When set, verifies the dataset spec via the trainer service without queueing the job. Connection errors to the trainer service return HTTP 502.

### `TrainerClient.verify()` in `atr_serving/clients.py`
New async method that calls `POST /jobs/verify` with the request body.

### Tests
- Full unit-test suite for `verify_dataset_spec` with faked network calls (repo-not-found, missing project, valid spec, no parquet files, structural validation, disk-size warning, error aggregation, revision passthrough)
- Updated proxy route tests to reflect the new call order (verify → submit)

## After merge

Close #46.