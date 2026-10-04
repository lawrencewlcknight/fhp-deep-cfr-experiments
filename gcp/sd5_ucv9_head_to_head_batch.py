#!/usr/bin/env python3
"""One VM, two read-only source buckets, a separate resumable analysis prefix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone

SD_REPO = "https://github.com/lawrencewlcknight/fhp-deep-cfr-experiments.git"
UCV_REPO = "https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git"
DEFAULT_UCV_REF = "1b61ebcabf3f3865ba32430b388fdbdf6d34297a"
MODULE = "experiments.fhp.retrospective_sd5_ucv9_evaluation.run"


def validate(args):
    for key in ("run_id", "sd_run_id", "ucv_run_id"):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,45}[a-z0-9]", getattr(args, key)):
            raise ValueError(f"{key} must be a 2–47 character lowercase Batch-compatible ID")
    if args.run_id in {args.sd_run_id, args.ucv_run_id}:
        raise ValueError("Evaluation must use a new output prefix, never a source run")
    for key in ("bucket", "sd_bucket", "ucv_bucket"):
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", getattr(args, key)):
            raise ValueError(f"{key} must be a bucket URI without a subdirectory")
    for key in ("repo_ref", "ucv_ref"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(args, key)):
            raise ValueError(f"{key} must be a full pushed commit SHA")
    if not 0 < args.max_hours <= 12:
        raise ValueError("Safety budget must be positive and at most 12 hours")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.iam\.gserviceaccount\.com", args.service_account):
        raise ValueError("Set the SD-CFR runner service-account email")


def script(args):
    q = shlex.quote
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/sd5-ucv9-mpl UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
WORK=/workspace/sd5-ucv9
SD_REPOSITORY="$WORK/sd-repository"
UCV_REPOSITORY="$WORK/ucv-repository"
INPUT="$WORK/input"
OUTPUT="$WORK/output"
DESTINATION={q(args.bucket + '/' + args.run_id)}
SD_SOURCE={q(args.sd_bucket + '/' + args.sd_run_id)}
UCV_SOURCE={q(args.ucv_bucket + '/' + args.ucv_run_id)}
SYNC_PID=""
mkdir -p "$INPUT/sd/workers" "$INPUT/ucv/workers" "$OUTPUT/analysis"
sync_outputs() {{
  gcloud storage rsync --recursive --exclude='.*[.]tmp$' "$OUTPUT" "$DESTINATION"
}}
cleanup() {{
  result=$?
  trap - EXIT
  if [[ -n "$SYNC_PID" ]]; then kill "$SYNC_PID" 2>/dev/null || true; wait "$SYNC_PID" 2>/dev/null || true; fi
  if ! sync_outputs; then
    echo "Final result upload failed" >&2
    if [[ "$result" -eq 0 ]]; then result=1; fi
  fi
  exit "$result"
}}
trap cleanup EXIT
apt-get update -qq
apt-get install -y -qq git curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone {q(SD_REPO)} "$SD_REPOSITORY"
git -C "$SD_REPOSITORY" checkout --detach {q(args.repo_ref)}
git clone {q(UCV_REPO)} "$UCV_REPOSITORY"
git -C "$UCV_REPOSITORY" checkout --detach {q(args.ucv_ref)}
uv python install 3.11.17
uv venv --python 3.11.17 "$WORK/venv"
uv pip install --python "$WORK/venv/bin/python" -r "$SD_REPOSITORY/requirements.txt"
# Do not install the UCV distribution: both repos own `experiments` and
# `fhp_evaluation`. Only the unique UCV policy-loader modules are appended.
export PYTHONPATH="$SD_REPOSITORY"
cd "$SD_REPOSITORY"
if [[ {1 if args.resume else 0} -eq 1 ]]; then
  gcloud storage rsync --recursive "$DESTINATION" "$OUTPUT"
fi
# This preflight runs as the VM service account before the policy download.
"$WORK/venv/bin/python" gcp/sd5_ucv9_head_to_head_batch.py check-sources \
  --project {q(args.project)} --region {q(args.region)} --service-account {q(args.service_account)} \
  --run-id {q(args.run_id)} --bucket {q(args.bucket)} --sd-bucket {q(args.sd_bucket)} \
  --ucv-bucket {q(args.ucv_bucket)} --sd-run-id {q(args.sd_run_id)} --ucv-run-id {q(args.ucv_run_id)} \
  --repo-ref {q(args.repo_ref)} --ucv-ref {q(args.ucv_ref)}
for seed in 0 1 2; do
  SD_WORKER="task_00${{seed}}_parallel_structured_uniform_sd_cfr_seed_${{seed}}"
  UCV_WORKER="task_00${{seed}}_cached_parallel_structured_ucv_escher_seed_${{seed}}"
  mkdir -p "$INPUT/sd/workers/$SD_WORKER/archive" "$INPUT/ucv/workers/$UCV_WORKER/checkpoints"
  for metadata in run_manifest.json checkpoint_manifest.json SUCCESS.json; do
    gcloud storage cp "$SD_SOURCE/workers/$SD_WORKER/$metadata" "$INPUT/sd/workers/$SD_WORKER/$metadata"
    gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/$metadata" "$INPUT/ucv/workers/$UCV_WORKER/$metadata"
  done
  # Full historical SD strategies are playable policy data, not replay states.
  gcloud storage rsync --recursive "$SD_SOURCE/workers/$SD_WORKER/archive" "$INPUT/sd/workers/$SD_WORKER/archive"
  gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/runtime_manifest.json" "$INPUT/ucv/workers/$UCV_WORKER/runtime_manifest.json"
  for hour in 06 12 18 24; do
    file="cached_parallel_structured_ucv_escher_seed_${{seed}}_time_${{hour}}h.pkl"
    gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/checkpoints/$file" "$INPUT/ucv/workers/$UCV_WORKER/checkpoints/$file"
  done
done
(while sleep 300; do sync_outputs || echo "Periodic upload failed; will retry" >&2; done) &
SYNC_PID=$!
ARGS=(--sd-root "$INPUT/sd" --ucv-root "$INPUT/ucv" --ucv-repo "$UCV_REPOSITORY"
      --sd-source-uri "$SD_SOURCE" --ucv-source-uri "$UCV_SOURCE"
      --output "$OUTPUT/analysis" --workers 8 --max-hours {args.max_hours})
"$WORK/venv/bin/python" -m {MODULE} "${{ARGS[@]}}" --stage smoke
sync_outputs
# Full run repeats source validation, then runs the real-checkpoint cost pilot.
# If the conservative estimate exceeds the limit it fails before main matches.
"$WORK/venv/bin/python" -m {MODULE} "${{ARGS[@]}}" --stage run
'''


def job_config(args):
    validate(args)
    return dict(taskGroups=[dict(taskCount=1, parallelism=1, taskSpec=dict(
        runnables=[dict(script=dict(text=script(args)))], computeResource=dict(cpuMilli=8000, memoryMib=30000),
        maxRetryCount=0, maxRunDuration=f"{int((args.max_hours + 2) * 3600)}s"))],
        allocationPolicy=dict(instances=[dict(policy=dict(machineType="n2-standard-8", provisioningModel="STANDARD",
            bootDisk=dict(image="projects/batch-custom-image/global/images/family/batch-debian", sizeGb=200, type="pd-balanced")))],
            serviceAccount=dict(email=args.service_account, scopes=["https://www.googleapis.com/auth/cloud-platform"])),
        logsPolicy=dict(destination="CLOUD_LOGGING"), labels=dict(workload="sd5-ucv9-h2h", run=args.run_id))


def cloud(args, *command, check=True):
    return subprocess.run(["gcloud", *command, "--project", args.project], text=True, capture_output=True, check=check)


def objects_exist(args, uri):
    result = cloud(args, "storage", "ls", uri, check=False)
    if result.returncode == 0:
        return bool(result.stdout.strip())
    if "matched no objects" in result.stderr:
        return False
    raise RuntimeError(result.stderr.strip())


def check_sources(args):
    for bucket, run, algorithm in ((args.sd_bucket, args.sd_run_id, "parallel_structured_uniform_sd_cfr"),
                                    (args.ucv_bucket, args.ucv_run_id, "cached_parallel_structured_ucv_escher")):
        for seed in range(3):
            prefix = f"{bucket}/{run}/workers/task_{seed:03d}_{algorithm}_seed_{seed}"
            for name in ("run_manifest.json", "checkpoint_manifest.json", "SUCCESS.json"):
                if not objects_exist(args, f"{prefix}/{name}"):
                    raise ValueError(f"Missing source {prefix}/{name}; check the correct algorithm's bucket")
            if objects_exist(args, f"{prefix}/FAILURE.json"):
                raise ValueError(f"Failed source worker: {prefix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "resume", "dry-run", "status", "check-sources"))
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID"), required=not os.environ.get("PROJECT_ID"))
    parser.add_argument("--region", default=os.environ.get("REGION", "europe-west1"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"), required=not os.environ.get("RUN_ID"))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET", "gs://clever-overview-399515-fhp-deep-cfr-results"))
    parser.add_argument("--sd-bucket", default=os.environ.get("SD_BUCKET", "gs://clever-overview-399515-fhp-deep-cfr-results"))
    parser.add_argument("--ucv-bucket", default=os.environ.get("UCV_BUCKET", "gs://clever-overview-399515-fhp-escher-results"))
    parser.add_argument("--sd-run-id", default=os.environ.get("SD_EXP5_RUN_ID", "sdcfr5-par8-20261002-102757"))
    parser.add_argument("--ucv-run-id", default=os.environ.get("UCV_EXP9_RUN_ID", "exp9-cache24-20261001-132550"))
    parser.add_argument("--repo-ref", default=os.environ.get("REPO_REF", ""))
    parser.add_argument("--ucv-ref", default=os.environ.get("UCV_REPO_REF", DEFAULT_UCV_REF))
    parser.add_argument("--service-account", default=os.environ.get("SA_EMAIL", "fhp-deep-cfr-runner@clever-overview-399515.iam.gserviceaccount.com"))
    parser.add_argument("--max-hours", type=float, default=float(os.environ.get("EVAL_MAX_HOURS", "12")))
    parser.add_argument("--output", type=Path, help="JSON location for dry-run")
    args = parser.parse_args()
    for name in ("bucket", "sd_bucket", "ucv_bucket"):
        setattr(args, name, "gs://" + getattr(args, name).removeprefix("gs://").rstrip("/"))
    args.resume = args.action == "resume"
    if args.action == "status":
        print(cloud(args, "batch", "jobs", "list", "--location", args.region,
                    "--filter", f"labels.workload=sd5-ucv9-h2h AND labels.run={args.run_id}").stdout)
        return
    validate(args)
    if args.action == "check-sources":
        check_sources(args)
        return
    config = job_config(args)
    if args.action == "dry-run":
        if not args.output:
            parser.error("Specify --output for dry-run")
        args.output.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Wrote {args.output}; no job submitted")
        return
    check_sources(args)
    cloud(args, "iam", "service-accounts", "describe", args.service_account)
    active = cloud(args, "batch", "jobs", "list", "--location", args.region,
                   "--filter", f"labels.workload=sd5-ucv9-h2h AND labels.run={args.run_id}", "--format=json")
    if any(j["status"]["state"] not in {"SUCCEEDED", "FAILED"} for j in json.loads(active.stdout)):
        raise ValueError("A job already uses this output prefix; do not run concurrently")
    exists = objects_exist(args, f"{args.bucket}/{args.run_id}/**")
    if exists and not args.resume:
        raise ValueError("Outputs already exist; use resume with the same code and sources")
    if args.resume and not objects_exist(args, f"{args.bucket}/{args.run_id}/analysis/evaluation_manifest.json"):
        raise ValueError("No evaluation manifest to resume")
    job_name = args.run_id + (datetime.now(timezone.utc).strftime("-r%H%M%S") if args.resume else "")
    with tempfile.TemporaryDirectory(prefix="sd5-ucv9-submit-") as temporary:
        file = Path(temporary) / "job.json"
        file.write_text(json.dumps(config))
        print(cloud(args, "batch", "jobs", "submit", job_name, "--location", args.region, "--config", str(file)).stdout)
    print(f"Outputs: {args.bucket}/{args.run_id}/analysis/ (one VM; no training or LBR)")


if __name__ == "__main__":
    main()
