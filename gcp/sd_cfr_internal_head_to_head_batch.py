#!/usr/bin/env python3
"""One-VM resumable Batch job for the saved SD-CFR internal league."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

REPOSITORY = "https://github.com/lawrencewlcknight/fhp-deep-cfr-experiments.git"
MODULE = "experiments.fhp.retrospective_sd_cfr_internal_evaluation.run"
SOURCES = OrderedDict((
    ("exp2", ("optimised_uniform_sd_cfr", "sdcfr2-24h-20261002-003338")),
    ("exp3", ("structured_uniform_sd_cfr", "sdcfr3-24h-20261002-010643")),
    ("exp4", ("structured_uniform_sd_cfr", "sdcfr4-vm16-20261002-095614")),
    ("exp5", ("parallel_structured_uniform_sd_cfr", "sdcfr5-par8-20261002-102757")),
    ("exp6", ("parallel_structured_uniform_sd_cfr_48h", "sdcfr6-48h-20261002-161544")),
    ("exp7", ("distributed_fitting_structured_uniform_sd_cfr", "sdcfr7-distfit-20261003-172011")),
))


def source_runs(args):
    return OrderedDict((name, getattr(args, f"{name}_run_id")) for name in SOURCES)


def validate(args):
    identifier = r"[a-z][a-z0-9-]{0,45}[a-z0-9]"
    if not re.fullmatch(identifier, args.run_id):
        raise ValueError("run_id must be a 2–47 character lowercase Batch-compatible ID")
    for name, run_id in source_runs(args).items():
        if not re.fullmatch(identifier, run_id):
            raise ValueError(f"{name}_run_id is not Batch-compatible")
        if args.run_id == run_id:
            raise ValueError("Evaluation must use a new output prefix, never a source run")
    for key in ("bucket", "source_bucket"):
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", getattr(args, key)):
            raise ValueError(f"{key} must be a bucket URI without a subdirectory")
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        raise ValueError("repo_ref must be a full pushed commit SHA")
    if not 0 < args.max_hours <= 12:
        raise ValueError("Safety budget must be positive and at most 12 hours")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+[.]iam[.]gserviceaccount[.]com", args.service_account):
        raise ValueError("Set the SD-CFR runner service-account email")


def source_uri(args, name):
    return f"{args.source_bucket}/{source_runs(args)[name]}"


def script(args):
    q = shlex.quote
    downloads, uri_args = [], []
    for name, (algorithm, _) in SOURCES.items():
        uri = source_uri(args, name)
        uri_args.append(f"--source-uri {q(name + '=' + uri)}")
        downloads.append(f'''for seed in 0 1 2; do
  WORKER="task_00${{seed}}_{algorithm}_seed_${{seed}}"
  TARGET="$INPUT/{name}/workers/$WORKER"
  mkdir -p "$TARGET/archive"
  for metadata in run_manifest.json checkpoint_manifest.json SUCCESS.json; do
    gcloud storage cp {q(uri)}/workers/$WORKER/$metadata "$TARGET/$metadata"
  done
  gcloud storage rsync --recursive {q(uri)}/workers/$WORKER/archive "$TARGET/archive"
done''')
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/sdcfr-internal-mpl UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
WORK=/workspace/sdcfr-internal
REPOSITORY="$WORK/repository"
INPUT="$WORK/input"
OUTPUT="$WORK/output"
DESTINATION={q(args.bucket + '/' + args.run_id)}
SYNC_PID=""
mkdir -p "$INPUT" "$OUTPUT/analysis"
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
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH"
git clone {q(REPOSITORY)} "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach {q(args.repo_ref)}
uv python install 3.11.17
uv venv --python 3.11.17 "$WORK/venv"
uv pip install --python "$WORK/venv/bin/python" -r "$REPOSITORY/requirements.txt"
export PYTHONPATH="$REPOSITORY"
cd "$REPOSITORY"
if [[ {1 if args.resume else 0} -eq 1 ]]; then
  gcloud storage rsync --recursive "$DESTINATION" "$OUTPUT"
fi
"$WORK/venv/bin/python" gcp/sd_cfr_internal_head_to_head_batch.py check-sources \
  --project {q(args.project)} --region {q(args.region)} --service-account {q(args.service_account)} \
  --run-id {q(args.run_id)} --bucket {q(args.bucket)} --source-bucket {q(args.source_bucket)} \
  --repo-ref {q(args.repo_ref)} {' '.join(f'--{name}-run-id {q(run_id)}' for name, run_id in source_runs(args).items())}
{chr(10).join(downloads)}
(while sleep 300; do sync_outputs || echo "Periodic upload failed; will retry" >&2; done) &
SYNC_PID=$!
ARGS=(--sources-root "$INPUT" --output "$OUTPUT/analysis" --workers 8 --max-hours {args.max_hours}
      {' '.join(uri_args)})
"$WORK/venv/bin/python" -m {MODULE} "${{ARGS[@]}}" --stage smoke
sync_outputs
"$WORK/venv/bin/python" -m {MODULE} "${{ARGS[@]}}" --stage run
'''


def job_config(args):
    validate(args)
    return dict(
        taskGroups=[dict(taskCount=1, parallelism=1, taskSpec=dict(
            runnables=[dict(script=dict(text=script(args)))],
            computeResource=dict(cpuMilli=8000, memoryMib=30000), maxRetryCount=0,
            maxRunDuration=f"{int((args.max_hours + 3) * 3600)}s"))],
        allocationPolicy=dict(
            instances=[dict(policy=dict(machineType="n2-standard-8", provisioningModel="STANDARD",
                                        bootDisk=dict(sizeGb=200, type="pd-balanced")))],
            serviceAccount=dict(email=args.service_account,
                                scopes=["https://www.googleapis.com/auth/cloud-platform"])),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=dict(workload="sdcfr-internal-h2h", run=args.run_id),
    )


def cloud(args, *command, check=True):
    return subprocess.run(["gcloud", *command, "--project", args.project], text=True,
                          capture_output=True, check=check)


def objects_exist(args, uri):
    result = cloud(args, "storage", "ls", uri, check=False)
    if result.returncode == 0:
        return bool(result.stdout.strip())
    if "matched no objects" in result.stderr:
        return False
    raise RuntimeError(result.stderr.strip())


def check_sources(args):
    for name, (algorithm, _) in SOURCES.items():
        run_id = source_runs(args)[name]
        for seed in range(3):
            prefix = f"{args.source_bucket}/{run_id}/workers/task_{seed:03d}_{algorithm}_seed_{seed}"
            for file in ("run_manifest.json", "checkpoint_manifest.json", "SUCCESS.json"):
                if not objects_exist(args, f"{prefix}/{file}"):
                    raise ValueError(f"Missing source {prefix}/{file}")
            if objects_exist(args, f"{prefix}/FAILURE.json"):
                raise ValueError(f"Failed source worker: {prefix}")
            if not objects_exist(args, f"{prefix}/archive/**"):
                raise ValueError(f"Missing playable archive: {prefix}/archive")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("run", "resume", "dry-run", "status", "check-sources"))
    result.add_argument("--project", default=os.environ.get("PROJECT_ID"),
                        required=not os.environ.get("PROJECT_ID"))
    result.add_argument("--region", default=os.environ.get("REGION", "europe-west1"))
    result.add_argument("--run-id", default=os.environ.get("RUN_ID"), required=not os.environ.get("RUN_ID"))
    result.add_argument("--bucket", default=os.environ.get(
        "BUCKET", "gs://clever-overview-399515-fhp-deep-cfr-results"))
    result.add_argument("--source-bucket", default=os.environ.get(
        "SD_BUCKET", "gs://clever-overview-399515-fhp-deep-cfr-results"))
    for name, (_, default) in SOURCES.items():
        result.add_argument(f"--{name}-run-id", default=os.environ.get(f"SD_{name.upper()}_RUN_ID", default))
    result.add_argument("--repo-ref", default=os.environ.get("REPO_REF", ""))
    result.add_argument("--service-account", default=os.environ.get(
        "SA_EMAIL", "fhp-deep-cfr-runner@clever-overview-399515.iam.gserviceaccount.com"))
    result.add_argument("--max-hours", type=float,
                        default=float(os.environ.get("EVAL_MAX_HOURS", "12")))
    result.add_argument("--output", type=Path, help="JSON location for dry-run")
    return result


def main():
    command = parser()
    args = command.parse_args()
    for name in ("bucket", "source_bucket"):
        setattr(args, name, "gs://" + getattr(args, name).removeprefix("gs://").rstrip("/"))
    args.resume = args.action == "resume"
    if args.action == "status":
        print(cloud(args, "batch", "jobs", "list", "--location", args.region,
                    "--filter", f"labels.workload=sdcfr-internal-h2h AND labels.run={args.run_id}").stdout)
        return
    validate(args)
    if args.action == "check-sources":
        check_sources(args)
        return
    config = job_config(args)
    if args.action == "dry-run":
        if not args.output:
            command.error("Specify --output for dry-run")
        args.output.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Wrote {args.output}; no job submitted")
        return
    check_sources(args)
    cloud(args, "iam", "service-accounts", "describe", args.service_account)
    active = cloud(args, "batch", "jobs", "list", "--location", args.region,
                   "--filter", f"labels.workload=sdcfr-internal-h2h AND labels.run={args.run_id}",
                   "--format=json")
    if any(job["status"]["state"] not in {"SUCCEEDED", "FAILED"}
           for job in json.loads(active.stdout)):
        raise ValueError("A job already uses this output prefix; do not run concurrently")
    exists = objects_exist(args, f"{args.bucket}/{args.run_id}/**")
    if exists and not args.resume:
        raise ValueError("Outputs already exist; use resume with the same code and sources")
    if args.resume and not objects_exist(
            args, f"{args.bucket}/{args.run_id}/analysis/evaluation_manifest.json"):
        raise ValueError("No evaluation manifest to resume")
    job_name = args.run_id + (datetime.now(timezone.utc).strftime("-r%H%M%S") if args.resume else "")
    with tempfile.TemporaryDirectory(prefix="sdcfr-internal-submit-") as temporary:
        file = Path(temporary) / "job.json"
        file.write_text(json.dumps(config))
        print(cloud(args, "batch", "jobs", "submit", job_name, "--location", args.region,
                    "--config", str(file)).stdout)
    print(f"Outputs: {args.bucket}/{args.run_id}/analysis/ (one VM; no training or LBR)")


if __name__ == "__main__":
    main()
