#!/usr/bin/env python3
"""Two resumable cross-algorithm evaluation jobs followed by aggregation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time

SD_REPO = "https://github.com/lawrencewlcknight/fhp-deep-cfr-experiments.git"
UCV_REPO = "https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git"
DEFAULT_UCV_REF = "cf04f1a710e7b947da9fc1c41ae0159c420c72eb"
MODULE = "experiments.fhp.retrospective_sd_ucv_duration_evaluation.run"
STAGES = ("eval24", "eval48", "aggregate")


def q(value):
    return shlex.quote(str(value))


def validate(args):
    for key in ("run_id", "sd5_run_id", "sd6_run_id", "ucv10_run_id", "ucv16_run_id"):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,33}[a-z0-9]", getattr(args, key)):
            raise ValueError(f"{key} must be a 2--35 character lowercase Batch-compatible ID")
    if args.run_id in {args.sd5_run_id, args.sd6_run_id, args.ucv10_run_id, args.ucv16_run_id}:
        raise ValueError("Evaluation must use a new output prefix, never a source run")
    for key in ("bucket", "sd_bucket", "ucv_bucket"):
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", getattr(args, key)):
            raise ValueError(f"{key} must be a bucket URI without a subdirectory")
    for key in ("repo_ref", "ucv_ref"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(args, key)):
            raise ValueError(f"{key} must be a full pushed commit SHA")
    if not 0 < args.max_hours <= 24:
        raise ValueError("Each cohort safety budget must be positive and at most 24 hours")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+[.]iam[.]gserviceaccount[.]com", args.service_account):
        raise ValueError("Set the SD-CFR runner service-account email")


def environment(args):
    values = dict(PROJECT_ID=args.project, REGION=args.region, BUCKET=args.bucket,
                  SD_BUCKET=args.sd_bucket, UCV_BUCKET=args.ucv_bucket,
                  SA_EMAIL=args.service_account, REPO_REF=args.repo_ref,
                  UCV_REPO_REF=args.ucv_ref, RUN_ID=args.run_id,
                  SD_EXP5_RUN_ID=args.sd5_run_id, SD_EXP6_RUN_ID=args.sd6_run_id,
                  UCV_EXP10_RUN_ID=args.ucv10_run_id, UCV_EXP16_RUN_ID=args.ucv16_run_id,
                  EVAL_MAX_HOURS=str(args.max_hours))
    return "\n".join(f"export {key}={q(value)}" for key, value in values.items())


def apt_bootstrap():
    # Native waiting covers dpkg locks; retries also cover apt's list lock.
    # Never delete lock files or interrupt the VM's unattended upgrader.
    return '''apt_with_retry() {
  local attempt result
  for attempt in {1..30}; do
    if apt-get -o DPkg::Lock::Timeout=10 "$@"; then
      return 0
    else
      result=$?
    fi
    if [[ "$attempt" -eq 30 ]]; then
      echo "apt-get $* failed after 30 attempts (exit $result)" >&2
      return "$result"
    fi
    echo "apt-get $* failed (attempt $attempt/30); retrying in 10 seconds" >&2
    sleep 10
  done
}
apt_with_retry update -qq
apt_with_retry install -y -qq git curl ca-certificates
'''


def clone_header(args, *, include_ucv):
    ucv = (f"git clone {q(UCV_REPO)} \"$UCV_REPOSITORY\"\n"
           f"git -C \"$UCV_REPOSITORY\" checkout --detach {q(args.ucv_ref)}\n") if include_ucv else ""
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/sd-ucv-duration-mpl UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
WORK=/workspace/sd-ucv-duration
SD_REPOSITORY="$WORK/sd-repository"
UCV_REPOSITORY="$WORK/ucv-repository"
mkdir -p "$WORK"
{apt_bootstrap()}
git clone {q(SD_REPO)} "$SD_REPOSITORY"
git -C "$SD_REPOSITORY" checkout --detach {q(args.repo_ref)}
{ucv}'''


def python_bootstrap():
    return '''curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH"
uv python install 3.11.17
uv venv --python 3.11.17 "$WORK/venv"
uv pip install --python "$WORK/venv/bin/python" -r "$SD_REPOSITORY/requirements.txt"
export PYTHONPATH="$SD_REPOSITORY"
cd "$SD_REPOSITORY"
'''


def controller_script(args):
    return clone_header(args, include_ucv=False) + environment(args) + f'''
cd "$SD_REPOSITORY"
exec python3 gcp/sd_ucv_duration_head_to_head_batch.py orchestrate{' --resume-controller' if args.resume else ''}
'''


def evaluation_script(args, stage):
    cohort = "24h" if stage == "eval24" else "48h"
    if cohort == "24h":
        sd_run, ucv_run = args.sd5_run_id, args.ucv10_run_id
        sd_algorithm = "parallel_structured_uniform_sd_cfr"
        hours = (6, 12, 18, 24)
    else:
        sd_run, ucv_run = args.sd6_run_id, args.ucv16_run_id
        sd_algorithm = "parallel_structured_uniform_sd_cfr_48h"
        hours = tuple(range(6, 49, 6))
    hour_loop = " ".join(f"{hour:02d}" for hour in hours)
    resume = "1" if args.resume else "0"
    lineage = ""
    lineage_arg = ""
    if cohort == "48h":
        lineage = f'''
  SOURCE10_WORKER="task_00${{seed}}_hand_board_cached_parallel_ucv_escher_seed_${{seed}}"
  mkdir -p "$INPUT/ucv_source10/workers/$SOURCE10_WORKER"
  for metadata in run_manifest.json checkpoint_manifest.json SUCCESS.json; do
    gcloud storage cp {q(args.ucv_bucket + '/' + args.ucv10_run_id)}/workers/$SOURCE10_WORKER/$metadata "$INPUT/ucv_source10/workers/$SOURCE10_WORKER/$metadata"
  done
  gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/continuation_source.json" "$INPUT/ucv/workers/$UCV_WORKER/continuation_source.json"
'''
        lineage_arg = ' --ucv-source10-root "$INPUT/ucv_source10"'
    return clone_header(args, include_ucv=True) + python_bootstrap() + f'''
COHORT={q(cohort)}
INPUT="$WORK/input-$COHORT"
OUTPUT="$WORK/output-$COHORT"
DESTINATION={q(args.bucket + '/' + args.run_id + '/stages/' + cohort)}
SD_SOURCE={q(args.sd_bucket + '/' + sd_run)}
UCV_SOURCE={q(args.ucv_bucket + '/' + ucv_run)}
SYNC_PID=""
mkdir -p "$INPUT/sd/workers" "$INPUT/ucv/workers" "$OUTPUT"
sync_outputs() {{ gcloud storage rsync --recursive --exclude='.*[.]tmp$' "$OUTPUT" "$DESTINATION"; }}
cleanup() {{
  result=$?
  trap - EXIT
  if [[ -n "$SYNC_PID" ]]; then kill "$SYNC_PID" 2>/dev/null || true; wait "$SYNC_PID" 2>/dev/null || true; fi
  if ! sync_outputs; then [[ "$result" -ne 0 ]] || result=1; fi
  exit "$result"
}}
trap cleanup EXIT
if [[ {resume} -eq 1 ]]; then gcloud storage rsync --recursive "$DESTINATION" "$OUTPUT"; fi
for seed in 0 1 2; do
  SD_WORKER="task_00${{seed}}_{sd_algorithm}_seed_${{seed}}"
  UCV_WORKER="task_00${{seed}}_hand_board_cached_parallel_ucv_escher_seed_${{seed}}"
  mkdir -p "$INPUT/sd/workers/$SD_WORKER/archive" "$INPUT/ucv/workers/$UCV_WORKER/checkpoints"
  for metadata in run_manifest.json checkpoint_manifest.json SUCCESS.json; do
    gcloud storage cp "$SD_SOURCE/workers/$SD_WORKER/$metadata" "$INPUT/sd/workers/$SD_WORKER/$metadata"
    gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/$metadata" "$INPUT/ucv/workers/$UCV_WORKER/$metadata"
  done
  gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/runtime_manifest.json" "$INPUT/ucv/workers/$UCV_WORKER/runtime_manifest.json"
  gcloud storage rsync --recursive "$SD_SOURCE/workers/$SD_WORKER/archive" "$INPUT/sd/workers/$SD_WORKER/archive"
  for hour in {hour_loop}; do
    file="hand_board_cached_parallel_ucv_escher_seed_${{seed}}_time_${{hour}}h.pkl"
    gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/checkpoints/$file" "$INPUT/ucv/workers/$UCV_WORKER/checkpoints/$file"
  done
{lineage}done
(while sleep 300; do sync_outputs || echo "Periodic upload failed; will retry" >&2; done) &
SYNC_PID=$!
ARGS=(--cohort "$COHORT" --sd-root "$INPUT/sd" --ucv-root "$INPUT/ucv" --ucv-repo "$UCV_REPOSITORY"{lineage_arg}
      --sd-source-uri "$SD_SOURCE" --ucv-source-uri "$UCV_SOURCE"
      --output "$OUTPUT" --workers 8 --max-hours {args.max_hours})
"$WORK/venv/bin/python" -m {MODULE} "${{ARGS[@]}}" --stage smoke
sync_outputs
"$WORK/venv/bin/python" -m {MODULE} "${{ARGS[@]}}" --stage run
'''


def aggregate_script(args):
    return clone_header(args, include_ucv=False) + python_bootstrap() + f'''
INPUT="$WORK/aggregate-input"
OUTPUT="$WORK/aggregate-output"
DESTINATION={q(args.bucket + '/' + args.run_id)}
mkdir -p "$INPUT/24h" "$INPUT/48h" "$OUTPUT"
for cohort in 24h 48h; do
  gcloud storage rsync --recursive --exclude='.*(task_results|profile_tasks|smoke_tasks)/.*' \
    "$DESTINATION/stages/$cohort" "$INPUT/$cohort"
done
"$WORK/venv/bin/python" -m {MODULE} --stage aggregate --stage-24 "$INPUT/24h" --stage-48 "$INPUT/48h" --output "$OUTPUT"
gcloud storage rsync --recursive "$OUTPUT" "$DESTINATION/analysis"
'''


def job_config(args, stage):
    if stage == "controller":
        script, machine, cpu, memory, disk, seconds = controller_script(args), "e2-small", 1000, 1500, 30, 172800
    elif stage in {"eval24", "eval48"}:
        script, machine, cpu, memory = evaluation_script(args, stage), "n2-standard-8", 8000, 30000
        disk = 200 if stage == "eval24" else 300
        seconds = int((args.max_hours + 2) * 3600)
    elif stage == "aggregate":
        # The aggregation process is allowed 3,000 MiB, so it cannot run on an
        # e2-small (2,048 MiB). Batch rejects that resource combination before
        # creating the job. Keep the memory allowance and use the next machine
        # size so completed evaluation stages can be aggregated on resume.
        script, machine, cpu, memory, disk, seconds = aggregate_script(args), "e2-medium", 1000, 3000, 30, 14400
    else:
        raise ValueError(stage)
    return dict(taskGroups=[dict(taskCount=1, parallelism=1, taskSpec=dict(
        runnables=[dict(script=dict(text=script))], computeResource=dict(cpuMilli=cpu, memoryMib=memory),
        maxRetryCount=0, maxRunDuration=f"{seconds}s"))],
        allocationPolicy=dict(instances=[dict(policy=dict(machineType=machine, provisioningModel="STANDARD",
            bootDisk=dict(sizeGb=disk, type="pd-balanced")))],
            serviceAccount=dict(email=args.service_account, scopes=["https://www.googleapis.com/auth/cloud-platform"])),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=dict(workload="sd-ucv-duration", stage=stage, run=args.run_id))


def cloud(args, *command, check=True, capture=True):
    return subprocess.run(["gcloud", *command, "--project", args.project], text=True,
                          capture_output=capture, check=check)


def objects_exist(args, uri):
    result = cloud(args, "storage", "ls", uri, check=False)
    if result.returncode == 0:
        return bool(result.stdout.strip())
    if "matched no objects" in result.stderr:
        return False
    raise RuntimeError(result.stderr.strip())


def check_sources(args):
    specifications = (
        (args.sd_bucket, args.sd5_run_id, "parallel_structured_uniform_sd_cfr", False),
        (args.sd_bucket, args.sd6_run_id, "parallel_structured_uniform_sd_cfr_48h", False),
        (args.ucv_bucket, args.ucv10_run_id, "hand_board_cached_parallel_ucv_escher", False),
        (args.ucv_bucket, args.ucv16_run_id, "hand_board_cached_parallel_ucv_escher", True),
    )
    for bucket, run, algorithm, continuation in specifications:
        for seed in range(3):
            prefix = f"{bucket}/{run}/workers/task_{seed:03d}_{algorithm}_seed_{seed}"
            names = ["run_manifest.json", "checkpoint_manifest.json", "SUCCESS.json"]
            if algorithm.startswith("hand_board"):
                names.append("runtime_manifest.json")
            if continuation:
                names.append("continuation_source.json")
            for name in names:
                if not objects_exist(args, f"{prefix}/{name}"):
                    raise ValueError(f"Missing source {prefix}/{name}")
            if objects_exist(args, f"{prefix}/FAILURE.json"):
                raise ValueError(f"Failed source worker: {prefix}")


def submit(args, stage, *, tag=""):
    name = f"{args.run_id}-{stage}{tag}"
    with tempfile.TemporaryDirectory(prefix="sd-ucv-duration-submit-") as temporary:
        file = Path(temporary) / "job.json"
        file.write_text(json.dumps(job_config(args, stage)))
        # Stream gcloud diagnostics into the Batch controller log. Capturing
        # stderr here previously reduced a submission failure to an opaque
        # CalledProcessError traceback.
        cloud(args, "batch", "jobs", "submit", name, "--location", args.region,
              "--config", str(file), capture=False)
    return name


def wait_many(args, names):
    pending = set(names)
    while pending:
        for name in tuple(pending):
            state = cloud(args, "batch", "jobs", "describe", name, "--location", args.region,
                          "--format=value(status.state)").stdout.strip()
            print(f"{name}: {state}", flush=True)
            if state == "SUCCEEDED":
                pending.remove(name)
            elif state in {"FAILED", "DELETION_IN_PROGRESS"}:
                raise RuntimeError(f"Batch job {name} ended in state {state}; inspect its task logs")
        if pending:
            time.sleep(30)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "resume", "orchestrate", "status", "dry-run", "check-sources"))
    parser.add_argument("--resume-controller", action="store_true")
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID"), required=not os.environ.get("PROJECT_ID"))
    parser.add_argument("--region", default=os.environ.get("REGION", "europe-west1"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"), required=not os.environ.get("RUN_ID"))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET", "gs://clever-overview-399515-fhp-deep-cfr-results"))
    parser.add_argument("--sd-bucket", default=os.environ.get("SD_BUCKET", "gs://clever-overview-399515-fhp-deep-cfr-results"))
    parser.add_argument("--ucv-bucket", default=os.environ.get("UCV_BUCKET", "gs://clever-overview-399515-fhp-escher-results"))
    parser.add_argument("--sd5-run-id", default=os.environ.get("SD_EXP5_RUN_ID", "sdcfr5-par8-20261002-102757"))
    parser.add_argument("--sd6-run-id", default=os.environ.get("SD_EXP6_RUN_ID", "sdcfr6-48h-20261002-161544"))
    parser.add_argument("--ucv10-run-id", default=os.environ.get("UCV_EXP10_RUN_ID", "exp10-features-20261001-161740"))
    parser.add_argument("--ucv16-run-id", default=os.environ.get("UCV_EXP16_RUN_ID", "exp16-feat48-20261004-182051"))
    parser.add_argument("--repo-ref", default=os.environ.get("REPO_REF", ""))
    parser.add_argument("--ucv-ref", default=os.environ.get("UCV_REPO_REF", DEFAULT_UCV_REF))
    parser.add_argument("--service-account", default=os.environ.get("SA_EMAIL", "fhp-deep-cfr-runner@clever-overview-399515.iam.gserviceaccount.com"))
    parser.add_argument("--max-hours", type=float, default=float(os.environ.get("EVAL_MAX_HOURS", "12")))
    parser.add_argument("--output", type=Path, help="JSON location for dry-run")
    args = parser.parse_args()
    for name in ("bucket", "sd_bucket", "ucv_bucket"):
        setattr(args, name, "gs://" + getattr(args, name).removeprefix("gs://").rstrip("/"))
    args.resume = args.action == "resume" or args.resume_controller
    validate(args)
    return parser, args


def main():
    parser, args = parse_args()
    if args.action == "status":
        print(cloud(args, "batch", "jobs", "list", "--location", args.region,
                    "--filter", f"labels.workload=sd-ucv-duration AND labels.run={args.run_id}").stdout)
        return
    if args.action == "check-sources":
        check_sources(args)
        return
    if args.action == "dry-run":
        if not args.output:
            parser.error("Specify --output for dry-run")
        args.output.write_text(json.dumps({stage: job_config(args, stage)
                                          for stage in ("controller",) + STAGES}, indent=2) + "\n")
        print(f"Wrote {args.output}; no job submitted")
        return
    if args.action == "orchestrate":
        check_sources(args)
        tag = "-r" + datetime.now(timezone.utc).strftime("%H%M%S") if args.resume else ""
        jobs = []
        for stage, cohort in (("eval24", "24h"), ("eval48", "48h")):
            success = f"{args.bucket}/{args.run_id}/stages/{cohort}/SUCCESS.json"
            if args.resume and objects_exist(args, success):
                print(f"Reusing complete {cohort} stage", flush=True)
            else:
                jobs.append(submit(args, stage, tag=tag))
        wait_many(args, jobs)
        aggregate_job = submit(args, "aggregate", tag=tag)
        wait_many(args, [aggregate_job])
        print(f"Complete analysis: {args.bucket}/{args.run_id}/analysis/", flush=True)
        return
    check_sources(args)
    cloud(args, "iam", "service-accounts", "describe", args.service_account)
    exists = objects_exist(args, f"{args.bucket}/{args.run_id}/**")
    if exists and not args.resume:
        raise ValueError("Outputs already exist; use resume with identical sources and refs")
    if args.resume and not (objects_exist(args, f"{args.bucket}/{args.run_id}/stages/24h/evaluation_manifest.json")
                            or objects_exist(args, f"{args.bucket}/{args.run_id}/stages/48h/evaluation_manifest.json")):
        raise ValueError("No duration-evaluation manifest exists to resume")
    active = cloud(args, "batch", "jobs", "list", "--location", args.region,
                   "--filter", f"labels.workload=sd-ucv-duration AND labels.run={args.run_id}", "--format=json")
    if any(job["status"]["state"] not in {"SUCCEEDED", "FAILED"} for job in json.loads(active.stdout)):
        raise ValueError("A job already uses this output prefix")
    tag = "-r" + datetime.now(timezone.utc).strftime("%H%M%S") if args.resume else ""
    name = submit(args, "controller", tag=tag)
    print(f"Submitted {name}; the laptop may disconnect. Outputs: {args.bucket}/{args.run_id}/analysis/")


if __name__ == "__main__":
    main()
