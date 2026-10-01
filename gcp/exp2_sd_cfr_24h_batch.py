#!/usr/bin/env python3
"""Stdlib-only Batch controller; three separate training VMs, then evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time


REPO_URL = "https://github.com/lawrencewlcknight/fhp-deep-cfr-experiments.git"
MODULE = "experiments.fhp.exp2_sd_cfr_24h"
STAGES = ("smoke", "train", "aggregate", "profile", "evaluate")


def q(value):
    return shlex.quote(str(value))


def environment(args):
    return dict(PROJECT_ID=args.project, REGION=args.region, BUCKET=args.bucket,
                SA_EMAIL=args.service_account, REPO_REF=args.repo_ref, RUN_ID=args.run_id,
                UCV_EXP1_RUN_ID=args.ucv_run_id, UCV_EVAL_RUN_ID=args.ucv_eval_run_id,
                EVAL_MAX_HOURS=str(args.eval_max_hours), PARALLELISM="3")


def bootstrap(args, *, controller=False):
    setup = f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/fhp-sdcfr-mpl
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
WORK=/workspace/sdcfr-exp2
mkdir -p "$WORK"
git clone --filter=blob:none {q(REPO_URL)} "$WORK/repository"
cd "$WORK/repository"
git checkout --detach {q(args.repo_ref)}
"""
    if controller:
        return setup
    return setup + """
export UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH"
uv python install 3.11
uv venv --python 3.11 --seed /tmp/fhp-sdcfr-exp2-venv
source /tmp/fhp-sdcfr-exp2-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
OUT="$WORK/output"
INPUT="$WORK/input"
mkdir -p "$OUT" "$INPUT"
"""


def script(args, stage):
    env = "\n".join(f"export {key}={q(value)}" for key, value in environment(args).items())
    remote = f"{args.bucket}/{args.run_id}"
    if stage == "controller":
        return bootstrap(args, controller=True) + env + "\n" + (
            f"exec python3 gcp/exp2_sd_cfr_24h_batch.py orchestrate --start-stage {q(args.start_stage)}\n")
    text = bootstrap(args) + env + "\n"
    if stage == "train":
        return text + f"""
SEED="${{BATCH_TASK_INDEX:?Missing Batch task index}}"
case "$SEED" in 0|1|2) ;; *) exit 2 ;; esac
TASK="task_$(printf '%03d' "$SEED")_optimised_uniform_sd_cfr_seed_$SEED"
REMOTE={q(remote)}/workers/$TASK
finish() {{
  code=$?
  if [[ -d "$OUT/workers/$TASK" ]]; then
    gcloud storage rsync --recursive --exclude='\\.tmp$' "$OUT/workers/$TASK" "$REMOTE" || {{ if [[ "$code" == 0 ]]; then code=1; fi; }}
  fi
  exit "$code"
}}
trap finish EXIT
python -m {MODULE}.train --seed "$SEED" --output-root "$OUT" --remote-uri "$REMOTE"
"""
    if stage == "smoke":
        return text + f"""
trap 'code=$?; gcloud storage rsync --recursive "$OUT" {q(remote + '/smoke')} || true; exit "$code"' EXIT
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_exp2_sd_cfr_24h.py tests/test_single_solver.py tests/test_sd_cfr_efficiency.py
python -m experiments.fhp.exp1_sd_cfr_efficiency.run --seeds 0 1 2 --repeats 1 --output-dir "$OUT/equivalence"
python -m {MODULE}.stress --output "$OUT/capacity_stress.json"
python -m {MODULE}.train --seed 0 --smoke --output-root "$OUT/training"
gcloud storage rsync --recursive --exclude='(^|/)training_states(/|$)' \
  {q(args.bucket + '/' + args.ucv_run_id + '/workers')} "$INPUT/ucv/workers"
gcloud storage rsync --recursive --exclude='(^|/)(tasks|task_results)(/|$)' \
  {q(args.bucket + '/' + args.ucv_eval_run_id + '/analysis')} "$INPUT/reference"
python -m {MODULE}.evaluate smoke --source "$OUT/training" --output "$OUT/evaluation" --workers 2 \
  --ucv-source "$INPUT/ucv" --reference-analysis "$INPUT/reference"
gcloud storage rsync --recursive "$OUT" {q(remote + '/smoke')}
"""
    text += f"gcloud storage rsync --recursive {q(remote + '/workers')} \"$INPUT/sd/workers\"\n"
    if stage == "aggregate":
        return text + f"""
python -m {MODULE}.report --source "$INPUT/sd" --output "$OUT/analysis"
gcloud storage rsync --recursive "$OUT/analysis" {q(remote + '/analysis')}
"""
    text += f"""
gcloud storage rsync --recursive --exclude='(^|/)training_states(/|$)' \
  {q(args.bucket + '/' + args.ucv_run_id + '/workers')} "$INPUT/ucv/workers"
gcloud storage rsync --recursive --exclude='(^|/)(tasks|task_results)(/|$)' \
  {q(args.bucket + '/' + args.ucv_eval_run_id + '/analysis')} "$INPUT/reference"
mkdir -p "$OUT/evaluation"
"""
    if stage == "evaluate":
        text += f"gcloud storage rsync --recursive {q(remote + '/evaluation')} \"$OUT/evaluation\"\n"
    # Preserve completed task shards even on timeout; periodically upload them
    # without repeatedly transferring the multi-GB read-only input archives.
    text += f"""
upload() {{ gcloud storage rsync --recursive --exclude='\\.tmp$' "$OUT/evaluation" {q(remote + '/evaluation')}; }}
periodic() {{ while sleep 300; do upload || true; done; }}
periodic & UPLOAD_PID=$!
finish() {{
  code=$?
  kill "$UPLOAD_PID" >/dev/null 2>&1 || true
  wait "$UPLOAD_PID" >/dev/null 2>&1 || true
  upload || {{ if [[ "$code" == 0 ]]; then code=1; fi; }}
  exit "$code"
}}
trap finish EXIT
trap 'exit 143' TERM
python -m {MODULE}.evaluate {'profile' if stage == 'profile' else 'run'} \
  --source "$INPUT/sd" --ucv-source "$INPUT/ucv" --reference-analysis "$INPUT/reference" \
  --output "$OUT/evaluation" --workers 8 --max-hours {args.eval_max_hours}
"""
    return text


def build_job(args, stage):
    if stage not in STAGES + ("controller",):
        raise ValueError(stage)
    if stage == "controller":
        machine, cpu, memory, disk, seconds = "e2-small", 1000, 1500, 30, 604800
    else:
        machine, cpu, memory, disk = "n2-standard-8", 8000, 30000, 200
        seconds = {"train": 129600, "smoke": 7200, "aggregate": 14400,
                   "profile": 14400, "evaluate": int((args.eval_max_hours + 2) * 3600)}[stage]
    count = 3 if stage == "train" else 1
    return dict(taskGroups=[dict(taskSpec=dict(runnables=[dict(script=dict(text=script(args, stage)))],
                computeResource=dict(cpuMilli=cpu, memoryMib=memory), maxRetryCount=0,
                maxRunDuration=f"{seconds}s"), taskCount=count, parallelism=count, taskCountPerNode=1)],
                allocationPolicy=dict(serviceAccount=dict(email=args.service_account),
                instances=[dict(policy=dict(machineType=machine, provisioningModel="STANDARD",
                bootDisk=dict(sizeGb=disk, type="pd-balanced")))]),
                logsPolicy=dict(destination="CLOUD_LOGGING"),
                labels=dict(experiment="fhp-sdcfr-exp2-24h", stage=stage))


def cloud(args, *command, capture=False):
    return subprocess.run(["gcloud", *command, "--project", args.project], check=True,
                          text=True, capture_output=capture)


def submit(args, stage, *, retry_tag=""):
    name = f"{args.run_id}-{stage}{retry_tag}"
    with tempfile.TemporaryDirectory(prefix="fhp-sdcfr-exp2-job-") as temporary:
        path = Path(temporary) / "job.json"
        path.write_text(json.dumps(build_job(args, stage), indent=2))
        cloud(args, "batch", "jobs", "submit", name, "--location", args.region, "--config", str(path))
    return name


def wait(args, name):
    while True:
        state = cloud(args, "batch", "jobs", "describe", name, "--location", args.region,
                      "--format=value(status.state)", capture=True).stdout.strip()
        print(f"{name}: {state}", flush=True)
        if state == "SUCCEEDED":
            return
        if state in {"FAILED", "DELETION_IN_PROGRESS"}:
            raise RuntimeError(f"{name} failed; later stages were not launched")
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "orchestrate", "status", "dry-run", "evaluate-only"))
    for name, env, default in (("project", "PROJECT_ID", None), ("region", "REGION", None),
                              ("bucket", "BUCKET", None), ("service-account", "SA_EMAIL", None),
                              ("repo-ref", "REPO_REF", None), ("run-id", "RUN_ID", None),
                              ("ucv-run-id", "UCV_EXP1_RUN_ID", "exp1-fhp-20260923-233627"),
                              ("ucv-eval-run-id", "UCV_EVAL_RUN_ID", "fhp-eval123-20260925-103616")):
        parser.add_argument("--" + name, default=os.environ.get(env, default))
    parser.add_argument("--eval-max-hours", type=float, default=float(os.environ.get("EVAL_MAX_HOURS", "36")))
    parser.add_argument("--start-stage", choices=("smoke", "profile"), default="smoke")
    args = parser.parse_args()
    if not all((args.project, args.region, args.bucket, args.service_account, args.repo_ref, args.run_id)):
        parser.error("Set PROJECT_ID, REGION, BUCKET, SA_EMAIL, REPO_REF and RUN_ID")
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,34}", args.run_id):
        parser.error("RUN_ID must be 2..35 lowercase Batch-compatible characters")
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        parser.error("REPO_REF must be the full pushed commit SHA")
    if not 0 < args.eval_max_hours <= 96:
        parser.error("EVAL_MAX_HOURS must be in (0, 96]")
    args.bucket = args.bucket.rstrip("/")
    if not args.bucket.startswith("gs://"):
        args.bucket = "gs://" + args.bucket
    if args.action == "evaluate-only":
        args.start_stage = "profile"
    if args.action == "dry-run":
        print(json.dumps({stage: build_job(args, stage) for stage in ("controller",) + STAGES}, indent=2))
        return
    if args.action == "status":
        cloud(args, "batch", "jobs", "list", "--location", args.region,
              "--filter", f"name:{args.run_id}", "--format=table(name.basename(),status.state)")
        return
    if args.action in ("run", "evaluate-only"):
        cloud(args, "iam", "service-accounts", "describe", args.service_account)
        for prefix in (args.ucv_run_id + "/workers/**/checkpoint_manifest.json",
                       args.ucv_eval_run_id + "/analysis/evaluation_manifest.json"):
            cloud(args, "storage", "ls", args.bucket + "/" + prefix)
        tag = "-" + time.strftime("%H%M%S", time.gmtime()) if args.action == "evaluate-only" else ""
        name = submit(args, "controller", retry_tag=tag)
        print(f"Submitted {name}; the laptop may disconnect. Outputs: {args.bucket}/{args.run_id}")
        return
    # Controller identity must have child-job creation + service-account use.
    cloud(args, "batch", "jobs", "list", "--location", args.region, "--limit=1")
    stages = STAGES[STAGES.index(args.start_stage):]
    tag = "-" + time.strftime("%H%M%S", time.gmtime()) if args.start_stage == "profile" else ""
    for stage in stages:
        wait(args, submit(args, stage, retry_tag=tag))


if __name__ == "__main__":
    main()
