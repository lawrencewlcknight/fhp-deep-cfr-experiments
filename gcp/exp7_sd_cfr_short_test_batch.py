#!/usr/bin/env python3
"""One VM, one short diagnostic task; never submits a long-run controller."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcp import exp2_sd_cfr_24h_batch as base

MODULE = "experiments.fhp.exp7_sd_cfr_distributed_fitting_24h.short_test"
REQUIRED_FILES = (
    "experiments/fhp/exp7_sd_cfr_distributed_fitting_24h/short_test.py",
    "experiments/fhp/exp7_sd_cfr_distributed_fitting_24h/config.py",
    "experiments/fhp/exp7_sd_cfr_distributed_fitting_24h/benchmark.py",
    "deep_cfr_poker/sd_cfr_distributed.py",
    "gcp/exp7_sd_cfr_short_test_batch.py",
)


def script(args):
    return base.bootstrap(args) + f"""
REMOTE={base.q(args.bucket + '/' + args.run_id + '/analysis')}
mkdir -p "$OUT/analysis"
UPLOAD_PID=
upload() {{
  gcloud storage rsync --recursive --exclude='.*[.]tmp$' "$OUT/analysis" "$REMOTE"
}}
finish() {{
  code=$?
  trap - EXIT
  if [[ -n "$UPLOAD_PID" ]]; then
    kill "$UPLOAD_PID" 2>/dev/null || true
    wait "$UPLOAD_PID" 2>/dev/null || true
  fi
  upload || {{ if [[ "$code" == 0 ]]; then code=1; fi; }}
  exit "$code"
}}
trap finish EXIT
trap 'exit 143' TERM
(while sleep 60; do upload || true; done) &
UPLOAD_PID=$!
python -m {MODULE} --output "$OUT/analysis" --seed {args.seed}
"""


def build_job(args):
    return dict(
        taskGroups=[dict(taskCount=1, parallelism=1, taskCountPerNode=1,
            taskSpec=dict(runnables=[dict(script=dict(text=script(args)))],
                computeResource=dict(cpuMilli=16000, memoryMib=60000),
                maxRunDuration="7200s", maxRetryCount=0))],
        allocationPolicy=dict(serviceAccount=dict(email=args.service_account),
            instances=[dict(policy=dict(machineType="n2-standard-16", provisioningModel="STANDARD",
                bootDisk=dict(sizeGb=100, type="pd-balanced")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=dict(experiment="fhp-sdcfr-exp7-short", stage="benchmark"))


def check_pinned_files(args):
    root = Path(__file__).resolve().parents[1]
    for path in REQUIRED_FILES:
        result = subprocess.run(["git", "cat-file", "-e", f"{args.repo_ref}:{path}"], cwd=root,
                                capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(f"{path} is absent from REPO_REF={args.repo_ref}. "
                             "Commit and push the short test, then refresh REPO_REF.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "dry-run"))
    for name, env in (("project", "PROJECT_ID"), ("region", "REGION"), ("bucket", "BUCKET"),
                      ("service-account", "SA_EMAIL"), ("repo-ref", "REPO_REF"), ("run-id", "RUN_ID")):
        parser.add_argument("--" + name, default=os.environ.get(env))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if not all((args.project, args.region, args.bucket, args.service_account, args.repo_ref, args.run_id)):
        parser.error("Set PROJECT_ID, REGION, BUCKET, SA_EMAIL, REPO_REF and RUN_ID")
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,49}", args.run_id):
        parser.error("RUN_ID must be 2..50 lowercase Batch-compatible characters")
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        parser.error("REPO_REF must be the full pushed commit SHA")
    args.bucket = args.bucket.rstrip("/")
    if not args.bucket.startswith("gs://"):
        args.bucket = "gs://" + args.bucket
    name = args.run_id + "-short"
    if args.action == "dry-run":
        print(json.dumps(build_job(args), indent=2))
        return
    if args.action == "status":
        base.cloud(args, "batch", "jobs", "describe", name, "--location", args.region,
                   "--format=value(status.state)")
        return
    check_pinned_files(args)
    base.cloud(args, "iam", "service-accounts", "describe", args.service_account)
    with tempfile.TemporaryDirectory(prefix="fhp-sdcfr7-short-job-") as temporary:
        path = Path(temporary) / "job.json"
        path.write_text(json.dumps(build_job(args), indent=2))
        base.cloud(args, "batch", "jobs", "submit", name, "--location", args.region, "--config", str(path))
    print(f"Submitted {name}; the laptop may disconnect. Outputs: {args.bucket}/{args.run_id}/analysis")
    print("One short VM job only; no long training or evaluation jobs will be submitted.")


if __name__ == "__main__":
    main()
