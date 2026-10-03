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
DEFAULT_EXPERIMENT = dict(number=2, module=MODULE, algorithm_id="optimised_uniform_sd_cfr",
                          batch_script="gcp/exp2_sd_cfr_24h_batch.py",
                          test_file="tests/test_exp2_sd_cfr_24h.py")


def settings(args):
    return getattr(args, "experiment", DEFAULT_EXPERIMENT)


def q(value):
    return shlex.quote(str(value))


def lbr_enabled(args):
    value = str(getattr(args, "eval_lbr", settings(args).get("default_lbr",
                "0" if settings(args)["number"] in (2, 3, 4, 5) else "1")))
    if value not in {"0", "1"}:
        raise ValueError("EVAL_LBR must be 0 (omit) or 1 (include)")
    return value == "1"


def evaluation_settings(args):
    # An old GPU export must not allocate a GPU for ordinary sampled matches.
    device = getattr(args, "eval_lbr_device", "cpu") if lbr_enabled(args) else "cpu"
    workers = getattr(args, "eval_workers", 0) or (2 if device == "cuda" else 8)
    return device, workers


def environment(args):
    device, workers = evaluation_settings(args)
    result = dict(PROJECT_ID=args.project, REGION=args.region, BUCKET=args.bucket,
                SA_EMAIL=args.service_account, REPO_REF=args.repo_ref, RUN_ID=args.run_id,
                EVAL_MAX_HOURS=str(args.eval_max_hours), PARALLELISM="3",
                EVAL_LBR_DEVICE=device, EVAL_WORKERS=str(workers),
                EVAL_LBR="1" if lbr_enabled(args) else "0",
                EVAL_PROFILE_ONLY="1" if getattr(args, "profile_only", False) else "0")
    if settings(args).get("comparison"):
        result["EXP5_RUN_ID"] = reference_run_id(args)
    if getattr(args, "resume_run_id", None):
        result.update(RESUME_RUN_ID=args.resume_run_id, ADDITIONAL_HOURS=str(args.additional_hours))
    return result


def reference_run_id(args):
    return getattr(args, "reference_run_id", None) or settings(args)["comparison"]["default_run_id"]


def bootstrap(args, *, controller=False, gpu=False):
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
    python_version = 'TARGET_PYTHON_VERSION="3.11"\n'
    if getattr(args, "resume_run_id", None):
        # A later 3.11 patch release must not silently change the saved runtime.
        task = f"task_000_{settings(args)['algorithm_id']}_seed_0"
        source = f"{args.bucket}/{args.resume_run_id}/workers/{task}/training_state/manifest.json"
        code = ('import json,re,sys; v=json.load(sys.stdin)["runtime"]["python"]; '
                'assert re.fullmatch(r"3[.]11[.][0-9]+", v), "Unsupported saved Python version"; print(v)')
        python_version = f'TARGET_PYTHON_VERSION="$(gcloud storage cat {q(source)} | python3 -c {q(code)})"\n'
    result = setup + python_version + """
export UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH"
uv python install "$TARGET_PYTHON_VERSION"
uv venv --python "$TARGET_PYTHON_VERSION" --seed /tmp/fhp-sdcfr-exp2-venv
source /tmp/fhp-sdcfr-exp2-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
OUT="$WORK/output"
INPUT="$WORK/input"
mkdir -p "$OUT" "$INPUT"
"""
    if gpu:
        # Evaluation only: training retains its pinned CPU wheel and machine.
        result += """
python -m pip install --no-cache-dir --upgrade torch==2.7.0+cu126 torchvision==0.22.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"; print(torch.cuda.get_device_name(0))'
"""
    return result


def script(args, stage):
    spec = settings(args)
    module = spec["module"]
    env = "\n".join(f"export {key}={q(value)}" for key, value in environment(args).items())
    remote = f"{args.bucket}/{args.run_id}"
    if stage == "controller":
        return bootstrap(args, controller=True) + env + "\n" + (
            f"exec python3 {q(spec['batch_script'])} orchestrate --start-stage {q(args.start_stage)}\n")
    device, workers = evaluation_settings(args)
    scope = "evaluation" if lbr_enabled(args) else "evaluation_no_lbr"
    lbr_flag = "" if lbr_enabled(args) else " --skip-lbr"
    gpu = stage in {"profile", "evaluate"} and device == "cuda"
    text = bootstrap(args, gpu=gpu) + env + "\n"
    if gpu and stage == "profile":
        # requirements-dev includes the CPU requirements; do not reinstall it
        # here and silently replace the just-validated CUDA wheel.
        text += "python -m pip install 'pytest>=7,<9'\npython -m pytest -q tests/test_sd_cfr_lbr.py\n"
    if stage == "train":
        resume = ""
        resume_flags = ""
        if getattr(args, "resume_run_id", None):
            resume = (f'gcloud storage rsync --recursive {q(args.bucket + "/" + args.resume_run_id + "/workers")}/"$TASK" "$INPUT/resume"\n')
            resume_flags = (' --resume-state "$INPUT/resume/training_state/manifest.json"'
                            f' --additional-hours {int(args.additional_hours)}')
        return text + f"""
SEED="${{BATCH_TASK_INDEX:?Missing Batch task index}}"
case "$SEED" in 0|1|2) ;; *) exit 2 ;; esac
TASK="task_$(printf '%03d' "$SEED")_{spec['algorithm_id']}_seed_$SEED"
REMOTE={q(remote)}/workers/$TASK
finish() {{
  code=$?
  if [[ -d "$OUT/workers/$TASK" ]]; then
    gcloud storage rsync --recursive --exclude='.*[.]tmp$' "$OUT/workers/$TASK" "$REMOTE" || {{ if [[ "$code" == 0 ]]; then code=1; fi; }}
  fi
  exit "$code"
}}
trap finish EXIT
{resume}python -m {module}.train --seed "$SEED" --output-root "$OUT" --remote-uri "$REMOTE"{resume_flags}
"""
    if stage == "smoke":
        reference_smoke = ""
        reference_flag = ""
        if spec.get("comparison"):
            reference_smoke = (f'gcloud storage rsync --recursive --exclude="(^|/)(archive|training_state)/.*" '
                f'{q(args.bucket + "/" + reference_run_id(args) + "/workers")} "$INPUT/reference/workers"\n'
                f'python -m {module}.comparison --validate-reference "$INPUT/reference"\n'
                f'python -m {spec["comparison"]["module"]}.train --seed 0 --smoke --output-root "$OUT/reference_training"\n')
            reference_flag = ' --reference-source "$OUT/reference_training"'
        test_files = " ".join(q(path) for path in
                              (spec["test_file"], *spec.get("extra_test_files", ())))
        ray_check = ""
        if spec.get("parallel_smoke"):
            ray_check = (f"RUN_RAY_SD_CFR_TESTS=1 python -m pytest -q {q(spec['test_file'])} "
                         "tests/test_sd_cfr_parallel.py -k real_ray\n")
        if spec.get("fitting_benchmark"):
            ray_check += (f'python -m {module}.benchmark --output "$OUT/fitting_benchmark" '
                          '--repeats 3' + (' --allow-trajectory-drift' if spec.get("allow_trajectory_drift") else '') + '\n')
        return text + f"""
trap 'code=$?; gcloud storage rsync --recursive "$OUT" {q(remote + '/smoke')} || true; exit "$code"' EXIT
python -m pip install -r requirements-dev.txt
python -m pytest -q {test_files} tests/test_single_solver.py tests/test_sd_cfr_efficiency.py tests/test_sd_cfr_lbr.py tests/test_sd_cfr_no_lbr.py
{reference_smoke}
{ray_check}
python -m experiments.fhp.exp1_sd_cfr_efficiency.run --seeds 0 1 2 --repeats 1 --output-dir "$OUT/equivalence"
python -m {module}.stress --output "$OUT/capacity_stress.json"
python -m {module}.train --seed 0 --smoke --output-root "$OUT/training"
python -m {module}.evaluate smoke --source "$OUT/training" --output "$OUT/evaluation" --workers 2{lbr_flag}{reference_flag}
gcloud storage rsync --recursive "$OUT" {q(remote + '/smoke')}
"""
    exclusions = " --exclude='(^|.*/)training_state/.*|.*[.]tmp$'" if spec.get("final_training_state") else ""
    text += f"gcloud storage rsync --recursive{exclusions} {q(remote + '/workers')} \"$INPUT/sd/workers\"\n"
    if stage == "aggregate":
        return text + f"""
python -m {module}.report --source "$INPUT/sd" --output "$OUT/analysis"
gcloud storage rsync --recursive "$OUT/analysis" {q(remote + '/analysis')}
"""
    reference_flag = ""
    if spec.get("comparison"):
        text += (f'gcloud storage rsync --recursive --exclude="(^|/)training_state/.*|.*[.]tmp$" '
                 f'{q(args.bucket + "/" + reference_run_id(args) + "/workers")} "$INPUT/reference/workers"\n')
        reference_flag = ' --reference-source "$INPUT/reference"'
    text += f"""
mkdir -p "$OUT/evaluation"
"""
    if stage == "evaluate":
        text += f"gcloud storage rsync --recursive {q(remote + '/' + scope)} \"$OUT/evaluation\"\n"
    # Preserve completed task shards even on timeout; periodically upload them
    # without repeatedly transferring the multi-GB read-only input archives.
    text += f"""
upload() {{ gcloud storage rsync --recursive --exclude='\\.tmp$' "$OUT/evaluation" {q(remote + '/' + scope)}; }}
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
python -m {module}.evaluate {'profile' if stage == 'profile' else 'run'} \
  --source "$INPUT/sd" \
  --output "$OUT/evaluation" --workers {workers} --max-hours {args.eval_max_hours} --lbr-device {q(device)}{lbr_flag}{reference_flag}
"""
    return text


def build_job(args, stage):
    if stage not in STAGES + ("controller",):
        raise ValueError(stage)
    if stage == "controller":
        machine, cpu, memory, disk, seconds = "e2-small", 1000, 1500, 30, 604800
    else:
        machine, cpu, memory, disk = "n2-standard-8", 8000, 30000, 200
        if stage in {"smoke", "train"}:
            resources = settings(args).get("training_resources", {})
            machine = resources.get("machine_type", machine)
            cpu = resources.get("cpu_milli", cpu)
            memory = resources.get("memory_mib", memory)
        seconds = {"train": 129600, "smoke": 7200, "aggregate": 14400,
                   "profile": 14400, "evaluate": int((args.eval_max_hours + 2) * 3600)}[stage]
        if stage == "train":
            seconds = settings(args).get("train_max_seconds", seconds)
        if stage in {"profile", "evaluate"} and evaluation_settings(args)[0] == "cuda":
            machine = "g2-standard-8"  # One 24-GB L4; opt-in, never changes training.
    count = 3 if stage == "train" else 1
    result = dict(taskGroups=[dict(taskSpec=dict(runnables=[dict(script=dict(text=script(args, stage)))],
                computeResource=dict(cpuMilli=cpu, memoryMib=memory), maxRetryCount=0,
                maxRunDuration=f"{seconds}s"), taskCount=count, parallelism=count, taskCountPerNode=1)],
                allocationPolicy=dict(serviceAccount=dict(email=args.service_account),
                instances=[dict(policy=dict(machineType=machine, provisioningModel="STANDARD",
                bootDisk=dict(sizeGb=disk, type="pd-balanced")))]),
                logsPolicy=dict(destination="CLOUD_LOGGING"),
                labels=dict(experiment=f"fhp-sdcfr-exp{settings(args)['number']}-{settings(args).get('hours', 24)}h", stage=stage))
    if stage in {"profile", "evaluate"} and evaluation_settings(args)[0] == "cuda":
        result["allocationPolicy"]["instances"][0]["installGpuDrivers"] = True
    return result


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


def main(*, experiment=DEFAULT_EXPERIMENT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "orchestrate", "status", "dry-run", "evaluate-only", "profile-only"))
    for name, env, default in (("project", "PROJECT_ID", None), ("region", "REGION", None),
                              ("bucket", "BUCKET", None), ("service-account", "SA_EMAIL", None),
                              ("repo-ref", "REPO_REF", None), ("run-id", "RUN_ID", None)):
        parser.add_argument("--" + name, default=os.environ.get(env, default))
    parser.add_argument("--eval-max-hours", type=float, default=float(os.environ.get("EVAL_MAX_HOURS", "36")))
    parser.add_argument("--eval-lbr", choices=("0", "1"),
                        default=os.environ.get("EVAL_LBR", experiment.get("default_lbr",
                                               "0" if experiment["number"] in (2, 3, 4, 5) else "1")),
                        help="0: rule/temporal matches only; 1: also include full-mixture LBR")
    parser.add_argument("--eval-lbr-device", choices=("cpu", "cuda"), default=os.environ.get("EVAL_LBR_DEVICE", "cpu"))
    parser.add_argument("--eval-workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "0")))
    parser.add_argument("--profile-only", action="store_true", default=os.environ.get("EVAL_PROFILE_ONLY", "0") == "1")
    parser.add_argument("--start-stage", choices=("smoke", "profile"), default="smoke")
    parser.add_argument("--reference-run-id", default=os.environ.get("EXP5_RUN_ID") if experiment.get("comparison") else None)
    parser.add_argument("--resume-run-id", default=os.environ.get("RESUME_RUN_ID") if experiment.get("final_training_state") else None)
    parser.add_argument("--additional-hours", type=int, default=int(os.environ.get("ADDITIONAL_HOURS", "24")))
    args = parser.parse_args()
    args.experiment = experiment
    if not all((args.project, args.region, args.bucket, args.service_account, args.repo_ref, args.run_id)):
        parser.error("Set PROJECT_ID, REGION, BUCKET, SA_EMAIL, REPO_REF and RUN_ID")
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,34}", args.run_id):
        parser.error("RUN_ID must be 2..35 lowercase Batch-compatible characters")
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        parser.error("REPO_REF must be the full pushed commit SHA")
    if not 0 < args.eval_max_hours <= 96:
        parser.error("EVAL_MAX_HOURS must be in (0, 96]")
    if experiment.get("comparison") and (not re.fullmatch(r"[a-z][a-z0-9-]{1,34}", reference_run_id(args))
                                          or reference_run_id(args) == args.run_id):
        parser.error("EXP5_RUN_ID must identify a distinct completed Experiment 5 run")
    if args.eval_lbr not in {"0", "1"}:
        parser.error("EVAL_LBR must be 0 (omit) or 1 (include)")
    device, workers = evaluation_settings(args)
    if device not in {"cpu", "cuda"} or not 1 <= workers <= (4 if device == "cuda" else 8):
        parser.error("EVAL_WORKERS: 1..8 for CPU, 1..4 sharing the evaluation GPU")
    if args.resume_run_id:
        if (not experiment.get("final_training_state")
                or not re.fullmatch(r"[a-z][a-z0-9-]{1,34}", args.resume_run_id)
                or args.resume_run_id == args.run_id or args.additional_hours not in range(6, 49, 6)):
            parser.error("Resume needs a supported experiment, distinct source/new RUN_IDs, and ADDITIONAL_HOURS=6..48 in steps of 6")
    args.bucket = args.bucket.rstrip("/")
    if not args.bucket.startswith("gs://"):
        args.bucket = "gs://" + args.bucket
    if args.action == "profile-only":
        args.profile_only = True
    if args.action in {"evaluate-only", "profile-only"} or args.profile_only:
        args.start_stage = "profile"
    if args.action == "dry-run":
        print(json.dumps({stage: build_job(args, stage) for stage in ("controller",) + STAGES}, indent=2))
        return
    if args.action == "status":
        cloud(args, "batch", "jobs", "list", "--location", args.region,
              "--filter", f"name:{args.run_id}", "--format=table(name.basename(),status.state)")
        return
    if args.action in ("run", "evaluate-only", "profile-only"):
        cloud(args, "iam", "service-accounts", "describe", args.service_account)
        if experiment.get("comparison"):
            for seed in range(3):
                task = f"task_{seed:03d}_{experiment['comparison']['algorithm_id']}_seed_{seed}"
                cloud(args, "storage", "ls", f"{args.bucket}/{reference_run_id(args)}/workers/{task}/SUCCESS.json")
        if args.resume_run_id and args.action == "run":
            for seed in range(3):
                task = f"task_{seed:03d}_{experiment['algorithm_id']}_seed_{seed}"
                cloud(args, "storage", "ls", f"{args.bucket}/{args.resume_run_id}/workers/{task}/training_state/manifest.json")
        tag = "-" + time.strftime("%H%M%S", time.gmtime()) if args.start_stage == "profile" else ""
        name = submit(args, "controller", retry_tag=tag)
        print(f"Submitted {name}; the laptop may disconnect. Outputs: {args.bucket}/{args.run_id}")
        return
    # Controller identity must have child-job creation + service-account use.
    cloud(args, "batch", "jobs", "list", "--location", args.region, "--limit=1")
    stages = ("profile",) if args.profile_only else STAGES[STAGES.index(args.start_stage):]
    tag = "-" + time.strftime("%H%M%S", time.gmtime()) if args.start_stage == "profile" else ""
    for stage in stages:
        wait(args, submit(args, stage, retry_tag=tag))


if __name__ == "__main__":
    main()
