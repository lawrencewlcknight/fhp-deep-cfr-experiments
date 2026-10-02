import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from deep_cfr_poker.sd_cfr_structured import StructuredSingleDeepCFRSolver
from experiments.fhp.exp2_sd_cfr_24h import train, evaluate, report, stress
from experiments.fhp.exp3_sd_cfr_structured_24h import config as baseline
from experiments.fhp.exp4_sd_cfr_structured_n2_standard16 import config
from gcp import exp3_sd_cfr_structured_24h_batch as old_batch
from gcp import exp4_sd_cfr_structured_n2_standard16_batch as batch


ROOT = Path(__file__).resolve().parents[1]
MODULE = "experiments.fhp.exp4_sd_cfr_structured_n2_standard16"


def arguments():
    return SimpleNamespace(project="test", region="europe-west1", bucket="gs://test",
                           service_account="runner@test", repo_ref="a" * 40,
                           run_id="sdcfr4-test", start_stage="smoke", eval_max_hours=36)


def test_only_hardware_and_experiment_identity_change():
    assert config.solver_config is baseline.solver_config
    assert config.solver_config() == baseline.solver_config()
    assert config.solver_config(True) == baseline.solver_config(True)
    assert config.FEATURE_ENCODER_METADATA == baseline.FEATURE_ENCODER_METADATA
    assert config.ALGORITHM_ID == baseline.ALGORITHM_ID
    assert config.SEEDS == baseline.SEEDS == (0, 1, 2)
    assert config.HOURS == baseline.HOURS == (6, 12, 18, 24)
    assert config.SECONDS == baseline.SECONDS and config.SECONDS[-1] == 86400
    assert config.ITERATION_CAP == baseline.ITERATION_CAP
    assert config.EXPERIMENT_NAME != baseline.EXPERIMENT_NAME
    assert config.REPORT_ID != baseline.REPORT_ID
    assert config.REFERENCE_VM == dict(baseline.REFERENCE_VM, machine_type="n2-standard-16",
                                       vcpus=16, memory_gib=64)
    # Copy, never mutate the baseline's machine declaration.
    assert baseline.REFERENCE_VM["machine_type"] == "n2-standard-8"
    for seed in config.SEEDS:
        assert config.task_name(seed) == baseline.task_name(seed)
    with pytest.raises(ValueError):
        config.task_name(3)


@pytest.mark.parametrize("stage", ("controller",) + batch.STAGES)
def test_cloud_resources_routing_and_shell_syntax(stage, tmp_path):
    job = batch.build_job(arguments(), stage)
    old = old_batch.build_job(arguments(), stage)
    group, old_group = job["taskGroups"][0], old["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    old_policy = old["allocationPolicy"]["instances"][0]["policy"]
    if stage in {"train", "smoke"}:
        assert policy == dict(old_policy, machineType=config.REFERENCE_VM["machine_type"])
        assert group["taskSpec"]["computeResource"] == dict(cpuMilli=16000, memoryMib=60000)
    else:
        assert job["allocationPolicy"] == old["allocationPolicy"]
        assert group["taskSpec"]["computeResource"] == old_group["taskSpec"]["computeResource"]
    assert group["taskCount"] == group["parallelism"] == (3 if stage == "train" else 1)
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRunDuration"] == old_group["taskSpec"]["maxRunDuration"]
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert job["labels"]["experiment"] == "fhp-sdcfr-exp4-24h"
    assert old["labels"]["experiment"] == "fhp-sdcfr-exp3-24h"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1" in script
    if stage == "controller":
        assert "gcp/exp4_sd_cfr_structured_n2_standard16_batch.py orchestrate" in script
    else:
        assert MODULE in script
    if stage == "smoke":
        command = next(line for line in script.splitlines() if "pytest -q" in line)
        files = shlex.split(command)[4:]
        assert "tests/test_exp4_sd_cfr_structured_n2_standard16.py" in files
        assert "tests/test_exp3_sd_cfr_structured.py" in files
        assert all((ROOT / path).is_file() for path in files)
    if stage in {"profile", "evaluate"}:
        assert "--workers 8" in script
    assert "UCV_EXP" not in script and "ucv-source" not in script
    path = tmp_path / f"{stage}.sh"
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.parametrize("command, shared, expected", [
    ("train", train, dict(experiment=config, solver_class=StructuredSingleDeepCFRSolver)),
    ("evaluate", evaluate, dict(experiment=config)),
    ("report", report, dict(experiment=config)),
    ("stress", stress, dict(solver_class=StructuredSingleDeepCFRSolver)),
])
def test_wrappers_reuse_baseline_implementations(command, shared, expected, monkeypatch):
    calls = []
    monkeypatch.setattr(shared, "main", lambda **kwargs: calls.append(kwargs))
    runpy.run_module(f"{MODULE}.{command}", run_name="__main__")
    assert calls == [expected]


def test_controller_cli_is_stdlib_only_and_uses_vm16(tmp_path):
    args = arguments()
    result = subprocess.run([sys.executable, "-S", str(ROOT / batch.EXPERIMENT["batch_script"]),
                             "dry-run", "--project", args.project, "--region", args.region,
                             "--bucket", args.bucket, "--service-account", args.service_account,
                             "--repo-ref", args.repo_ref, "--run-id", args.run_id,
                             "--eval-max-hours", "36"], check=True, capture_output=True, text=True)
    jobs = json.loads(result.stdout)
    assert set(jobs) == set(("controller",) + batch.STAGES)
    assert jobs["train"]["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    assert jobs["evaluate"]["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"


def test_launcher_rejects_stale_ref_before_cloud_submission():
    env = dict(os.environ, REPO_REF="d217b6cddf887867c2833a7f7f26945074f9eb14")
    result = subprocess.run(["bash", str(ROOT / "gcp/run_exp4_sd_cfr_structured_n2_standard16.sh"),
                             "run"], env=env, capture_output=True, text=True)
    assert result.returncode == 64
    assert "Commit and push Experiment 4" in result.stderr


def test_new_worker_manifest_checkpoint_loading_and_reports(tmp_path):
    # Use a child process so Torch thread setup does not affect the other tests.
    subprocess.run([sys.executable, "-m", f"{MODULE}.train", "--seed", "0", "--smoke",
                    "--output-root", str(tmp_path)], check=True, capture_output=True, text=True)
    root = tmp_path / "workers" / config.task_name(0)
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["experiment_name"] == config.EXPERIMENT_NAME
    assert manifest["reference_vm"] == config.REFERENCE_VM
    assert manifest["feature_encoder"] == baseline.FEATURE_ENCODER_METADATA
    assert manifest["torch_threads"] == manifest["interop_threads"] == 1
    assert manifest["config"] == json.loads(json.dumps(baseline.solver_config(True)))
    assert manifest["full_training_states_retained"] is False
    assert (root / "SUCCESS.json").exists()
    checkpoints = json.loads((root / "checkpoint_manifest.json").read_text())
    assert [row["checkpoint_target_hours"] for row in checkpoints] == [6, 12, 18, 24]
    assert len(evaluate.checkpoint_index(tmp_path, smoke=True, experiment=config)) == 4
    with pytest.raises(ValueError):
        evaluate.checkpoint_index(tmp_path, smoke=True, experiment=baseline)
