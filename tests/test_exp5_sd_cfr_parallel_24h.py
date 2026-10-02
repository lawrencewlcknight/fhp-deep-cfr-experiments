"""Experiment 4/5 control contract and opt-in production-worker integration."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from deep_cfr_poker.sd_cfr_parallel import ParallelStructuredSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_structured import StructuredSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_disk import DiskArchiveReader
from deep_cfr_poker.seeding import set_seed
from experiments.fhp.exp1_sd_cfr_efficiency.run import compare_trees
from experiments.fhp.exp2_sd_cfr_24h import train, evaluate, report, stress
from experiments.fhp.exp4_sd_cfr_structured_n2_standard16 import config as baseline
from experiments.fhp.exp5_sd_cfr_parallel_24h import config
from gcp import exp4_sd_cfr_structured_n2_standard16_batch as old_batch
from gcp import exp5_sd_cfr_parallel_24h_batch as batch

ROOT = Path(__file__).resolve().parents[1]
MODULE = "experiments.fhp.exp5_sd_cfr_parallel_24h"


def arguments():
    return SimpleNamespace(project="test", region="europe-west1", bucket="gs://test",
                           service_account="runner@test", repo_ref="a" * 40,
                           run_id="sdcfr5-test", start_stage="smoke", eval_max_hours=36)


def test_production_learner_budget_hardware_and_evaluation_are_unchanged():
    assert config.solver_config() == baseline.solver_config()
    assert config.solver_config()["num_traversals"] == 320
    assert config.TRAVERSAL_WORKERS == 8
    assert config.solver_config()["num_traversals"] // config.TRAVERSAL_WORKERS == 40
    assert config.solver_config(True) == dict(baseline.solver_config(True), num_traversals=8)
    assert config.SEEDS == baseline.SEEDS == (0, 1, 2)
    assert config.HOURS == baseline.HOURS == (6, 12, 18, 24)
    assert config.SECONDS == baseline.SECONDS and config.SECONDS[-1] == 86400
    assert config.ITERATION_CAP == baseline.ITERATION_CAP
    assert config.REFERENCE_VM == baseline.REFERENCE_VM
    assert config.FEATURE_ENCODER_METADATA == baseline.FEATURE_ENCODER_METADATA
    assert config.EXPERIMENT_NAME != baseline.EXPERIMENT_NAME
    assert config.REPORT_ID != baseline.REPORT_ID
    for seed in config.SEEDS:
        assert config.task_name(seed) != baseline.task_name(seed)
        execution = config.execution_config(seed)
        assert execution["parallel_run_seed"] == seed
        assert execution["parallel_num_workers"] == 8
        assert execution["parallel_backend"] == "ray"
    for fn in (config.execution_config, config.task_name):
        with pytest.raises(ValueError):
            fn(3)


@pytest.mark.parametrize("stage", ("controller",) + batch.STAGES)
def test_same_cloud_resources_and_real_parallel_smoke_gate(stage, tmp_path):
    job = batch.build_job(arguments(), stage)
    old = old_batch.build_job(arguments(), stage)
    assert job["allocationPolicy"] == old["allocationPolicy"]
    group, old_group = job["taskGroups"][0], old["taskGroups"][0]
    assert group["taskSpec"]["computeResource"] == old_group["taskSpec"]["computeResource"]
    assert group["taskSpec"]["maxRunDuration"] == old_group["taskSpec"]["maxRunDuration"]
    assert group["taskCount"] == group["parallelism"] == (3 if stage == "train" else 1)
    assert group["taskCountPerNode"] == 1
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1" in script
    if stage == "controller":
        assert batch.EXPERIMENT["batch_script"] in script
    else:
        assert MODULE in script
    if stage == "smoke":
        assert "RUN_RAY_SD_CFR_TESTS=1 python -m pytest" in script
        assert "-k real_ray" in script
        assert "tests/test_sd_cfr_parallel_efficiency.py" in script
        old_script = old_group["taskSpec"]["runnables"][0]["script"]["text"]
        assert "RUN_RAY_SD_CFR_TESTS" not in old_script
    if stage == "train":
        assert config.ALGORITHM_ID in script
    if stage in {"profile", "evaluate"}:
        assert "--workers 8" in script  # Evaluation workers, not training actors.
    assert "ucv-source" not in script
    path = tmp_path / f"{stage}.sh"
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.parametrize("command, shared, expected", [
    ("train", train, dict(experiment=config, solver_class=ParallelStructuredSingleDeepCFRSolver)),
    ("evaluate", evaluate, dict(experiment=config)),
    ("report", report, dict(experiment=config)),
    ("stress", stress, dict(solver_class=StructuredSingleDeepCFRSolver)),
])
def test_shared_protocol_wrappers(command, shared, expected, monkeypatch):
    calls = []
    monkeypatch.setattr(shared, "main", lambda **kwargs: calls.append(kwargs))
    runpy.run_module(f"{MODULE}.{command}", run_name="__main__")
    assert calls == [expected]


def test_stdlib_controller_dry_run_and_stale_ref_guard():
    args = arguments()
    result = subprocess.run([sys.executable, "-S", str(ROOT / batch.EXPERIMENT["batch_script"]),
                             "dry-run", "--project", args.project, "--region", args.region,
                             "--bucket", args.bucket, "--service-account", args.service_account,
                             "--repo-ref", args.repo_ref, "--run-id", args.run_id],
                            check=True, capture_output=True, text=True)
    jobs = json.loads(result.stdout)
    assert jobs["train"]["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    env = dict(os.environ, REPO_REF="0a6228671886446d56255fb249c3c6aa07ad7416")
    result = subprocess.run(["bash", str(ROOT / "gcp/run_exp5_sd_cfr_parallel_24h.sh"), "run"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 64
    assert "Commit and push Experiment 5" in result.stderr


@pytest.mark.parametrize("fail", [False, True])
def test_shared_runner_passes_execution_records_metadata_and_closes(tmp_path, monkeypatch, fail):
    # Test the runner lifecycle without process/socket requirements. The
    # production Ray path has its own integration test below.
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *_: None)
    instances = []
    def factory(**kwargs):
        assert {key: kwargs[key] for key in config.execution_config(0)} == config.execution_config(0)
        kwargs["parallel_backend"] = "serial"
        solver = ParallelStructuredSingleDeepCFRSolver(**kwargs)
        instances.append(solver)
        if fail:
            def broken(**_):
                raise RuntimeError("injected collector failure")
            monkeypatch.setattr(solver, "solve", broken)
        return solver
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            train.run_worker(tmp_path, 0, smoke=True, experiment=config, solver_class=factory)
    else:
        train.run_worker(tmp_path, 0, smoke=True, experiment=config, solver_class=factory)
    assert len(instances) == 1 and instances[0]._closed and not instances[0]._workers
    root = tmp_path / "workers" / config.task_name(0)
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["execution"] == config.execution_config(0)
    assert manifest["config"] == json.loads(json.dumps(config.solver_config(True)))
    assert manifest["parallel_execution"]["workers"] == 8
    assert manifest["parallel_startup_included_in_training_time"] is True
    assert manifest["peak_rss_scope"] == "central_learner_only_excludes_ray_and_actors"
    assert (root / ("FAILURE.json" if fail else "SUCCESS.json")).exists()
    if not fail:
        checkpoints = json.loads((root / "checkpoint_manifest.json").read_text())
        assert [c["checkpoint_target_hours"] for c in checkpoints] == [6, 12, 18, 24]
        reader = DiskArchiveReader(root / checkpoints[-1]["path"], instances[0]._game)
        assert reader.contract["metadata"]["parallel_execution"]["workers"] == 8
        # A serial execution cannot be silently passed off as Experiment 5.
        with pytest.raises(ValueError, match="parallel execution"):
            evaluate.checkpoint_index(tmp_path, smoke=True, experiment=config)


@pytest.mark.skipif(os.environ.get("RUN_RAY_SD_CFR_TESTS") != "1",
                   reason="Opt-in eight-actor integration requires local processes and sockets")
def test_real_ray_eight_workers_production_partition_matches_serial():
    import ray
    from tests.test_sd_cfr_parallel import training_signature
    assert not ray.is_initialized()
    outcomes = []
    # Keep the production encoder, networks, 320 traversals and eight workers.
    # Shorten optimisation and replay only; this is a correctness gate, not a
    # statistically meaningful speed benchmark or the 24-hour experiment.
    learner = dict(config.solver_config(True), num_iterations=2, num_traversals=320)
    for backend in ("serial", "ray"):
        set_seed(0)
        with ParallelStructuredSingleDeepCFRSolver(
                **learner, **dict(config.execution_config(0), parallel_backend=backend)) as solver:
            result = solver.solve()
            assert solver.last_parallel_collection["worker_traversals"] == [40] * 8
            assert len(solver._workers) == 8
            if backend == "ray":
                assert all(s["torch_threads"] == 1 for s in ray.get([w.ping.remote() for w in solver._workers]))
            outcomes.append(training_signature(solver, result))
        assert not ray.is_initialized()
    comparison = compare_trees(*outcomes)
    assert comparison["exact"], comparison


@pytest.mark.skipif(os.environ.get("RUN_RAY_SD_CFR_TESTS") != "1",
                   reason="Opt-in actual Experiment 5 training CLI and policy reload")
def test_real_ray_experiment_smoke_cli_and_evaluation_contract(tmp_path):
    subprocess.run([sys.executable, "-m", f"{MODULE}.train", "--seed", "0", "--smoke",
                    "--output-root", str(tmp_path)], check=True, capture_output=True, text=True,
                   timeout=300)
    records = evaluate.checkpoint_index(tmp_path, smoke=True, experiment=config)
    assert len(records) == 4
    root = tmp_path / "workers" / config.task_name(0)
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["parallel_execution"]["backend"] == "ray_parallel_sd_cfr"
    assert manifest["parallel_execution"]["workers"] == 8
    subprocess.run([sys.executable, "-m", f"{MODULE}.evaluate", "smoke",
                    "--source", str(tmp_path), "--output", str(tmp_path / "evaluation"),
                    "--workers", "2"], check=True, capture_output=True, text=True, timeout=300)
    evaluated = json.loads((tmp_path / "evaluation/evaluation_manifest.json").read_text())
    assert evaluated["status"] == "complete"
    assert evaluated["experiment_name"] == config.EXPERIMENT_NAME
    assert evaluated["evaluated_hours"] == [6, 24]
    assert (tmp_path / "evaluation/policy_quality_by_mean_nodes.png").is_file()
    manifest["execution"]["parallel_num_workers"] = 1
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="execution configuration"):
        evaluate.checkpoint_index(tmp_path, smoke=True, experiment=config)
