import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from deep_cfr_poker.sd_cfr_disk import DiskArchiveReader
from deep_cfr_poker.sd_cfr_parallel import ParallelStructuredSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_training_state import inspect_training_state
from experiments.fhp.exp2_sd_cfr_24h import train, evaluate, report
from experiments.fhp.exp5_sd_cfr_parallel_24h import config as baseline
from experiments.fhp.exp6_sd_cfr_parallel_48h import config
from gcp import exp5_sd_cfr_parallel_24h_batch as old_batch
from gcp import exp6_sd_cfr_parallel_48h_batch as batch

ROOT = Path(__file__).resolve().parents[1]
MODULE = "experiments.fhp.exp6_sd_cfr_parallel_48h"


def arguments(resume=False):
    return SimpleNamespace(project="test", region="europe-west1", bucket="gs://test",
                           service_account="runner@test", repo_ref="a" * 40,
                           run_id="sdcfr6-test", start_stage="smoke", eval_max_hours=36,
                           resume_run_id="sdcfr6-original" if resume else None, additional_hours=24)


def test_only_horizon_and_retention_change():
    assert config.solver_config() == baseline.solver_config()
    assert config.solver_config(True) == baseline.solver_config(True)
    assert config.SEEDS == baseline.SEEDS == (0, 1, 2)
    assert config.REFERENCE_VM == baseline.REFERENCE_VM
    assert config.HOURS == tuple(range(6, 49, 6))
    assert config.SECONDS[-1] == 48 * 3600
    assert config.RETAIN_FINAL_TRAINING_STATE
    assert config.REPORT_ID != baseline.REPORT_ID
    for seed in config.SEEDS:
        assert config.execution_config(seed) == baseline.execution_config(seed)
        assert config.task_name(seed) != baseline.task_name(seed)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("stage", ("controller",) + batch.STAGES)
def test_cloud_resources_timeout_and_resume_routing(stage, resume, tmp_path):
    job = batch.build_job(arguments(resume), stage)
    old = old_batch.build_job(arguments(), stage)
    assert job["allocationPolicy"] == old["allocationPolicy"]
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == (3 if stage == "train" else 1)
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["computeResource"] == old["taskGroups"][0]["taskSpec"]["computeResource"]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert job["labels"]["experiment"] == "fhp-sdcfr-exp6-48h"
    if stage == "train":
        assert group["taskSpec"]["maxRunDuration"] == "259200s"
        assert ("--resume-state" in script) == resume
        assert config.ALGORITHM_ID in script
        if resume:
            assert 'sdcfr6-original/workers/"$TASK"' in script
            assert "--additional-hours 24" in script
            assert "gcloud storage cat" in script and 'uv python install "$TARGET_PYTHON_VERSION"' in script
    if stage in {"aggregate", "profile", "evaluate"}:
        assert "--exclude='(^|.*/)training_state/.*|.*[.]tmp$'" in script
    if stage == "smoke":
        assert "tests/test_sd_cfr_training_state.py" in script
        assert "RUN_RAY_SD_CFR_TESTS=1" in script
    path = tmp_path / f"{stage}.sh"
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_evaluation_covers_eight_checkpoints_all_temporal_pairs_and_profiles_latest(tmp_path, monkeypatch):
    rows = [dict(seed=s, training_hours=h, path=f"s{s}/time_{h:02}h.json", sha256="a",
                 nodes_touched=h * 100, experiment=config.REPORT_ID, active_seconds=h * 3600)
            for s in config.SEEDS for h in config.HOURS]
    tasks = evaluate.make_tasks(rows)
    assert len([t for t in tasks if t["kind"] == "rule"]) == 3 * 8 * 5
    assert len([t for t in tasks if t["kind"] == "lbr"]) == 3 * 8 * 100
    assert len([t for t in tasks if t["kind"] == "temporal"]) == 3 * 28
    assert {t["training_hours"] for t in tasks} == set(config.HOURS)
    assert {t["training_hours"] for t in evaluate.make_tasks(rows, smoke=True)} == {6, 48}
    probes = []
    def fake_tasks(selected, *_args, **_kwargs):
        probes.extend(selected)
        return [dict(task=t, elapsed_seconds=0.001, lbr_backend_validation=dict(passed=True)) for t in selected]
    monkeypatch.setattr(evaluate, "run_tasks", fake_tasks)
    evaluate.profile(tasks, tmp_path, workers=8, max_hours=36)
    assert {t["training_hours"] for t in probes} == {48}
    monkeypatch.setattr(evaluate, "checkpoint_index", lambda *_args, **_kw: rows)
    report.training_report(tmp_path, tmp_path / "analysis", experiment=config)
    assert json.loads((tmp_path / "analysis/training_summary.json").read_text())["checkpoints"] == 24


def test_runner_saves_only_final_state_and_resumes_with_all_prior_policies(tmp_path, monkeypatch):
    # Real learner and archive; serial actor backend avoids socket requirements.
    monkeypatch.setattr(config, "execution_config", lambda seed: dict(
        baseline.execution_config(seed), parallel_backend="serial"))
    instances = []
    def factory(**kw):
        solver = ParallelStructuredSingleDeepCFRSolver(**kw)
        instances.append(solver)
        return solver
    old = train.run_worker(tmp_path / "old", 0, smoke=True, experiment=config, solver_class=factory)
    saved = old / "training_state/manifest.json"
    meta = inspect_training_state(saved)
    assert meta["hours"] == list(config.HOURS)
    assert len(list(old.rglob("learner.pt"))) == 1
    assert len(list(old.rglob("replay_*.npy"))) == 6
    new = train.run_worker(tmp_path / "new", 0, smoke=True, experiment=config, solver_class=factory,
                           resume_state=saved, additional_hours=24)
    rows = json.loads((new / "checkpoint_manifest.json").read_text())
    assert [r["checkpoint_target_hours"] for r in rows] == list(range(6, 73, 6))
    assert rows[:8] == json.loads((old / "checkpoint_manifest.json").read_text())
    assert instances[-1]._nodes_touched > meta["nodes_touched"]
    assert all(s._closed for s in instances)
    assert len(list(new.rglob("learner.pt"))) == 1
    assert (new / "SUCCESS.json").exists() and not (new / "FAILURE.json").exists()
    for row in rows:
        assert DiskArchiveReader(new / row["path"], instances[-1]._game).count == row["outer_iteration"]
    assert config.checkpoint_hours(json.loads((new / "run_manifest.json").read_text())) == tuple(range(6, 73, 6))
    import csv
    with (new / "solver_diagnostics.csv").open() as stream:
        diagnostics = list(csv.DictReader(stream))
    assert [int(r["iteration"]) for r in diagnostics] == sorted({int(r["iteration"]) for r in diagnostics})
    assert [float(r["wall_clock_seconds"]) for r in diagnostics] == sorted(float(r["wall_clock_seconds"]) for r in diagnostics)
    with pytest.raises(ValueError, match="Non-empty"):
        train.run_worker(tmp_path / "old", 0, smoke=True, experiment=config, solver_class=factory)
    with pytest.raises(ValueError, match="both"):
        train.run_worker(tmp_path / "bad", 0, smoke=True, experiment=config, resume_state=saved)


def test_active_clock_continuation_preserves_cumulative_budget():
    now = [100.0]
    clock = train.ActiveClock(lambda: now[0], active=48 * 3600 + 17, elapsed=49 * 3600)
    assert clock() == 48 * 3600 + 17
    now[0] += 60
    with clock.paused():
        now[0] += 300
    assert clock() == 48 * 3600 + 77
    assert clock.elapsed() == 49 * 3600 + 360


def test_stdlib_controller_and_launcher_guards():
    args = arguments()
    result = subprocess.run([sys.executable, "-S", str(ROOT / batch.EXPERIMENT["batch_script"]), "dry-run",
                             "--project", args.project, "--region", args.region, "--bucket", args.bucket,
                             "--service-account", args.service_account, "--repo-ref", args.repo_ref,
                             "--run-id", args.run_id], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["train"]["taskGroups"][0]["taskCount"] == 3
    result = subprocess.run(["bash", str(ROOT / "gcp/run_exp6_sd_cfr_parallel_48h.sh"), "run"],
                            env=dict(os.environ, REPO_REF="0a6228671886446d56255fb249c3c6aa07ad7416"),
                            capture_output=True, text=True)
    assert result.returncode == 64 and "Commit and push Experiment 6" in result.stderr


@pytest.mark.skipif(os.environ.get("RUN_RAY_SD_CFR_TESTS") != "1", reason="Eight-actor integration needs processes and sockets")
def test_real_ray_save_reload_continue_exact(tmp_path):
    from tests.test_sd_cfr_training_state import test_continuation_is_bit_exact_including_reservoir_replacement_and_partial_capacity
    test_continuation_is_bit_exact_including_reservoir_replacement_and_partial_capacity(tmp_path, 17, backend="ray")


@pytest.mark.skipif(os.environ.get("RUN_RAY_SD_CFR_TESTS") != "1", reason="Actual eight-actor experiment CLI")
def test_real_ray_smoke_cli_resume_and_evaluation(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    base = [sys.executable, "-m", MODULE + ".train", "--seed", "0", "--smoke"]
    subprocess.run(base + ["--output-root", str(old)], check=True, capture_output=True, text=True, timeout=300)
    assert len(evaluate.checkpoint_index(old, smoke=True, experiment=config)) == 8
    state = old / "workers" / config.task_name(0) / "training_state/manifest.json"
    subprocess.run(base + ["--output-root", str(new), "--resume-state", str(state), "--additional-hours", "24"],
                   check=True, capture_output=True, text=True, timeout=300)
    assert len(evaluate.checkpoint_index(new, smoke=True, experiment=config)) == 12
    subprocess.run([sys.executable, "-m", MODULE + ".evaluate", "smoke", "--source", str(old),
                    "--output", str(tmp_path / "evaluation"), "--workers", "2"],
                   check=True, capture_output=True, text=True, timeout=300)
    manifest = json.loads((tmp_path / "evaluation/evaluation_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["evaluated_hours"] == [6, 48]
