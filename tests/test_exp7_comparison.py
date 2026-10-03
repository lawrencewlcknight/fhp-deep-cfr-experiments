import json
import subprocess
import sys

import numpy as np
import pytest

from deep_cfr_poker.game import serialisable_game_definition
from deep_cfr_poker.sd_cfr_disk import sha256, write_json
from experiments.fhp.exp2_sd_cfr_24h import evaluate
from experiments.fhp.exp7_sd_cfr_distributed_fitting_24h import comparison as c
from experiments.fhp.exp7_sd_cfr_distributed_fitting_24h.benchmark import validation_decision
from gcp import exp2_sd_cfr_24h_batch as shared_batch
from gcp import exp7_sd_cfr_distributed_fitting_24h_batch as batch
from tests.test_exp5_sd_cfr_parallel_24h import arguments


def records(experiment, smoke=False):
    return [dict(experiment=experiment.REPORT_ID, seed=s, training_hours=h,
                 path=f"/{experiment.REPORT_ID}/{s}/{h}.json", sha256=f"{experiment.REPORT_ID}-{s}-{h}",
                 nodes_touched=h * 1000 + s * 100, active_seconds=h * 3600 + 2, outer_iteration=h * 10)
            for s in ((0,) if smoke else (0, 1, 2)) for h in c.config.HOURS]


def tasks(smoke=False):
    left, right = records(c.config, smoke), records(c.baseline, smoke)
    routine = evaluate.make_tasks(left, smoke=smoke, include_lbr=False)
    return c.extend_tasks(routine, left, right, smoke=smoke)


def metadata(root, experiment=c.baseline, smoke=False):
    for seed in ((0,) if smoke else experiment.SEEDS):
        worker = root / "workers" / experiment.task_name(seed)
        worker.mkdir(parents=True)
        targets = [.01 * (i + 1) for i in range(4)] if smoke else experiment.SECONDS
        write_json(worker / "run_manifest.json", dict(experiment_name=experiment.EXPERIMENT_NAME,
            algorithm_id=experiment.ALGORITHM_ID, seed=seed, smoke=smoke,
            config=experiment.solver_config(smoke), execution=experiment.execution_config(seed),
            game=serialisable_game_definition(), feature_encoder=experiment.FEATURE_ENCODER_METADATA,
            reference_vm=experiment.REFERENCE_VM, checkpoints_hours=experiment.HOURS, target_seconds=targets,
            torch_threads=1, interop_threads=1, strategy_weighting="uniform",
            checkpoint_boundary="first_completed_outer_iteration_crossing_threshold",
            time_excludes="checkpoint_serialization_reload_validation_and_upload",
            archive_capture_included_in_training_time=True,
            node_definition="calls_to_external_sampling_traversal_including_terminal_states"))
        write_json(worker / "checkpoint_manifest.json", [dict(checkpoint_target_hours=h,
            checkpoint_target_seconds=s, actual_training_elapsed_seconds=s + 1,
            nodes_touched=100, outer_iteration=1) for h, s in zip(experiment.HOURS, targets)])
        write_json(worker / "SUCCESS.json", dict(seed=seed, checkpoints=4))


def test_production_plan_and_resume_fingerprints():
    plan = tasks()
    cross = [t for t in plan if t["kind"] == "cross_experiment"]
    assert len(plan) == 114 and len(cross) == 36
    assert sum(t["num_deals"] for t in plan) == 3_300_000
    assert all(not t["lbr_enabled"] for t in plan)
    assert {t["num_deals"] for t in cross} == {50_000}
    assert len({t["evaluation_seed"] for t in cross}) == 1
    for hour in c.config.HOURS:
        selected = [t for t in cross if t["training_hours"] == hour]
        assert len(selected) == 9 and sum(t["same_seed"] for t in selected) == 3
        assert all(str(hour) in t["path_a"] and str(hour) in t["path_b"] for t in selected)
    original = evaluate.task_fingerprint(plan)
    plan[-1]["path_b"] = "/another-vm/same-file"
    assert evaluate.task_fingerprint(plan) == original
    plan[-1]["sha_b"] = "changed-baseline"
    assert evaluate.task_fingerprint(plan) != original
    assert len(tasks(True)) == 13


@pytest.mark.parametrize("correct,near,allow,expected", [
    (True, True, False, True), (True, False, False, False),
    (True, False, True, True), (False, True, True, False), (False, False, True, False)])
def test_explicit_drift_acceptance_never_bypasses_correctness(correct, near, allow, expected):
    result = validation_decision(correctness_passed=correct, fits_near=near, allow_trajectory_drift=allow)
    assert result["approved_for_comparative_run"] is expected
    assert result["full_fit_equivalence_passed"] is near
    assert result["numerical_drift_requires_review"] is (not near)


@pytest.mark.parametrize("experiment", [c.baseline, c.config])
def test_reference_metadata_accepts_exact_contract(tmp_path, experiment):
    metadata(tmp_path, experiment)
    provenance = c.validate_metadata(tmp_path, experiment=experiment)
    assert len(provenance) == 3 and all(len(p["manifest_sha256"]) == 64 for p in provenance)


@pytest.mark.parametrize("field,value", [("config", {}), ("reference_vm", {}),
    ("time_excludes", "evaluation"), ("feature_encoder", {}), ("smoke", True)])
def test_reference_metadata_rejects_unmatched_control(tmp_path, field, value):
    metadata(tmp_path)
    path = next(tmp_path.glob("workers/*/run_manifest.json"))
    manifest = json.loads(path.read_text())
    manifest[field] = value
    write_json(path, manifest)
    with pytest.raises(ValueError, match="Incompatible comparison source"):
        c.validate_metadata(tmp_path)


def test_missing_source_or_failed_worker_rejected(tmp_path):
    with pytest.raises(ValueError, match="Missing comparison"):
        c.validate_metadata(tmp_path)
    metadata(tmp_path)
    write_json(next(tmp_path.glob("workers/*")) / "FAILURE.json", {})
    with pytest.raises(ValueError, match="Incomplete comparison"):
        c.validate_metadata(tmp_path)


def fake_results():
    return [dict(task=t, result=dict(num_deal_pairs=t["num_deals"], mean_mbb_per_hand=10. + t["training_seed"]),
                 elapsed_seconds=1.) for t in tasks() if t["kind"] == "cross_experiment"]


def test_report_uses_three_seed_pairs_not_nine_cells(tmp_path):
    c.report(fake_results(), records(c.config), records(c.baseline), tmp_path)
    result = json.loads((tmp_path / "comparison_summary.json").read_text())
    assert result["primary_24h_paired_mbb_per_hand"]["n"] == 3
    assert result["primary_24h_paired_mbb_per_hand"]["mean"] == 11
    assert result["rate_ratio_at_24h"]["mean"] == 1
    assert (tmp_path / "exp7_vs_exp5_head_to_head.png").is_file()
    assert (tmp_path / "exp7_vs_exp5_nodes_by_training_time.png").is_file()
    assert "not nine replicates" in (tmp_path / "comparison_interpretation.txt").read_text()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "hash", "count"])
def test_report_rejects_partial_or_stale_results(tmp_path, mutation):
    data = fake_results()
    if mutation == "missing":
        data.pop()
    elif mutation == "duplicate":
        data.append(data[0])
    elif mutation == "hash":
        data[0]["task"]["sha_b"] = "bad"
    else:
        data[0]["result"]["num_deal_pairs"] = 2
    with pytest.raises(ValueError):
        c.report(data, records(c.config), records(c.baseline), tmp_path)


def test_crossplay_uses_two_playable_archives_not_mixture_or_lbr(tmp_path, monkeypatch):
    from tests.test_sd_cfr_lbr import fixture_archive
    _, left = fixture_archive(tmp_path / "left", structured=True)
    _, right = fixture_archive(tmp_path / "right", structured=True)
    task = next(t for t in tasks(True) if t["kind"] == "cross_experiment")
    task.update(path_a=str(tmp_path / "left/policy.json"), path_b=str(tmp_path / "right/policy.json"))
    task.update(sha_a=sha256(task["path_a"]), sha_b=sha256(task["path_b"]))
    def no_mixture(*args, **kwargs):
        raise AssertionError("Routine crossplay must not reconstruct the full mixture")
    monkeypatch.setattr(evaluate, "BatchedDiskBehaviouralPolicy", no_mixture)
    row = evaluate.execute_task(task)
    assert row["result"]["num_deal_pairs"] == 2
    assert row["result"]["num_games"] == 4
    assert np.isfinite(row["result"]["mean_mbb_per_hand"])
    evaluate._CACHE.clear()


def test_cloud_smoke_and_evaluation_include_read_only_reference(tmp_path):
    args = arguments()
    for stage in ("controller",) + batch.STAGES:
        job = batch.build_job(args, stage)
        script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
        path = tmp_path / f"{stage}.sh"
        path.write_text(script)
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert "export EXP5_RUN_ID=sdcfr5-par8-20261002-102757" in script
        assert "export EVAL_LBR=0" in script
        if stage == "smoke":
            assert "--allow-trajectory-drift" in script
            assert "comparison --validate-reference" in script
            assert "exp5_sd_cfr_parallel_24h.train --seed 0 --smoke" in script
            assert "tests/test_exp7_comparison.py" in script
        elif stage == "train":
            assert "exp5_sd_cfr_parallel_24h.train" not in script
            assert "--resume-state" not in script
            assert job["taskGroups"][0]["taskCount"] == 3
        if stage in ("profile", "evaluate"):
            assert '--reference-source "$INPUT/reference"' in script
            assert "--skip-lbr" in script
            assert "--delete" not in script
    assert shared_batch.evaluation_settings(args) == ("cpu", 8)


def test_launch_preflights_three_reference_workers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["batch.py", "run", "--project", "test", "--region", "europe-west1",
        "--bucket", "bucket", "--service-account", "runner@test", "--repo-ref", "a" * 40,
        "--run-id", "sdcfr7-test", "--reference-run-id", "sdcfr5-reference"])
    for key in ("RESUME_RUN_ID", "EVAL_PROFILE_ONLY", "EVAL_LBR"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    monkeypatch.setattr(shared_batch, "cloud", lambda args, *command, **kw: calls.append(command))
    monkeypatch.setattr(shared_batch, "submit", lambda *a, **kw: calls.append(("submit",)) or "controller")
    shared_batch.main(experiment=batch.EXPERIMENT)
    assert calls[0] == ("iam", "service-accounts", "describe", "runner@test")
    assert len(calls) == 5 and calls[-1] == ("submit",)
    assert all("sdcfr5-reference/workers/" in r[2] and r[2].endswith("/SUCCESS.json") for r in calls[1:4])


def test_profile_includes_every_final_cross_seed_pair(tmp_path, monkeypatch):
    seen = []
    def run(probes, output, *, workers):
        seen.extend(probes)
        return [dict(task=t, elapsed_seconds=t["num_deals"] * .1) for t in probes]
    monkeypatch.setattr(evaluate, "run_tasks", run)
    plan = tasks()
    result = evaluate.profile(plan, tmp_path, workers=8, max_hours=36)
    assert result["passed"] and not result["lbr_enabled"]
    assert len([t for t in seen if t["kind"] == "cross_experiment"]) == 9
    assert all(t["training_hours"] == 24 for t in seen)
    assert result["estimated_elapsed_hours_with_2x_margin"] == pytest.approx(2 * 3_300_000 * .1 / 8 / 3600)
    with pytest.raises(RuntimeError, match="exceeds"):
        evaluate.profile(plan, tmp_path, workers=8, max_hours=.01)
