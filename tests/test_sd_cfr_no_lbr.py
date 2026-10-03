"""Evaluation-only recovery never retrains, deletes archives, or implies zero LBR."""
import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.fhp.exp2_sd_cfr_24h import evaluate
from experiments.fhp.exp2_sd_cfr_24h.report import evaluation_report
from gcp import exp2_sd_cfr_24h_batch as batch


def records():
    return [dict(experiment="sd_cfr_exp2", seed=s, training_hours=h, path=f"sd-{s}-{h}",
                 sha256="abc", nodes_touched=100000 * h) for s in (0, 1, 2) for h in (6, 12, 18, 24)]


def test_no_lbr_preserves_every_rule_and_temporal_match():
    full = evaluate.make_tasks(records())
    tasks = evaluate.make_tasks(records(), include_lbr=False)
    assert len(tasks) == 78
    assert sum(t["kind"] == "rule" for t in tasks) == 60
    assert sum(t["kind"] == "temporal" for t in tasks) == 18
    assert sum(t["num_deals"] for t in tasks) == 1_500_000
    assert all(t["lbr_enabled"] is False and t["lbr_device"] is None for t in tasks)
    def match_spec(task):
        return {k: v for k, v in task.items() if not k.startswith("lbr_")}
    assert [match_spec(t) for t in tasks] == [match_spec(t) for t in full if t["kind"] != "lbr"]
    assert evaluate.task_fingerprint(tasks) != evaluate.task_fingerprint(full)
    assert evaluate.task_fingerprint(tasks) == evaluate.task_fingerprint(
        evaluate.make_tasks(records(), include_lbr=False, lbr_device="cuda"))


def test_profile_and_production_cli_without_any_lbr_probe(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "evaluation_no_lbr"
    monkeypatch.setattr(evaluate, "checkpoint_index", lambda *a, **k: records())
    calls = []
    def run_tasks(tasks, output, *, workers):
        calls.append(tasks)
        assert {t["kind"] for t in tasks} == {"rule", "temporal"}
        assert not any(t.get("validate_lbr_backend") for t in tasks)
        return [dict(task=t, elapsed_seconds=.01 * t["num_deals"],
                     result=dict(mean_mbb_per_hand=10.)) for t in tasks]
    monkeypatch.setattr(evaluate, "run_tasks", run_tasks)
    for mode in ("profile", "run"):
        monkeypatch.setattr(sys, "argv", ["evaluate", mode, "--source", str(source),
                                         "--output", str(output), "--skip-lbr"])
        evaluate.main()
    assert [len(c) for c in calls] == [18, 78]  # 15 rule probes + 3 temporal probes
    profile = json.loads((output / "evaluation_profile.json").read_text())
    assert profile["passed"] and not profile["lbr_enabled"]
    assert profile["lbr_validations"] == []
    assert set(profile["seconds_per_pair"]) == {"rule", "temporal"}
    assert profile["estimated_elapsed_hours_with_2x_margin"] == pytest.approx(2 * 15000 / 8 / 3600)
    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["tasks"] == 78
    assert manifest["lbr_status"] == "omitted_by_configuration"
    assert manifest["lbr_policy"] is None and manifest["lbr_backend"] is None
    assert manifest["evaluated_metrics"] == ["rule", "temporal"]
    assert manifest["source_checkpoints"] == records()
    with (output / "quality_aggregate.csv").open() as f:
        quality = list(csv.DictReader(f))
    assert len(quality) == 4 and {r["metric"] for r in quality} == {"rule_agent_mean"}
    assert {r["n"] for r in quality} == {"3"}
    for filename, count in (("rule_agent_by_seed.csv", 60), ("temporal_crossplay_by_seed.csv", 18),
                            ("temporal_crossplay_aggregate.csv", 6)):
        with (output / filename).open() as f:
            assert len(list(csv.DictReader(f))) == count
    assert not list(output.glob("lbr*"))
    assert "deliberately omitted" in (output / "interpretation.txt").read_text()
    for name in ("temporal_head_to_head.png", "policy_quality_by_training_hours.png", "policy_quality_by_mean_nodes.png"):
        assert (output / name).is_file()
    # A successful no-LBR pilot cannot authorize an expensive LBR run.
    monkeypatch.setattr(sys, "argv", ["evaluate", "run", "--source", str(source), "--output", str(output)])
    with pytest.raises(ValueError, match="matching successful cost profile"):
        evaluate.main()
    assert len(calls) == 2


def test_report_rejects_missing_requested_lbr_or_stale_lbr_output(tmp_path):
    results = [dict(task=t, elapsed_seconds=.1, result=dict(mean_mbb_per_hand=1.))
               for t in evaluate.make_tasks(records(), include_lbr=False)]
    with pytest.raises(ValueError, match="requested LBR scope"):
        evaluation_report(results, records(), tmp_path)
    stale = tmp_path / "lbr_by_seed.csv"
    stale.write_text("old data")
    with pytest.raises(ValueError, match="separate no-LBR"):
        evaluation_report(results, records(), tmp_path, include_lbr=False)
    assert stale.read_text() == "old data"


@pytest.mark.parametrize("structured", [False, True])
def test_ordinary_play_does_not_construct_behavioural_mixture(tmp_path, monkeypatch, structured):
    from tests.test_sd_cfr_lbr import fixture_archive
    from deep_cfr_poker.sd_cfr_disk import sha256
    game, reader = fixture_archive(tmp_path / "archive", structured)
    evaluate._CACHE.clear()
    def forbidden(*args, **kwargs):
        raise AssertionError("Non-LBR play must never construct the full behavioural mixture")
    monkeypatch.setattr(evaluate, "BatchedDiskBehaviouralPolicy", forbidden)
    tasks = evaluate.make_tasks([dict(seed=0, training_hours=h, path=str(reader.path),
                                    sha256=sha256(reader.path), nodes_touched=10) for h in (6, 24)],
                               smoke=True, include_lbr=False)
    before = {p: sha256(p) for p in reader.path.parent.iterdir() if p.is_file()}
    for task in (tasks[0], tasks[-1]):  # real FHP rule and temporal matches
        first = evaluate.execute_task(task)["result"]
        evaluate._CACHE.clear()
        second = evaluate.execute_task(task)["result"]
        assert first == second
    assert before == {p: sha256(p) for p in before}
    evaluate._CACHE.clear()


BUILDERS = ["exp2_sd_cfr_24h", "exp3_sd_cfr_structured_24h",
            "exp4_sd_cfr_structured_n2_standard16", "exp5_sd_cfr_parallel_24h"]


@pytest.mark.parametrize("module", BUILDERS)
def test_cloud_defaults_cpu_no_lbr_and_separate_outputs(tmp_path, module):
    builder = importlib.import_module(f"gcp.{module}_batch")
    args = SimpleNamespace(project="test", region="europe-west1", bucket="gs://test",
                           service_account="runner@test", repo_ref="a" * 40, run_id="sdcfr-test",
                           start_stage="profile", eval_max_hours=12, eval_lbr_device="cuda", eval_workers=8)
    for stage in ("controller", "profile", "evaluate"):
        job = builder.build_job(args, stage)
        script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
        assert "export EVAL_LBR=0" in script
        assert "export EVAL_LBR_DEVICE=cpu" in script
        assert "--delete" not in script and "storage rm" not in script
        if stage == "controller":
            assert "orchestrate --start-stage profile" in script
        else:
            instance = job["allocationPolicy"]["instances"][0]
            assert instance["policy"]["machineType"] == "n2-standard-8"
            assert not instance.get("installGpuDrivers")
            assert "--skip-lbr" in script and "--workers 8" in script
            assert "gs://test/sdcfr-test/evaluation_no_lbr" in script
            assert "gs://test/sdcfr-test/evaluation'" not in script
            assert ".train --seed" not in script and ".report --source" not in script
        path = tmp_path / f"{stage}.sh"
        path.write_text(script)
        subprocess.run(["bash", "-n", str(path)], check=True)
    args.eval_lbr = "1"
    assert batch.lbr_enabled(args)
    assert batch.environment(args)["EVAL_LBR"] == "1"
    assert "--skip-lbr" not in builder.build_job(args, "evaluate")["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]


@pytest.mark.parametrize("module", BUILDERS)
def test_evaluation_only_controller_launches_profile_then_evaluate(monkeypatch, module):
    builder = importlib.import_module(f"gcp.{module}_batch")
    spec = getattr(builder, "EXPERIMENT", batch.DEFAULT_EXPERIMENT)
    monkeypatch.setenv("EVAL_LBR", "0")
    monkeypatch.setenv("EVAL_PROFILE_ONLY", "0")
    monkeypatch.setattr(sys, "argv", ["batch", "orchestrate", "--start-stage", "profile",
                                     "--project", "test", "--region", "europe-west1", "--bucket", "test",
                                     "--service-account", "runner@test", "--repo-ref", "a" * 40,
                                     "--run-id", "sdcfr-test"])
    submitted = []
    monkeypatch.setattr(batch, "cloud", lambda *a, **k: None)
    monkeypatch.setattr(batch, "wait", lambda *a, **k: None)
    monkeypatch.setattr(batch, "submit", lambda args, stage, **k: submitted.append(stage) or "job")
    batch.main(experiment=spec)
    assert submitted == ["profile", "evaluate"]


def test_exp6_default_unchanged_and_invalid_scope_rejected():
    from gcp.exp6_sd_cfr_parallel_48h_batch import EXPERIMENT
    assert batch.lbr_enabled(SimpleNamespace(experiment=EXPERIMENT))
    assert not batch.lbr_enabled(SimpleNamespace(experiment=EXPERIMENT, eval_lbr="0"))
    with pytest.raises(ValueError, match="EVAL_LBR"):
        batch.lbr_enabled(SimpleNamespace(eval_lbr="false"))
