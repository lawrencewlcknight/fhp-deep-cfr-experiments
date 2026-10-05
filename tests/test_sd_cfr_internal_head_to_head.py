"""Protocol, inference, resumability, reporting and cloud-plan tests."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.retrospective_sd_cfr_internal_evaluation import run as audit
from gcp import sd_cfr_internal_head_to_head_batch as batch


def records():
    return [dict(
        experiment=experiment, experiment_name=spec["experiment_name"],
        algorithm_id=spec["algorithm_id"], seed=seed, training_hours=hour,
        path=f"/{experiment}/{seed}/{hour}", sha256=f"{experiment}-{seed}-{hour}",
        nodes_touched=hour * 10_000 + list(audit.SOURCE_SPECS).index(experiment),
        active_seconds=hour * 3600 + 1, outer_iteration=hour,
        source_commit=spec["repository_commit"], run_manifest_sha256=f"m-{experiment}-{seed}",
        checkpoint_manifest_sha256=f"c-{experiment}-{seed}",
    ) for experiment, spec in audit.SOURCE_SPECS.items()
       for seed in audit.SEEDS for hour in spec["hours"]]


def result(task, mean=1.0, std=2.0):
    n = task["num_deals"]
    row = dict(task=audit.portable(task), elapsed_seconds=1., result=dict(
        num_deal_pairs=n, num_games=2*n, mean_chips_per_hand=mean,
        std_chips_per_pair=std, se_chips_per_hand=std/np.sqrt(n),
        mean_mbb_per_hand=10*mean, policy_a_player0_mean_chips=mean+1,
        policy_a_player1_mean_chips=mean-1,
    ))
    row["result_sha256"] = audit.digest(row)
    return row


def test_prespecified_schedule_covers_targeted_questions_and_all_seed_cells():
    tasks = audit.build_tasks(records(), "implementation")
    assert len(audit.COMPARISONS) == 8
    assert len(tasks) == 720
    assert len({task["task_id"] for task in tasks}) == 720
    assert len({task["cell_id"] for task in tasks}) == 72
    assert sum(task["num_deals"] for task in tasks) == 3_600_000
    assert len({task["evaluation_seed"] for task in tasks}) == 720
    for comparison, _, a_exp, a_hour, b_exp, b_hour in audit.COMPARISONS:
        selected = [task for task in tasks if task["comparison"] == comparison]
        assert {(task["policy_a"]["seed"], task["policy_b"]["seed"])
                for task in selected} == set(itertools.product(audit.SEEDS, repeat=2))
        assert {(task["policy_a"]["experiment"], task["policy_a"]["training_hours"],
                 task["policy_b"]["experiment"], task["policy_b"]["training_hours"])
                for task in selected} == {(a_exp, a_hour, b_exp, b_hour)}
        assert sum(task["num_deals"] for task in selected) == 9 * audit.PAIRS_PER_CELL


def test_probes_cover_every_cell_with_disjoint_randomness():
    production = audit.build_tasks(records(), "implementation")
    for stage, pairs in (("profile", 128), ("smoke", 2)):
        probes = audit.build_tasks(records(), "implementation", stage=stage)
        assert len(probes) == 72
        assert {task["num_deals"] for task in probes} == {pairs}
        assert {task["cell_id"] for task in probes} == {task["cell_id"] for task in production}
        assert not ({task["evaluation_seed"] for task in probes}
                    & {task["evaluation_seed"] for task in production})


def test_source_index_and_path_escape_fail_closed(tmp_path):
    audit.validate_index(records())
    with pytest.raises(ValueError, match="six complete"):
        audit.validate_index(records()[:-1])
    duplicate = records()
    duplicate[0] = duplicate[1]
    with pytest.raises(ValueError, match="six complete"):
        audit.validate_index(duplicate)
    with pytest.raises(ValueError, match="escapes"):
        audit.contained(tmp_path, "../outside")


def test_cached_shards_bind_code_protocol_and_policy_hashes_but_not_paths():
    task = audit.build_tasks(records(), "implementation")[0]
    row = result(task)
    audit.validate_result(row, task)
    for mutate in (
        lambda item: item.update(implementation="changed"),
        lambda item: item["policy_a"].update(sha256="changed"),
        lambda item: item.update(num_deals=1),
    ):
        changed = deepcopy(task)
        mutate(changed)
        with pytest.raises(ValueError, match="Corrupt"):
            audit.validate_result(row, changed)
    moved = deepcopy(task)
    moved["policy_a"]["path"] = "/new/local/root"
    audit.validate_result(row, moved)
    row["result"]["mean_mbb_per_hand"] += 1
    with pytest.raises(ValueError, match="Corrupt"):
        audit.validate_result(row, task)


def test_resume_skips_completed_shards(tmp_path, monkeypatch):
    from deep_cfr_poker.sd_cfr_disk import write_json
    tasks = audit.build_tasks(records(), "implementation")[:2]
    for task in tasks:
        write_json(tmp_path / (task["task_id"] + ".json"), result(task))
    monkeypatch.setattr(audit, "ProcessPoolExecutor",
                        lambda **kwargs: pytest.fail("Unexpected worker launch"))
    assert len(audit.run_tasks(tasks, tmp_path, workers=8)) == 2
    with pytest.raises(TimeoutError):
        audit.run_tasks(audit.build_tasks(records(), "implementation")[:3], tmp_path,
                        workers=8, deadline=0)


def test_pooled_variance_matches_unsharded_samples():
    samples = (np.array([1., 3., 7.]), np.array([-2., 8., 20., 22.]))
    task = audit.build_tasks(records(), "implementation")[0]
    rows = [result(dict(task, num_deals=len(values)), float(values.mean()),
                   float(values.std(ddof=1))) for values in samples]
    pooled = audit.pool_shards(rows)
    full = np.concatenate(samples)
    assert pooled["mean_mbb_per_hand"] == pytest.approx(10*full.mean())
    assert pooled["mc_se_mbb_per_hand"] == pytest.approx(
        10*full.std(ddof=1)/np.sqrt(len(full)))
    assert pooled["hands"] == 14


def test_bootstrap_resamples_three_rows_and_three_columns():
    matrix = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20.]])
    assert audit.cluster_interval(matrix, draws=100_000) == (0, 20)
    assert audit.cluster_interval(np.ones((3, 3))) == (1, 1)
    assert audit.cluster_interval(matrix) == audit.cluster_interval(matrix)
    with pytest.raises(ValueError, match="3x3"):
        audit.cluster_interval(np.ones((2, 3)))


def test_timing_gate_estimates_only_remaining_work():
    tasks = audit.build_tasks(records(), "implementation")
    probes = [result(task) for task in audit.build_tasks(
        records(), "implementation", stage="profile")]
    for probe in probes:
        probe["elapsed_seconds"] = .025 * probe["task"]["num_deals"]
    estimate = audit.cost_estimate(probes, tasks, 8)
    assert estimate["predicted_hours_with_2x_margin"] == pytest.approx(
        (2*3_600_000*.025/8+600)/3600)
    assert audit.cost_estimate(probes, tasks[:1], 8)["remaining_duplicate_pairs"] == 5_000
    with pytest.raises(ValueError, match="Missing"):
        audit.cost_estimate(probes[:-1], tasks, 8)


def test_reporting_requires_complete_production_budget(tmp_path):
    tasks = audit.build_tasks(records(), "implementation")
    rows = [result(task, mean=(task["policy_a"]["seed"]-
                              task["policy_b"]["seed"]+.2)) for task in tasks]
    summaries = audit.report(rows, tmp_path)
    assert len(summaries) == 8
    assert all(row["mean_mbb_per_hand"] == pytest.approx(2) for row in summaries)
    assert sum(row["duplicate_pairs"] for row in summaries) == 3_600_000
    assert len(list(tmp_path.glob("*.png"))) == 3
    with pytest.raises(ValueError, match="Incomplete"):
        audit.report(rows[:-1], tmp_path)
    smoke = [result(task) for task in audit.build_tasks(
        records(), "implementation", stage="smoke")]
    with pytest.raises(ValueError, match="production"):
        audit.report(smoke, tmp_path)


def test_main_timing_gate_never_runs_production_when_over_budget(tmp_path, monkeypatch):
    import sys
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["run", "--sources-root", str(tmp_path / "sources"),
                                      "--output", str(output)])
    monkeypatch.setattr(audit, "validate_sources", lambda *args: records())
    monkeypatch.setattr(audit, "implementation_digest", lambda: "implementation")

    def slow_profile(tasks, *args, **kwargs):
        assert {task["stage"] for task in tasks} == {"profile"}
        rows = [result(task) for task in tasks]
        for row in rows:
            row["elapsed_seconds"] = 1_000_000
        return rows

    monkeypatch.setattr(audit, "run_tasks", slow_profile)
    with pytest.raises(RuntimeError, match="Timing pilot exceeds"):
        audit.main()
    assert audit.read_json(output / "timing_pilot.json")["passed"] is False
    assert not (output / "SUCCESS.json").exists()


def cloud_args():
    values = dict(
        project="project", region="europe-west1", run_id="fhp-sdcfr-h2h-test",
        bucket="gs://output", source_bucket="gs://source", repo_ref="a"*40,
        service_account="runner@test.iam.gserviceaccount.com", max_hours=12,
        resume=False,
    )
    for name, (_, run_id) in batch.SOURCES.items():
        values[f"{name}_run_id"] = run_id
    return argparse.Namespace(**values)


def test_cloud_plan_downloads_only_playable_archives_and_all_six_sources(tmp_path):
    config = batch.job_config(cloud_args())
    policy = config["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "n2-standard-8"
    assert policy["bootDisk"] == {"sizeGb": 200, "type": "pd-balanced"}
    spec = config["taskGroups"][0]["taskSpec"]
    assert spec["maxRetryCount"] == 0 and spec["maxRunDuration"] == "54000s"
    text = spec["runnables"][0]["script"]["text"]
    for name, (algorithm, run_id) in batch.SOURCES.items():
        assert f"$INPUT/{name}/workers" in text
        assert algorithm in text and run_id in text
        assert f"--source-uri {name}=gs://source/{run_id}" in text
    assert "training_state" not in text and "reservoir" not in text
    assert "--stage smoke" in text and "--stage run" in text
    assert "--workers 8" in text and "--max-hours 12" in text
    shell = tmp_path / "job.sh"
    shell.write_text(text)
    subprocess.run(["bash", "-n", str(shell)], check=True)


def test_cloud_plan_validates_output_identity_and_budget():
    args = cloud_args()
    args.run_id = args.exp2_run_id
    with pytest.raises(ValueError, match="new output"):
        batch.job_config(args)
    args = cloud_args()
    args.max_hours = 13
    with pytest.raises(ValueError, match="at most 12"):
        batch.job_config(args)
    args = cloud_args()
    args.repo_ref = "main"
    with pytest.raises(ValueError, match="full pushed"):
        batch.job_config(args)


def test_cloud_access_errors_are_not_interpreted_as_missing(monkeypatch):
    args = cloud_args()
    monkeypatch.setattr(batch, "cloud", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="", stderr="Permission denied"))
    with pytest.raises(RuntimeError, match="Permission"):
        batch.objects_exist(args, "gs://output/run/**")
    monkeypatch.setattr(batch, "cloud", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="", stderr="One or more URLs matched no objects."))
    assert not batch.objects_exist(args, "gs://output/run/**")


def test_source_contracts_are_frozen_to_downloaded_production_runs():
    expected = {
        "exp2": ("sdcfr2-24h-20261002-003338", "fbf3c398ab5804d35515effe3a29b5a6607d44b5"),
        "exp3": ("sdcfr3-24h-20261002-010643", "0a6228671886446d56255fb249c3c6aa07ad7416"),
        "exp4": ("sdcfr4-vm16-20261002-095614", "0a6228671886446d56255fb249c3c6aa07ad7416"),
        "exp5": ("sdcfr5-par8-20261002-102757", "e1118a1e1fd316d40f71c7c898868b5432873151"),
        "exp6": ("sdcfr6-48h-20261002-161544", "bb689252d6b322adb3de6102d095a49c3ed87250"),
        "exp7": ("sdcfr7-distfit-20261003-172011", "dca6d2463792dc452d6aeb0854c181060c7a13a8"),
    }
    assert {name: (spec["run_id"], spec["repository_commit"])
            for name, spec in audit.SOURCE_SPECS.items()} == expected
