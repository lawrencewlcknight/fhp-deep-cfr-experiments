"""Short diagnostic, not permission to run the expensive experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.fhp.exp7_sd_cfr_distributed_fitting_24h import short_test as short
from gcp import exp7_sd_cfr_short_test_batch as batch
from tests.test_exp5_sd_cfr_parallel_24h import arguments

ROOT = Path(__file__).resolve().parents[1]


def test_short_budget_and_single_vm_job(tmp_path):
    assert sum(short.BUDGETS.values()) == 1800
    assert short.BUDGETS == dict(warmup=300, fitting=1200, end_to_end=300)
    args = arguments()
    args.seed = 0
    job = batch.build_job(args)
    group, = job["taskGroups"]
    assert group["taskCount"] == group["parallelism"] == group["taskCountPerNode"] == 1
    task = group["taskSpec"]
    assert task["maxRetryCount"] == 0
    assert task["maxRunDuration"] == "7200s"
    assert task["computeResource"] == dict(cpuMilli=16000, memoryMib=60000)
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    script = task["runnables"][0]["script"]["text"]
    assert batch.MODULE + ' --output "$OUT/analysis" --seed 0' in script
    assert "batch jobs submit" not in script and "orchestrate" not in script
    assert ".train " not in script and ".evaluate " not in script
    assert "trap finish EXIT" in script and "trap 'exit 143' TERM" in script
    assert "while sleep 60" in script and "--smoke" not in script
    generated = tmp_path / "job.sh"
    generated.write_text(script)
    for path in (generated, ROOT / "gcp/run_exp7_sd_cfr_short_test.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_cloud_dry_run_needs_no_ml_libraries():
    result = subprocess.run([sys.executable, "-S", "gcp/exp7_sd_cfr_short_test_batch.py", "dry-run",
        "--project", "test", "--region", "europe-west1", "--bucket", "test-bucket/",
        "--service-account", "runner@test", "--repo-ref", "a" * 40, "--run-id", "sdcfr7-test"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    job = json.loads(result.stdout)
    assert "gs://test-bucket/sdcfr7-test/analysis" in json.dumps(job)


def test_stale_ref_rejected_before_cloud_submission():
    with pytest.raises(SystemExit, match="refresh REPO_REF"):
        batch.check_pinned_files(SimpleNamespace(repo_ref="0" * 40))


def test_paired_work_accounting():
    rows = [dict(pair=0, arm="central", seconds=8, updates=200, examples=409600, nodes=80),
            dict(pair=0, arm="distributed", seconds=4, updates=200, examples=409600, nodes=72)]
    result = short.pair_metrics(rows)
    assert result["matched_work_speedup"] == 2
    assert result["node_throughput_ratio"] == 1.8
    with pytest.raises(ValueError, match="complete"):
        short.pair_metrics(rows[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        short.pair_metrics(rows + rows[:1])
    with pytest.raises(ValueError, match="Unmatched"):
        short.pair_metrics([rows[0], dict(rows[1], updates=201)])
    with pytest.raises(ValueError, match="positive"):
        short.pair_metrics([rows[0], dict(rows[1], seconds=0)])


def fake_solver():
    networks = [torch.nn.Linear(2, 3), torch.nn.Linear(2, 3)]
    return SimpleNamespace(_advantage_networks=networks,
        _optimizer_advantages=[torch.optim.Adam(n.parameters(), lr=0.004) for n in networks],
        _advantage_network_train_steps=2, _batch_size_advantage=8,
        advantage_buffers=[SimpleNamespace(state_dict=lambda: dict(rows=[1, 2])) for _ in range(2)],
        last_distributed_fit=dict(preparation_seconds=0.001,
            worker_compute_seconds=[0.001] * 8, worker_communication_seconds=[0.001] * 8))


def test_full_fit_drift_is_retained_not_a_correctness_failure(tmp_path, monkeypatch):
    solver, calls, probed = fake_solver(), [], []

    def probes(solver, player):
        probed.append(player)
        return torch.eye(2), [(0, 1, 2)] * 2

    def fit(solver, player, arm):
        calls.append((player, arm))
        with torch.no_grad():
            solver._advantage_networks[player].weight.add_(0.25 if arm == "central" else 0.5)
        return 0.5

    monkeypatch.setattr(short, "probes", probes)
    monkeypatch.setattr(short, "fit", fit)
    result = short.fitting_screen(solver, tmp_path, 0)
    assert probed == [0, 1]
    assert calls == [(0, "central"), (0, "distributed"), (0, "distributed"), (0, "central"),
                     (1, "central"), (1, "distributed"), (1, "distributed"), (1, "central")]
    assert result["pairs"] == 4
    assert not result["all_full_fits_near"]
    assert (tmp_path / "fitting_differences.csv").is_file()
    assert (tmp_path / "fitting_timings.csv").is_file()


def test_nonfinite_update_fails_instead_of_accepting_drift(monkeypatch):
    solver = fake_solver()

    def fit(solver, player, arm):
        for p in solver._advantage_networks[player].parameters():
            p.grad = torch.full_like(p, float("nan"))
        return float("nan")

    monkeypatch.setattr(short, "fit", fit)
    with pytest.raises(RuntimeError, match="Non-finite"):
        short.check_update(solver, 0, 1)
    assert solver._advantage_network_train_steps == 2
    assert not solver._capture_fit_gradients


def test_incorrect_single_update_is_still_a_failed_gate(monkeypatch):
    solver = fake_solver()

    def fit(solver, player, arm):
        for p in solver._advantage_networks[player].parameters():
            p.grad = torch.ones_like(p)
        solver._last_fit_gradient = np.zeros(9, dtype=np.float32)
        return 1.0

    monkeypatch.setattr(short, "fit", fit)
    result = short.check_update(solver, 0, 1)
    assert not result["passed"]
    assert not result["gradient"]["near"]
    assert solver._advantage_network_train_steps == 2


def test_mismatched_sampling_stream_is_fatal(tmp_path, monkeypatch):
    def fit(solver, player, arm):
        if arm == "distributed":
            np.random.random()
        return 1.0

    monkeypatch.setattr(short, "fit", fit)
    monkeypatch.setattr(short, "probes", lambda solver, player: (torch.eye(2), [(0, 1, 2)] * 2))
    with pytest.raises(RuntimeError, match="different sampling"):
        short.fitting_screen(fake_solver(), tmp_path, 0)


def test_end_to_end_matched_work_counterbalances_order(tmp_path):
    calls = []

    def runner(arm, iterations, pair, seed, smoke):
        calls.append((arm, iterations, pair, seed))
        return dict(pair=pair, arm=arm, seconds=iterations * 10, startup_seconds=5,
                    outer_iterations=iterations, root_traversals=iterations * 640, nodes=iterations * 1000), [
                    dict(iteration=iterations, seconds=iterations * 10, nodes=iterations * 1000)]

    summary = short.end_to_end_screen(tmp_path, 300, 0, False, runner=runner)
    assert calls == [("central", 1, 0, 0), ("distributed", 1, 0, 0),
                     ("distributed", 14, 1, 0), ("central", 14, 1, 0)]
    assert summary["pairs"] == 2 and summary["matched_work_speedup"] == 1
    assert summary["central_seconds"] + summary["distributed_seconds"] == 300
    assert summary["startup_seconds"] == 20


@pytest.mark.skipif(os.environ.get("RUN_RAY_SD_CFR_TESTS") != "1", reason="Needs local Ray/Gloo sockets")
def test_real_ray_short_smoke_completes_without_large_artifacts(tmp_path):
    subprocess.run([sys.executable, "-m", batch.MODULE, "--smoke", "--output", str(tmp_path)],
                   cwd=ROOT, check=True, timeout=300)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed"] and summary["correctness_checks_passed"]
    assert not summary["long_run_authorized"]
    assert summary["fitting"]["pairs"] >= 4
    assert summary["end_to_end"]["pairs"] >= 1
    assert all(p.suffix in {".csv", ".json"} for p in tmp_path.iterdir())
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) < 1_000_000
