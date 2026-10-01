"""Equivalence must precede performance claims."""

import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from deep_cfr_poker.replay import AdvantageMemory, CompactAdvantageReservoirBuffer
from deep_cfr_poker.sd_cfr_optimised import PackedAdvantageReservoirBuffer, OptimisedSingleDeepCFRSolver
from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.single_solver import SingleDeepCFRSolver
from experiments.fhp.exp1_sd_cfr_efficiency.config import solver_config
from experiments.fhp.exp1_sd_cfr_efficiency.run import compare_trees, require_finite, validate_pair


def records():
    rng = np.random.default_rng(5)
    return [AdvantageMemory(rng.integers(0, 2, size=19).astype(np.float32), i + 1,
                            rng.normal(size=3).astype(np.float32)) for i in range(40)]


def buffer(cls, capacity=7):
    return cls(capacity, info_state_size=19, num_actions=3)


@pytest.mark.parametrize("batched", [False, True])
def test_packed_replay_exact_including_replacement_and_sampling(batched):
    outcomes = []
    for cls in (CompactAdvantageReservoirBuffer, PackedAdvantageReservoirBuffer):
        random.seed(18)
        replay = buffer(cls)
        rows = records()
        if batched:
            replay.add_batch({"info_states": np.stack([r.info_state for r in rows]),
                              "targets": np.stack([r.advantage for r in rows]),
                              "iterations": np.array([r.iteration for r in rows])})
        else:
            for row in rows:
                replay.add(row)
        outcomes.append((replay.as_batch(), replay.sample_batch(5),
                         list(replay), random.getstate(), replay.add_calls))
    assert compare_trees(*outcomes)["exact"]
    replay = buffer(PackedAdvantageReservoirBuffer)
    assert replay._info_states.dtype == np.uint8
    assert replay._info_states.shape == (7, 3)
    assert replay._targets.dtype == np.float32
    assert replay._iterations.dtype == np.int32


@pytest.mark.parametrize("bad", [.5, -1, float("nan"), float("inf"), 1 + 1e-12])
def test_pack_fails_closed_before_rng_or_buffer_mutation(bad):
    replay = buffer(PackedAdvantageReservoirBuffer, capacity=1)
    replay.add(records()[0])
    before = random.getstate()
    row = records()[1]._replace(info_state=np.array([bad] + [0.] * 18))
    with pytest.raises(ValueError, match="exactly binary"):
        replay.add(row)
    assert random.getstate() == before
    assert replay.add_calls == 1


@pytest.mark.parametrize("cls", [CompactAdvantageReservoirBuffer, PackedAdvantageReservoirBuffer])
@pytest.mark.parametrize("populated", [False, True])
def test_packed_state_reload_accepts_old_float_and_new_packed(cls, populated):
    original = buffer(cls)
    if populated:
        for row in records():
            original.add(row)
    restored = buffer(PackedAdvantageReservoirBuffer, capacity=3)
    restored.load_state_dict(original.state_dict())
    assert compare_trees(original.as_batch(), restored.as_batch())["exact"]
    assert (original.capacity, original.add_calls) == (restored.capacity, restored.add_calls)
    state = restored.state_dict()
    assert state["feature_encoding"] == "binary_packbits_little_v1"
    assert state["info_states"].dtype == np.uint8


@pytest.mark.parametrize("packed", [False, True])
def test_training_parity_live_weights_and_optimizer(packed):
    outputs = []
    for optimised in (False, True):
        set_seed(41)
        config = solver_config(smoke=True)
        learner = (OptimisedSingleDeepCFRSolver(pack_replay=packed, **config)
                   if optimised else SingleDeepCFRSolver(**config))
        def check_live(solver, player, iteration):
            if optimised:
                eager = solver._advantage_networks[player]
                compiled = solver._scripted_advantages[player]
                observations = torch.tensor(solver.advantage_buffers[player].as_batch()["info_states"])
                with torch.no_grad():
                    assert torch.equal(eager(observations), compiled(observations))
                assert all(a.data_ptr() == b.data_ptr()
                           for a, b in zip(eager.parameters(), compiled.parameters()))
        result = learner.solve(post_player_update_callback=check_live)
        outputs.append({"archive": [[dict(e.state_dict) for e in es]
                                     for es in learner.archive.entries_by_player.values()],
                        "replay": [b.as_batch() for b in learner.advantage_buffers],
                        "optimizer": [o.state_dict() for o in learner._optimizer_advantages],
                        "losses": result.advantage_losses, "nodes": result.nodes_touched,
                        "python_rng": random.getstate(), "np_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state()})
    assert compare_trees(*outputs)["exact"]


def test_equivalence_checker_catches_drift_and_accepts_small_roundoff():
    values = {"weights": np.array([1., 2.]), "nodes": np.array([1, 2]), "diagnostic": float("nan")}
    assert compare_trees(values, values)["exact"]
    candidate = copy.deepcopy(values)
    candidate["weights"][0] += 1e-7
    assert compare_trees(values, candidate)["near"]
    assert not compare_trees(values, candidate)["exact"]
    candidate["weights"][0] += .1
    assert not compare_trees(values, candidate)["near"]
    candidate = copy.deepcopy(values)
    candidate["nodes"][0] += 1
    assert not compare_trees(values, candidate)["near"]
    assert not compare_trees(np.array([np.inf]), np.array([np.inf]))["near"]


def test_benchmark_smoke_subprocesses_and_artifact_contract(tmp_path):
    output = tmp_path / "benchmark"
    subprocess.run([sys.executable, "-m", "experiments.fhp.exp1_sd_cfr_efficiency.run",
                    "--smoke", "--output-dir", str(output)], check=True, capture_output=True)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["equivalence_passed"]
    assert summary["all_bit_identical"]
    assert len(json.loads((output / "runs.json").read_text())) == 3
    assert len(json.loads((output / "comparisons.json").read_text())) == 2
    assert (output / "speed_and_replay_memory.png").exists()
    assert not list(output.rglob("*.pt"))
    assert "Speedup below 1 means slower" in (output / "report.md").read_text()


def test_equivalence_rejects_sampler_drift_and_nonfinite_learners():
    baseline = {"rng": (1, 2), "nodes_by_iteration": [30, 60],
                "replay": [{"capacity": 10, "add_calls": 20,
                            "info_states": np.array([[1., 0.]], dtype=np.float32),
                            "iterations": np.array([2]), "targets": np.array([[.2]])}]}
    assert validate_pair(baseline, baseline)["passed"]
    candidate = copy.deepcopy(baseline)
    candidate["rng"] = (1, 3)
    assert not validate_pair(baseline, candidate)["passed"]
    candidate = copy.deepcopy(baseline)
    candidate["replay"][0]["info_states"][0, 0] += 1e-7
    assert not validate_pair(baseline, candidate)["passed"]
    for value in (np.array([np.nan]), torch.tensor(float("inf")), float("nan")):
        with pytest.raises(RuntimeError, match="Non-finite"):
            require_finite({"value": value})


def test_gcp_job_generation_accepts_pinned_ref_and_explicit_project(tmp_path):
    # No submission/network: intercept gcloud and inspect the generated script.
    fake_gcloud = tmp_path / "gcloud"
    fake_gcloud.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake_gcloud.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "gcp/submit_batch_experiment.sh", "sdcfr-test-only",
         "python -m experiments.fhp.exp1_sd_cfr_efficiency.run --smoke"],
        cwd=root, capture_output=True, text=True, check=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "PROJECT_ID": "test-project", "REGION": "europe-west1",
             "BUCKET": "gs://test-bucket", "SA_EMAIL": "test@test-project.iam.gserviceaccount.com",
             "REPO_REF": "1" * 40})
    assert f"git fetch --depth 1 origin {'1' * 40}" in result.stdout
    assert "git checkout --detach FETCH_HEAD" in result.stdout
    assert "--project\ntest-project" in result.stdout
    assert "fhp-deep-cfr-experiments.git" in result.stdout
