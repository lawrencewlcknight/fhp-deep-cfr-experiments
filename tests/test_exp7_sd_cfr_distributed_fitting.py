import copy
import os
import random
import subprocess

import numpy as np
import pytest
import torch

from deep_cfr_poker.sd_cfr_distributed import DistributedFittingSingleDeepCFRSolver
from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.solver import DeepCFRSolver
from experiments.fhp.exp1_sd_cfr_efficiency.run import compare_trees
from experiments.fhp.exp5_sd_cfr_parallel_24h import config as baseline
from experiments.fhp.exp7_sd_cfr_distributed_fitting_24h import config
from gcp import exp5_sd_cfr_parallel_24h_batch as old_batch
from gcp import exp7_sd_cfr_distributed_fitting_24h_batch as batch
from tests.test_exp5_sd_cfr_parallel_24h import arguments


def test_exact_exp5_contract_except_fitting():
    assert config.solver_config() == baseline.solver_config()
    assert config.solver_config(True) == baseline.solver_config(True)
    assert config.SEEDS == (0, 1, 2)
    assert config.HOURS == (6, 12, 18, 24)
    assert config.REFERENCE_VM == baseline.REFERENCE_VM
    assert config.FEATURE_ENCODER_METADATA == baseline.FEATURE_ENCODER_METADATA
    for seed in config.SEEDS:
        assert config.execution_config(seed) == dict(baseline.execution_config(seed), distributed_fitting=True)
        assert config.task_name(seed) != baseline.task_name(seed)
    with pytest.raises(ValueError):
        config.task_name(3)


@pytest.mark.parametrize("stage", ("controller",) + batch.STAGES)
def test_cloud_resources_stages_and_smoke_gate(stage, tmp_path):
    job = batch.build_job(arguments(), stage)
    control = old_batch.build_job(arguments(), stage)
    assert job["allocationPolicy"] == control["allocationPolicy"]
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == (3 if stage == "train" else 1)
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["computeResource"] == control["taskGroups"][0]["taskSpec"]["computeResource"]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    if stage == "smoke":
        assert "RUN_RAY_SD_CFR_TESTS=1 python -m pytest" in script
        assert "exp7_sd_cfr_distributed_fitting_24h.benchmark" in script
        assert "--repeats 3" in script
    path = tmp_path / (stage + ".sh")
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.skipif(os.environ.get("RUN_RAY_SD_CFR_TESTS") != "1",
                   reason="Real eight-process Gloo/Ray integration requires sockets")
@pytest.mark.parametrize("batch_size", [3, 17, 2048])
def test_real_ray_global_minibatch_gradients_adam_and_alternation(batch_size):
    torch.set_num_threads(1)
    set_seed(13)
    kwargs = dict(config.solver_config(True), memory_capacity=3000,
                  num_traversals=6400 if batch_size == 2048 else 320, batch_size_advantage=batch_size,
                  advantage_network_train_steps=1, num_iterations=1)
    with DistributedFittingSingleDeepCFRSolver(**kwargs, **config.execution_config(0)) as solver:
        solver._capture_fit_gradients = True
        # Both actual production-player phases: player 1 sees player 0's fit.
        for player in (0, 1):
            solver._collect_traversals_for_player(player)
            assert len(solver.advantage_buffers[player]) >= batch_size
            network = solver._advantage_networks[player]
            optimizer = solver._optimizer_advantages[player]
            for fit in range(2):  # second fit must retain Adam's nonzero moments
                initial = copy.deepcopy((network.state_dict(), optimizer.state_dict()))
                random.seed(42 + fit)
                expected_loss = DeepCFRSolver._learn_advantage_network(solver, player)
                expected_gradient = np.concatenate([p.grad.numpy().reshape(-1) for p in network.parameters()])
                expected = copy.deepcopy((network.state_dict(), optimizer.state_dict()))
                expected_rng = random.getstate()
                network.load_state_dict(initial[0])
                optimizer.load_state_dict(initial[1])
                random.seed(42 + fit)
                loss = solver._learn_advantage_network(player)
                report = compare_trees(expected, (network.state_dict(), optimizer.state_dict()),
                                       atol=3e-6, rtol=3e-5)
                assert report["near"], report
                np.testing.assert_allclose(solver._last_fit_gradient, expected_gradient, atol=2e-6, rtol=3e-5)
                assert loss == pytest.approx(expected_loss, rel=2e-5, abs=3e-6)
                assert random.getstate() == expected_rng
                assert solver.last_distributed_fit["examples"] == min(batch_size, len(solver.advantage_buffers[player]))
            solver.archive.capture_from_solver(solver, player, solver._iteration)
        assert solver.distributed_fit_totals["phases"] == 4
        assert len(solver._workers) == 8
        assert solver.archive.entries_by_player[0] and solver.archive.entries_by_player[1]


@pytest.mark.skipif(os.environ.get("RUN_RAY_SD_CFR_TESTS") != "1",
                   reason="Real worker failure needs local Ray processes")
def test_real_ray_dead_actor_fails_closed():
    import ray
    set_seed(19)
    with DistributedFittingSingleDeepCFRSolver(
            **config.solver_config(True), **config.execution_config(0)) as solver:
        solver._collect_traversals_for_player(0)
        solver._start_fitting()
        previous = copy.deepcopy(solver._advantage_networks[0].state_dict())
        ray.kill(solver._workers[0], no_restart=True)
        with pytest.raises(Exception):
            solver._learn_advantage_network(0)
        assert solver._closed and not solver._workers
        assert compare_trees(previous, solver._advantage_networks[0].state_dict())["exact"]
