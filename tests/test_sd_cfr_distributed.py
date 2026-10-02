"""Same global data/normalisation/RNG, synchronized gradients and Adam."""
import random

import numpy as np
import pytest
import torch

from deep_cfr_poker.sd_cfr_distributed import DistributedFittingSingleDeepCFRSolver
from deep_cfr_poker.seeding import set_seed
from experiments.fhp.exp1_sd_cfr_efficiency.run import compare_trees


SMALL = dict(num_iterations=1, num_traversals=8, memory_capacity=64,
             batch_size_advantage=11, advantage_network_train_steps=3,
             advantage_network_layers=(8, 8), evaluation_interval=1,
             parallel_num_workers=8, parallel_backend="serial")


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("size", [3, 11, 100])
def test_prefetch_matches_every_global_batch_and_rng(size):
    set_seed(39)
    with DistributedFittingSingleDeepCFRSolver(**dict(SMALL, batch_size_advantage=size)) as solver:
        solver._collect_traversals_for_player(0)
        random.seed(823)
        expected = []
        for _ in range(SMALL["advantage_network_train_steps"]):
            batch = solver._draw_advantage_batch(0, solver.advantage_buffers[0])
            batch["targets"] = solver._process_advantage_targets(batch["targets"], 0)
            batch["iterations"] = batch["iterations"].astype(np.float32)
            expected.append(batch)
        expected_rng = random.getstate()
        random.seed(823)
        shards, n = solver._prepare_fit(0)
        assert n == min(size, len(solver.advantage_buffers[0]))
        assert random.getstate() == expected_rng
        for step, expected_batch in enumerate(expected):
            actual = {key: np.concatenate([s[key][step] for s in shards])
                      for key in expected_batch}
            actual["info_states"] = solver.advantage_buffers[0]._unpack(actual["info_states"])
            assert compare_trees(actual, expected_batch)["exact"]


@pytest.mark.parametrize("override", [
    dict(distributed_fitting=False), dict(parallel_ray_address="auto"),
    dict(advantage_replay_sampling="priority_abs_adv"), dict(batch_size_advantage=0),
    dict(advantage_network_type="mlp"), dict(advantage_network_train_steps=0),
])
def test_unsupported_recipes_fail_closed(override):
    with pytest.raises(ValueError):
        DistributedFittingSingleDeepCFRSolver(**dict(SMALL, **override))


def test_empty_fit_does_not_start_actors():
    with DistributedFittingSingleDeepCFRSolver(**dict(SMALL, parallel_backend="ray")) as solver:
        assert solver._learn_advantage_network(0) is None
        assert not solver._workers


def test_real_ray_fitting_checks_are_in_exp7_cloud_smoke():
    # Keep costly real-process coverage in the experiment test module, which
    # the shared cloud smoke explicitly runs with RUN_RAY_SD_CFR_TESTS=1.
    from gcp.exp7_sd_cfr_distributed_fitting_24h_batch import EXPERIMENT
    assert EXPERIMENT["parallel_smoke"] and EXPERIMENT["fitting_benchmark"]
