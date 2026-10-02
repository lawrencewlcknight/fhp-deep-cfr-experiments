"""Parallel SD-CFR: central sampling, alternating phases and real actor parity."""
import copy
import os
import random

import numpy as np
import pytest
import torch

from deep_cfr_poker import ParallelSingleDeepCFRSolver, ParallelStructuredSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive, DiskArchiveReader, DiskSampledPolicy
from deep_cfr_poker.sd_cfr_parallel import SDCFRTraversalWorker, isolated_rng, phase_seed
from deep_cfr_poker.sd_cfr_optimised import OptimisedSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_structured import StructuredSingleDeepCFRSolver
from deep_cfr_poker.seeding import set_seed
from experiments.fhp.exp1_sd_cfr_efficiency.run import compare_trees


SMALL = dict(num_iterations=3, num_traversals=7, advantage_network_layers=(8, 8),
             memory_capacity=17, batch_size_advantage=2,
             advantage_network_train_steps=2, evaluation_interval=1)
CLASSES = [ParallelSingleDeepCFRSolver, ParallelStructuredSingleDeepCFRSolver]


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def weights(solver):
    return [{k: v.detach().clone() for k, v in network.state_dict().items()}
            for network in solver._advantage_networks]


@pytest.mark.parametrize("cls", CLASSES)
def test_worker_emits_every_sample_and_central_reservoir_matches_scalar(cls):
    set_seed(71)
    with cls(parallel_backend="serial", parallel_chunk_rows=3, **SMALL) as learner:
        worker = SDCFRTraversalWorker(worker_index=1, **learner._worker_kwargs)
        before = rng_state()
        row = worker.collect(7, 0, weights(learner), 1)
        assert compare_trees(before, rng_state())["exact"]
        assert row["rows"] > learner.advantage_buffers[0].capacity
        assert all(len(c["iterations"]) <= 3 for c in row["chunks"])
        assert all(not len(c) for c in worker.collectors)
        assert all(c.codec.capacity == 1 for c in worker.collectors)
        assert not any(worker.solver.archive.entries_by_player.values())
        assert worker.solver.strategy_buffer.add_calls == 0
        # Independently reproduce the worker's trajectory with the original
        # sequential collector, large enough to retain every sample.
        reference_cls = StructuredSingleDeepCFRSolver if cls is CLASSES[1] else OptimisedSingleDeepCFRSolver
        reference = reference_cls(**{**SMALL, "memory_capacity": 10000})
        for network, state in zip(reference._advantage_networks, weights(learner)):
            network.load_state_dict(state)
        reference._iteration = 1
        with isolated_rng(phase_seed(0, 1, 0, 1)):
            reference._collect_traversals_for_player(0)
        codec = learner.advantage_buffers[0]
        dense = {key: np.concatenate([chunk[key] for chunk in row["chunks"]])
                 for key in ("info_states", "iterations", "targets")}
        dense["info_states"] = codec._unpack(dense["info_states"])
        assert compare_trees(dense, reference.advantage_buffers[0].as_batch())["exact"]
        assert row["nodes_touched"] == reference._nodes_touched
        outcomes = []
        for packed in (False, True):
            random.seed(143)
            buffer = type(codec)(17, info_state_size=codec.feature_count, num_actions=3)
            if packed:
                for chunk in row["chunks"]:
                    buffer.add_packed_batch(chunk, feature_encoding=row["feature_encoding"])
            else:
                for sample in reference.advantage_buffers[0]:
                    buffer.add(sample)
            outcomes.append((buffer.as_batch(), buffer.add_calls,
                             buffer.sample_batch(5), random.getstate()))
        assert compare_trees(*outcomes)["exact"]
        repeated = worker.collect(7, 0, weights(learner), 1)
        row.pop("collection_seconds")
        repeated.pop("collection_seconds")
        assert compare_trees(row, repeated)["exact"]


@pytest.mark.parametrize("cls", CLASSES)
def test_alternating_weights_archive_and_complete_iteration_budget(cls, monkeypatch):
    set_seed(71)
    with cls(parallel_backend="serial", **SMALL) as solver:
        original_collect = SDCFRTraversalWorker.collect
        observed = []
        def inspect(worker, n, player, sent, iteration):
            expected = [np.concatenate([value.numpy().reshape(-1) for value in state.values()])
                        for state in weights(solver)]
            assert compare_trees(sent["arrays"], expected)["exact"]
            if player == 1:
                archived = solver.archive.entries_by_player[0][-1]
                assert archived.iteration == iteration
                np.testing.assert_array_equal(sent["arrays"][0], np.concatenate(
                    [value.numpy().reshape(-1) for value in archived.state_dict.values()]))
            observed.append((iteration, player, worker.worker_index, n))
            return original_collect(worker, n, player, sent, iteration)
        monkeypatch.setattr(SDCFRTraversalWorker, "collect", inspect)
        def forbidden(*_args):
            raise AssertionError("No average-policy learning in SD-CFR")
        monkeypatch.setattr(solver, "_learn_strategy_network", forbidden)
        optimizer_ids = [id(o) for o in solver._optimizer_advantages]
        original_policy = copy.deepcopy(solver._policy_network.state_dict())
        initial = weights(solver)
        result = solver.solve(max_training_seconds=1e-12)
        assert observed == [(1, p, w, n) for p in (0, 1) for w, n in enumerate((3, 2, 2))]
        assert [len(es) for es in solver.archive.entries_by_player.values()] == [1, 1]
        assert solver.strategy_buffer.add_calls == 0 and result.policy_network is None
        assert [id(o) for o in solver._optimizer_advantages] == optimizer_ids
        assert all(o.state for o in solver._optimizer_advantages)
        assert compare_trees(original_policy, solver._policy_network.state_dict())["exact"]
        assert not compare_trees(initial, weights(solver))["exact"]
        assert solver.last_parallel_collection["traversals"] == 7
        assert solver.archive.metadata["parallel_execution"]["worker_reservoir_sampling"] is False
    with pytest.raises(RuntimeError, match="closed"):
        solver.solve()


def test_worker_initialisation_preserves_central_rng_and_skips_zero_assignments():
    set_seed(1)
    with CLASSES[0](parallel_backend="serial", parallel_num_workers=4,
                    **{**SMALL, "num_traversals": 2}) as solver:
        before = rng_state()
        solver._start_workers()
        assert compare_trees(before, rng_state())["exact"]
        solver._collect_traversals_for_player(0)
        assert solver.last_parallel_collection["worker_traversals"] == [1, 1, 0, 0]
        # Traversal RNG does not leak into the learner; reservoir replacement
        # may legitimately advance the central Python RNG.
        assert compare_trees(before[1:], rng_state()[1:])["exact"]


@pytest.mark.parametrize("bad", ["stale", "missing_rows", "nan", "encoding"])
def test_invalid_worker_results_fail_before_replay_mutation(bad, monkeypatch):
    set_seed(71)
    with CLASSES[0](parallel_backend="serial", **SMALL) as solver:
        original = SDCFRTraversalWorker.collect
        def corrupt(worker, *args):
            row = original(worker, *args)
            if worker.worker_index == 1:
                if bad == "stale":
                    row["iteration"] += 1
                if bad == "missing_rows":
                    row["rows"] += 1
                if bad == "encoding":
                    row["feature_encoding"] = "wrong"
                if bad == "nan":
                    row["chunks"][0]["targets"][0, 0] = np.nan
            return row
        monkeypatch.setattr(SDCFRTraversalWorker, "collect", corrupt)
        with pytest.raises(RuntimeError):
            solver.solve()
        assert all(b.add_calls == 0 for b in solver.advantage_buffers)
        assert solver._closed and not solver._workers


def test_row_limit_fails_without_local_sample_thinning():
    set_seed(71)
    with CLASSES[0](parallel_backend="serial", parallel_max_rows_per_worker=1, **SMALL) as solver:
        with pytest.raises(RuntimeError, match="refusing to discard"):
            solver.solve()
        assert all(b.add_calls == 0 for b in solver.advantage_buffers)
        assert solver._closed


@pytest.mark.parametrize("cls", CLASSES)
def test_disk_archive_and_existing_episode_policy_compatibility(cls, tmp_path):
    set_seed(41)
    with cls(parallel_backend="serial", **SMALL) as solver:
        solver.archive = DiskSDCFRArchive(solver, tmp_path, chunk_iterations=2)
        solver.solve()
        path = solver.archive.checkpoint(tmp_path / "policy.json")
        reader = DiskArchiveReader(path, solver._game)
        assert reader.count == 3
        assert reader.contract["metadata"]["parallel_execution"]["workers"] == 3
        policy = DiskSampledPolicy(reader)
        policy.begin_episode(seed=17)
        state = solver._game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(state.chance_outcomes()[0][0])
            else:
                probabilities = policy.action_probabilities(state)
                assert sum(probabilities.values()) == pytest.approx(1)
                assert set(probabilities) == set(state.legal_actions())
                state.apply_action(max(probabilities, key=probabilities.get))


@pytest.mark.parametrize("kwargs", [dict(parallel_num_workers=0), dict(parallel_num_workers=1.5),
    dict(parallel_run_seed=-1), dict(parallel_backend="unknown"), dict(pack_replay=False),
    dict(parallel_timeout_seconds=float("nan")), dict(parallel_chunk_rows=0)])
def test_invalid_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        CLASSES[0](**SMALL, **kwargs)


def training_signature(solver, result):
    return dict(weights=weights(solver), replay=[b.state_dict() for b in solver.advantage_buffers],
                optimizers=[o.state_dict() for o in solver._optimizer_advantages],
                archives=[[dict(entry.state_dict) for entry in es]
                          for es in solver.archive.entries_by_player.values()],
                losses=result.advantage_losses, nodes=result.nodes_touched, rng=rng_state())


@pytest.mark.skipif(os.environ.get("RUN_RAY_SD_CFR_TESTS") != "1",
                    reason="Opt-in real multi-process Ray test (local sockets/processes required)")
@pytest.mark.parametrize("cls", CLASSES)
def test_real_ray_matches_same_stream_serial_workers(cls):
    import ray
    assert not ray.is_initialized(), "Run integration test in an isolated process"
    signatures = []
    for backend in ("serial", "ray"):
        set_seed(45)
        with cls(parallel_backend=backend, parallel_num_workers=2,
                 parallel_timeout_seconds=90,
                 **{**SMALL, "advantage_network_layers": (32,) * 8}) as solver:
            result = solver.solve()
            if backend == "ray":
                assert all(x["torch_threads"] == 1 for x in ray.get([w.ping.remote() for w in solver._workers]))
            signatures.append(training_signature(solver, result))
        assert not ray.is_initialized()
    comparison = compare_trees(*signatures)
    assert comparison["exact"], comparison


@pytest.mark.skipif(os.environ.get("RUN_RAY_SD_CFR_TESTS") != "1",
                    reason="Opt-in real Ray actor-failure and runtime-ownership test")
def test_real_ray_failure_does_not_shutdown_external_runtime():
    import ray
    assert not ray.is_initialized()
    ray.init(num_cpus=2, object_store_memory=128 * 1024 * 1024,
             include_dashboard=False, log_to_driver=False)
    try:
        set_seed(71)
        with CLASSES[0](parallel_num_workers=2, parallel_max_rows_per_worker=1,
                        parallel_timeout_seconds=90, **SMALL) as solver:
            with pytest.raises(ray.exceptions.RayTaskError, match="refusing to discard"):
                solver.solve()
            assert solver._closed and not solver._workers
            assert all(b.add_calls == 0 for b in solver.advantage_buffers)
        assert ray.is_initialized()
        # Verify the caller-owned runtime is still usable, not just flagged up.
        assert ray.get(ray.remote(lambda: 42).remote()) == 42
    finally:
        ray.shutdown()
