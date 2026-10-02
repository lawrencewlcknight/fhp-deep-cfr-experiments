"""Actual replay/Adam/RNG/archive round trips, not merely reloadable weights."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive, DiskArchiveReader, DiskSampledPolicy, sha256
from deep_cfr_poker.sd_cfr_parallel import ParallelStructuredSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_training_state import save_training_state, load_training_state, inspect_training_state
from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.solver import DeepCFRSolver
from experiments.fhp.exp1_sd_cfr_efficiency.run import compare_trees

LEARNER = dict(num_iterations=2, num_traversals=8, advantage_network_layers=(8, 8),
               memory_capacity=17, batch_size_advantage=2,
               advantage_network_train_steps=2, evaluation_interval=2)
EXECUTION = dict(parallel_backend="serial", parallel_num_workers=8, parallel_run_seed=71,
                 parallel_chunk_rows=3, parallel_inference_cache_entries=32)


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def checkpoint(solver, root, config, execution):
    path = solver.archive.checkpoint(root / "archive/time_48h.json")
    context = dict(seed=71, smoke=True, hours=[48], active_seconds=1.0,
                   records=[dict(path="archive/time_48h.json", sha256=sha256(path))])
    return save_training_state(solver, root, path, config=config, execution=execution, context=context)


def signature(solver, result):
    payload = DeepCFRSolver.extract_full_model(solver, include_buffers=False)
    replay = [dict(rows=b.as_batch(), add_calls=b.add_calls) for b in solver.advantage_buffers]
    return copy.deepcopy(dict(payload=payload, replay=replay, losses=result.advantage_losses,
                              iteration_diagnostics=result.diagnostics["iteration"]))


@pytest.mark.parametrize("capacity", [17, 4096])
def test_continuation_is_bit_exact_including_reservoir_replacement_and_partial_capacity(tmp_path, capacity, backend="serial"):
    # Cover both full Algorithm-R reservoirs and arrays smaller than capacity.
    learner = dict(LEARNER, memory_capacity=capacity)
    execution = dict(EXECUTION, parallel_backend=backend)
    set_seed(71)
    original_root = tmp_path / "original"
    with ParallelStructuredSingleDeepCFRSolver(**learner, **execution) as solver:
        solver.archive = DiskSDCFRArchive(solver, original_root / "archive", chunk_iterations=2)
        solver.solve()
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        state = checkpoint(solver, original_root, learner, execution)
        assert compare_trees(before, (random.getstate(), np.random.get_state(), torch.get_rng_state()))["exact"]
        assert solver.advantage_buffers[0].add_calls > 17
        if capacity == 4096:
            assert len(solver.advantage_buffers[0]) < capacity
        result = solver.solve()
        expected = signature(solver, result)
        expected_path = solver.archive.checkpoint(original_root / "archive/reference_72h.json")
        reader = DiskArchiveReader(expected_path, solver._game)
        history = [reader.weights(p, i) for i in range(1, 5) for p in (0, 1)]
    source_hashes = {str(p): sha256(p) for p in original_root.rglob("*") if p.is_file()}
    set_seed(999)
    restored, context = load_training_state(state, tmp_path / "continued/archive",
                                           expected_config=learner, expected_execution=execution)
    with restored:
        assert restored._iteration == 3
        assert restored.archive.count == 2
        assert not restored._workers
        assert context["seed"] == 71
        result = restored.solve()
        comparison = compare_trees(expected, signature(restored, result))
        assert comparison["exact"], comparison
        assert result.diagnostics["iteration"] == [4]
        new_path = restored.archive.checkpoint(tmp_path / "continued/archive/time_72h.json")
        reader = DiskArchiveReader(new_path, restored._game)
        assert compare_trees(history, [reader.weights(p, i) for i in range(1, 5) for p in (0, 1)])["exact"]
        assert DiskArchiveReader(tmp_path / "continued/archive/time_48h.json", restored._game).count == 2
        policy = DiskSampledPolicy(reader)
        policy.begin_episode(seed=14)
        state0 = restored._game.new_initial_state()
        while state0.is_chance_node():
            state0.apply_action(state0.chance_outcomes()[0][0])
        assert sum(policy.action_probabilities(state0).values()) == pytest.approx(1)
    assert source_hashes == {str(p): sha256(p) for p in original_root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["replay", "archive", "config", "execution", "runtime", "code", "partial"])
def test_restore_fails_closed_on_missing_or_incompatible_state(tmp_path, damage):
    set_seed(71)
    root = tmp_path / "source"
    with ParallelStructuredSingleDeepCFRSolver(**LEARNER, **EXECUTION) as solver:
        solver.archive = DiskSDCFRArchive(solver, root / "archive", chunk_iterations=2)
        solver.solve()
        state = checkpoint(solver, root, LEARNER, EXECUTION)
    meta = json.loads(state.read_text())
    if damage in ("replay", "archive"):
        file = (state.parent / meta["buffers"][0]["arrays"]["info_states"]["path"] if damage == "replay"
                else next((root / "archive").glob("chunk_*.npy")))
        file.write_bytes(b"incomplete")
    if damage in ("runtime", "code", "partial"):
        key = dict(runtime="runtime", code="implementation_sha256", partial="status")[damage]
        meta[key] = "wrong"
        state.write_text(json.dumps(meta))
    expected_config = dict(LEARNER, memory_capacity=31) if damage == "config" else LEARNER
    expected_execution = dict(EXECUTION, parallel_num_workers=4) if damage == "execution" else EXECUTION
    with pytest.raises(ValueError):
        load_training_state(state, tmp_path / "new/archive", expected_config=expected_config,
                            expected_execution=expected_execution)


def test_refuse_in_place_continuation(tmp_path):
    set_seed(71)
    with ParallelStructuredSingleDeepCFRSolver(**LEARNER, **EXECUTION) as solver:
        solver.archive = DiskSDCFRArchive(solver, tmp_path / "archive", chunk_iterations=2)
        solver.solve()
        state = checkpoint(solver, tmp_path, LEARNER, EXECUTION)
    with pytest.raises(ValueError, match="separate"):
        load_training_state(state, tmp_path / "archive")
