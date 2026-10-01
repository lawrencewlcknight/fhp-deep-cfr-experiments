from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from deep_cfr_poker.parallel_solver import DeepCFRTraversalWorker


def test_traversal_worker_returns_only_typed_array_batches():
    worker = DeepCFRTraversalWorker(
        "FHP",
        {
            "policy_network_layers": (8, 8),
            "advantage_network_layers": (8, 8),
            "num_iterations": 1,
            "num_traversals": 1,
            "memory_capacity": 256,
            "replay_buffer_type": "python",
            "compute_exploitability": False,
        },
        worker_seed_value=1234,
    )
    state_dicts = [
        network.state_dict() for network in worker._solver._advantage_networks
    ]

    result = worker.collect(1, 0, state_dicts, iteration=1)

    assert result["nodes_touched"] > 0
    assert "advantage_memories" not in result
    assert "strategy_memories" not in result
    batches = [*result["advantage_batches"], result["strategy_batch"]]
    for batch in batches:
        assert set(batch) == {"info_states", "iterations", "targets"}
        assert batch["info_states"].dtype == np.float32
        assert batch["iterations"].dtype == np.int32
        assert batch["targets"].dtype == np.float32
        assert batch["info_states"].dtype != object
        assert batch["targets"].dtype != object

    assert result["advantage_batches"][0]["info_states"].shape[1] == 190
    assert result["strategy_batch"]["targets"].shape[1] == 3
    assert all(
        not network.training
        for network in worker._solver._advantage_networks
    )
