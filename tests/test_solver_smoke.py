from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deep_cfr_poker.solver import DeepCFRSolver, SolveResult


def test_tiny_fhp_training_does_not_enumerate_full_game_tree(fhp_game):
    solver = DeepCFRSolver(
        fhp_game,
        policy_network_layers=(8, 8),
        advantage_network_layers=(8, 8),
        num_iterations=1,
        num_traversals=2,
        learning_rate=1e-3,
        batch_size_advantage=2,
        batch_size_strategy=2,
        memory_capacity=256,
        policy_network_train_steps=1,
        policy_network_train_every=1,
        advantage_network_train_steps=1,
        reinitialize_advantage_networks=False,
        evaluation_interval=1,
        compute_exploitability=False,
    )

    result = solver.solve()

    assert isinstance(result, SolveResult)
    assert result.nodes_touched[-1] > 0
    assert np.isnan(result.nash_conv[-1])
    assert np.isnan(result.average_policy_value[-1])
    assert result.diagnostics["policy_training_events"] == [1]
