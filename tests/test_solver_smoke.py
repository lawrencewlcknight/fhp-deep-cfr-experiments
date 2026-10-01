from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.solver import DeepCFRSolver, SolveResult


@pytest.mark.parametrize("replay_buffer_type", ["python", "compact"])
def test_tiny_fhp_training_does_not_enumerate_full_game_tree(
    fhp_game, replay_buffer_type
):
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
        replay_buffer_type=replay_buffer_type,
    )

    result = solver.solve()

    assert isinstance(result, SolveResult)
    assert result.nodes_touched[-1] > 0
    assert np.isnan(result.nash_conv[-1])
    assert np.isnan(result.average_policy_value[-1])
    assert result.diagnostics["policy_training_events"] == [1]


def test_collection_reuses_info_state_and_scopes_inference_mode(fhp_game):
    solver = DeepCFRSolver(
        fhp_game,
        policy_network_layers=(8, 8),
        advantage_network_layers=(8, 8),
        num_iterations=1,
        num_traversals=1,
        memory_capacity=32,
        replay_buffer_type="compact",
    )

    class TerminalState:
        def is_terminal(self):
            return True

        def returns(self):
            return [1.0, -1.0]

    class DecisionState:
        def __init__(self, embedding_size):
            self.embedding_size = embedding_size
            self.info_state_calls = 0

        def is_terminal(self):
            return False

        def is_chance_node(self):
            return False

        def current_player(self):
            return 0

        def information_state_tensor(self, player):
            assert player == 0
            self.info_state_calls += 1
            return np.zeros(self.embedding_size, dtype=np.float32)

        def legal_actions(self, player=None):
            return [0, 1]

        def child(self, action):
            assert action in (0, 1)
            return TerminalState()

    state = DecisionState(solver._embedding_size)
    solver._root_node = state
    solver._advantage_networks[0].train()
    observed_modes = []

    def record_mode(module, _inputs):
        observed_modes.append(
            (module.training, torch.is_inference_mode_enabled())
        )

    handle = solver._advantage_networks[0].register_forward_pre_hook(record_mode)
    try:
        solver._collect_traversals_for_player(0)
    finally:
        handle.remove()

    assert state.info_state_calls == 1
    assert observed_modes == [(False, True)]
    assert solver._advantage_networks[0].training is True


def test_compact_replay_matches_python_training_for_the_same_seed(fhp_game):
    kwargs = {
        "policy_network_layers": (8, 8),
        "advantage_network_layers": (8, 8),
        "num_iterations": 2,
        "num_traversals": 2,
        "learning_rate": 1e-3,
        "batch_size_advantage": 2,
        "batch_size_strategy": 2,
        "memory_capacity": 256,
        "policy_network_train_steps": 1,
        "policy_network_train_every": 1,
        "advantage_network_train_steps": 1,
        "reinitialize_advantage_networks": False,
        "evaluation_interval": 1,
        "compute_exploitability": False,
    }
    runs = []
    for replay_buffer_type in ("python", "compact"):
        set_seed(888)
        solver = DeepCFRSolver(
            fhp_game,
            replay_buffer_type=replay_buffer_type,
            **kwargs,
        )
        runs.append((solver, solver.solve()))

    python_solver, python_result = runs[0]
    compact_solver, compact_result = runs[1]
    assert python_result.advantage_losses == compact_result.advantage_losses
    assert python_result.policy_losses == compact_result.policy_losses
    assert python_result.nodes_touched == compact_result.nodes_touched

    network_groups = [
        ([python_solver._policy_network], [compact_solver._policy_network]),
        (
            python_solver._advantage_networks,
            compact_solver._advantage_networks,
        ),
    ]
    for python_networks, compact_networks in network_groups:
        for python_network, compact_network in zip(
            python_networks, compact_networks
        ):
            for name, python_tensor in python_network.state_dict().items():
                torch.testing.assert_close(
                    python_tensor,
                    compact_network.state_dict()[name],
                    rtol=0,
                    atol=0,
                )
