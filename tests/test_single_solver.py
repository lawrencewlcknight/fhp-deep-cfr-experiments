"""FHP SD-CFR safety, trajectory, metadata and selected-configuration tests."""

import random

import numpy as np
import pytest
import torch

from deep_cfr_poker.game import load_fhp_game
from deep_cfr_poker.sd_cfr import (
    HistoricalSDCFRPolicy, SDCFRArchive, SampledSDCFRPolicy,
    build_information_set_catalog, exact_average_policy,
)
from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.single_solver import SELECTED_SD_CFR_KWARGS, SingleDeepCFRSolver
from deep_cfr_poker.solver import DeepCFRSolver


SMALL = dict(num_iterations=2, num_traversals=2, advantage_network_layers=(8, 8),
             policy_network_layers=(8, 8), memory_capacity=64,
             batch_size_advantage=2, advantage_network_train_steps=1,
             evaluation_interval=1)


def test_selected_learner_contract():
    cfg = dict(SELECTED_SD_CFR_KWARGS)
    assert cfg["advantage_network_layers"] == (32,) * 8
    assert cfg["advantage_network_type"] == "residual_layer_norm_centered_advantage_mlp"
    assert cfg["num_traversals"] == 320
    assert cfg["advantage_network_train_steps"] == 200
    assert cfg["learning_rate"] == 0.004
    assert cfg["learning_rate_schedule"] == "constant"
    assert cfg["memory_capacity"] == 5_000_000
    assert cfg["batch_size_advantage"] == 2048
    assert cfg["target_processing"] == "standardize"
    assert cfg["advantage_replay_sampling"] == "uniform"
    assert cfg["reinitialize_advantage_networks"] is False
    assert cfg["collect_strategy_replay"] is False
    assert cfg["policy_training_mode"] == "disabled"
    assert cfg["replay_buffer_type"] == "compact"


@pytest.mark.parametrize("backend", ["python", "compact"])
def test_fhp_train_archive_reload_and_play_without_tree_enumeration(
    fhp_game, tmp_path, monkeypatch, backend
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Average-policy fitting/tree enumeration is forbidden")
    monkeypatch.setattr(DeepCFRSolver, "_learn_strategy_network", forbidden)
    monkeypatch.setattr(DeepCFRSolver, "_policy_network_diagnostics", forbidden)
    monkeypatch.setattr("deep_cfr_poker.solver.policy.tabular_policy_from_callable", forbidden)
    solver = SingleDeepCFRSolver(fhp_game, replay_buffer_type=backend, **SMALL)
    optimizers = list(solver._optimizer_advantages)
    originals = [{k: v.clone() for k, v in net.state_dict().items()}
                 for net in solver._advantage_networks]
    result = solver.solve()
    assert solver.weighting == "uniform"
    assert result.policy_network is None
    assert result.diagnostics["policy_training_events"] == [0, 0]
    assert not solver._optimizer_policy.state
    assert solver.strategy_buffer.add_calls == 0
    assert solver.strategy_buffer.capacity == 1
    assert all(len(buffer) > 0 for buffer in solver.advantage_buffers)
    assert solver._optimizer_advantages == optimizers  # continuous Adam state
    assert all(opt.state for opt in optimizers)
    assert all(any(not torch.equal(initial[k], net.state_dict()[k]) for k in initial)
               for initial, net in zip(originals, solver._advantage_networks))
    for player in (0, 1):
        assert [x.iteration for x in solver.archive.entries_by_player[player]] == [1, 2]
    archive = SDCFRArchive.load(solver.save_archive(tmp_path / "policy_archive.pt"))
    assert archive.metadata["primary_weighting"] == "uniform"
    assert archive.game_name == "FHP"
    payload = archive.to_payload()
    assert not {"policy_state_dict", "buffers", "optimizer_state_dict"} & payload.keys()
    policy = SampledSDCFRPolicy(fhp_game, archive, seed=11, weighting="uniform")
    fresh = solver.make_policy(seed=11)
    rng = np.random.default_rng(4)
    for _ in range(6):
        policy.resample_episode()
        fresh.resample_episode()
        selected = dict(policy.selected_iterations)
        state = fhp_game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probabilities = zip(*state.chance_outcomes())
            else:
                actual = policy.action_probabilities(state)
                assert actual == fresh.action_probabilities(state)
                assert actual == policy.action_probabilities(state)
                assert set(actual) == set(state.legal_actions())
                assert all(v >= 0 and np.isfinite(v) for v in actual.values())
                assert sum(actual.values()) == pytest.approx(1)
                actions, probabilities = zip(*actual.items())
            state.apply_action(int(rng.choice(actions, p=probabilities)))
            assert policy.selected_iterations == selected
        assert np.isfinite(state.returns()).all()
    with pytest.raises(ValueError, match="Full-tree"):
        exact_average_policy(fhp_game, archive, weighting="uniform")


def test_fhp_tree_guard_precedes_enumeration(fhp_game):
    with pytest.raises(ValueError, match="Full-tree"):
        build_information_set_catalog(fhp_game)


def test_capture_phase_frozen_copies_and_rng_preservation(fhp_game):
    solver = SingleDeepCFRSolver(fhp_game, **SMALL)
    phases = []
    def inspect(active, player, iteration):
        phases.append((player, iteration))
        entry = active.archive.entries_by_player[player][-1]
        assert entry.iteration == iteration == active._iteration
        for key, value in active._advantage_networks[player].state_dict().items():
            assert torch.equal(value, entry.state_dict[key])
            assert value.data_ptr() != entry.state_dict[key].data_ptr()
    solver.solve(post_player_update_callback=inspect)
    assert phases == [(0, 1), (1, 1), (0, 2), (1, 2)]
    before = solver.archive.entries_by_player[0][0].state_dict
    saved = {k: v.clone() for k, v in before.items()}
    with torch.no_grad():
        for param in solver._advantage_networks[0].parameters():
            param.add_(10)
    assert all(torch.equal(v, before[k]) for k, v in saved.items())
    torch_rng, np_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    sampled = solver.make_policy(seed=3)
    sampled.resample_episode()
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    assert np.random.get_state()[2:] == np_rng[2:]
    assert random.getstate() == python_rng


def test_compact_and_python_sd_cfr_match(fhp_game):
    runs = []
    for backend in ("python", "compact"):
        set_seed(32)
        solver = SingleDeepCFRSolver(fhp_game, replay_buffer_type=backend, **SMALL)
        result = solver.solve()
        runs.append((solver, result))
    assert runs[0][1].advantage_losses == runs[1][1].advantage_losses
    assert runs[0][1].nodes_touched == runs[1][1].nodes_touched
    for player in (0, 1):
        for a, b in zip(runs[0][0].archive.entries_by_player[player],
                        runs[1][0].archive.entries_by_player[player]):
            assert all(torch.equal(v, b.state_dict[k]) for k, v in a.state_dict.items())


def test_time_budget_stops_at_complete_iteration_with_final_metrics(fhp_game):
    solver = SingleDeepCFRSolver(fhp_game, **{**SMALL, "evaluation_interval": 100})
    calls = []
    result = solver.solve(max_training_seconds=1e-12,
                          post_iteration_callback=lambda s, i: calls.append(i))
    assert calls == [1]
    assert solver._iteration == 2
    assert result.nodes_touched == [solver._nodes_touched]
    assert all(len(entries) == 1 for entries in solver.archive.entries_by_player.values())


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_time_budget_is_rejected_before_training(fhp_game, budget):
    solver = SingleDeepCFRSolver(fhp_game, **SMALL)
    with pytest.raises(ValueError, match="positive and finite"):
        solver.solve(max_training_seconds=budget)
    assert solver._nodes_touched == 0


def test_wrong_game_and_dummy_policy_exports_are_rejected(fhp_game, tmp_path):
    solver = SingleDeepCFRSolver(fhp_game, **SMALL)
    solver.solve()
    with pytest.raises(RuntimeError, match="make_policy"):
        solver.action_probabilities(fhp_game.new_initial_state())
    with pytest.raises(RuntimeError, match="save_archive"):
        solver.save_policy_snapshot(tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="omits"):
        solver.save_full_model(tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="historical"):
        solver.load_full_model({})
    from deep_cfr_poker.snapshots import save_policy_snapshot
    with pytest.raises(ValueError, match="advantage archive"):
        save_policy_snapshot(solver, tmp_path / "bad.pt", seed=0, target_iteration=2)
    solver.archive.metadata["game_string"] = "wrong-game"
    with pytest.raises(ValueError, match="game contract"):
        solver.make_policy()
    assert not (tmp_path / "bad.pt").exists()


def test_conventional_evaluation_loader_rejects_archive(fhp_game, tmp_path):
    from deep_cfr_poker.evaluation_adapter import load_policy_for_evaluation
    solver = SingleDeepCFRSolver(fhp_game, **SMALL)
    solver.solve()
    path = solver.save_archive(tmp_path / "archive.pt")
    with pytest.raises(ValueError, match="average-policy"):
        load_policy_for_evaluation(path)


@pytest.mark.parametrize("overrides", [{"policy_training_mode": "intermittent"},
                                      {"collect_strategy_replay": True},
                                      {"compute_exploitability": True}])
def test_standalone_safety_constraints(overrides):
    with pytest.raises(ValueError, match="Standalone SD-CFR requires"):
        SingleDeepCFRSolver(**SMALL, **overrides)


def test_weighting_and_checkpoint_control_episode_sampling(fhp_game):
    solver = SingleDeepCFRSolver(fhp_game, **SMALL)
    solver.solve()
    restricted = solver.make_policy(seed=1, checkpoint=1)
    for _ in range(5):
        restricted.resample_episode()
        assert restricted.selected_iterations == {0: 1, 1: 1}
    for weighting, expected in (("uniform", .5), ("linear", 2/3)):
        sampled = SampledSDCFRPolicy(fhp_game, solver.archive, seed=20, weighting=weighting)
        counts = []
        for _ in range(300):
            sampled.resample_episode()
            counts.append(sampled.selected_iterations[0] == 2)
        assert np.mean(counts) == pytest.approx(expected, abs=.08)
    with pytest.raises(ValueError, match="No player"):
        solver.make_policy(checkpoint=0)


def test_opponent_private_cards_do_not_affect_policy(fhp_game):
    solver = SingleDeepCFRSolver(fhp_game, **SMALL)
    solver.solve()
    sampled = solver.make_policy(seed=8)
    first, second = fhp_game.new_initial_state(), fhp_game.new_initial_state()
    for card in (0, 4, 8, 12):
        first.apply_action(card)
    for card in (0, 4, 16, 20):
        second.apply_action(card)
    assert first.current_player() == second.current_player() == 0
    assert first.information_state_string(0) == second.information_state_string(0)
    assert sampled.action_probabilities(first) == sampled.action_probabilities(second)
