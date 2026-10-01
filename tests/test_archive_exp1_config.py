from copy import deepcopy

import pytest

from experiments.fhp.archive_exp1_deep_cfr_best_config_transfer.config import (
    CHECKPOINT_SCHEDULE,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    validate_config,
)


def test_archived_exp1_approved_configuration_is_pinned():
    assert DEFAULT_SEEDS == (1234, 2025, 31415, 27182, 16180)
    assert DEFAULT_CONFIG == {
        "experiment_id": 1,
        "experiment_name": "archive_exp1_fhp_deep_cfr_best_config_transfer",
        "game_name": "FHP",
        "num_iterations": 1050,
        "num_traversals": 320,
        "evaluation_interval": 25,
        "checkpoint_schedule": CHECKPOINT_SCHEDULE,
        "policy_network_layers": (32, 32),
        "advantage_network_layers": (32, 32, 32, 32, 32, 32, 32, 32),
        "policy_network_type": "mlp",
        "advantage_network_type": "residual_layer_norm_centered_advantage_mlp",
        "learning_rate": 0.004,
        "learning_rate_schedule": "constant",
        "batch_size_advantage": 2048,
        "batch_size_strategy": 1024,
        "memory_capacity": 5_000_000,
        "replay_buffer_type": "compact",
        "reinitialize_advantage_networks": False,
        "policy_network_train_steps": 200,
        "advantage_network_train_steps": 200,
        "policy_network_train_every": 10,
        "policy_training_mode": "intermittent",
        "final_policy_network_train_steps": None,
        "target_processing": "standardize",
        "target_clip_value": 1.0,
        "target_standardize_epsilon": 1e-6,
        "advantage_replay_sampling": "uniform",
        "average_strategy_weighting": "uniform",
        "priority_alpha": 1.0,
        "priority_epsilon": 1e-6,
        "execution_backend": "sequential",
        "compute_exploitability": False,
        "exploitability_threshold": 0.05,
        "save_final_checkpoint": True,
        "final_checkpoint_include_buffers": True,
    }
    validate_config(DEFAULT_CONFIG)


def test_archived_exp1_rejects_full_tree_evaluation_and_stale_snapshots():
    unsafe = deepcopy(DEFAULT_CONFIG)
    unsafe["compute_exploitability"] = True
    with pytest.raises(ValueError, match="exploitability"):
        validate_config(unsafe)

    stale = deepcopy(DEFAULT_CONFIG)
    stale["checkpoint_schedule"] = (101, 1050)
    with pytest.raises(ValueError, match="average-policy training"):
        validate_config(stale)
