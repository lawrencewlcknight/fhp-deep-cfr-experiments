"""Archived configuration for the former FHP Deep CFR Experiment 1."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from deep_cfr_poker.constants import DEFAULT_EXPLOITABILITY_THRESHOLD


EXPERIMENT_ID = 1
EXPERIMENT_NAME = "archive_exp1_fhp_deep_cfr_best_config_transfer"
DEFAULT_SEEDS = (1234, 2025, 31415, 27182, 16180)
CHECKPOINT_SCHEDULE = (100, 250, 500, 750, 1050)

DEFAULT_CONFIG = {
    "experiment_id": EXPERIMENT_ID,
    "experiment_name": EXPERIMENT_NAME,
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
    "exploitability_threshold": DEFAULT_EXPLOITABILITY_THRESHOLD,
    "save_final_checkpoint": True,
    "final_checkpoint_include_buffers": True,
}


def validate_config(config: Mapping[str, object]) -> None:
    """Validate FHP safety constraints and checkpoint freshness."""
    if str(config["game_name"]) != "FHP":
        raise ValueError("Archived Deep CFR experiment must use the canonical FHP game")
    if bool(config["compute_exploitability"]):
        raise ValueError("Exact full-tree exploitability must remain disabled for FHP")
    if str(config["execution_backend"]) != "sequential":
        raise ValueError("Archived Deep CFR experiment's execution backend is sequential")

    positive_fields = (
        "num_iterations",
        "num_traversals",
        "evaluation_interval",
        "batch_size_advantage",
        "batch_size_strategy",
        "memory_capacity",
        "policy_network_train_steps",
        "advantage_network_train_steps",
        "policy_network_train_every",
    )
    invalid = [name for name in positive_fields if int(config[name]) < 1]
    if invalid:
        raise ValueError(f"These configuration values must be positive: {invalid}")

    schedule = tuple(int(value) for value in config["checkpoint_schedule"])
    if not schedule or any(a >= b for a, b in zip(schedule, schedule[1:])):
        raise ValueError("checkpoint_schedule must be non-empty and increasing")
    if schedule[-1] != int(config["num_iterations"]):
        raise ValueError("The final policy snapshot must equal num_iterations")
    train_every = int(config["policy_network_train_every"])
    stale = [iteration for iteration in schedule[:-1] if iteration % train_every]
    if stale:
        raise ValueError(
            "Snapshots must coincide with average-policy training; "
            f"incompatible iterations: {stale}"
        )


def smoke_config() -> dict:
    """Return a tiny configuration preserving the archived experiment's code paths."""
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "experiment_name": f"{EXPERIMENT_NAME}_smoke",
            "num_iterations": 2,
            "num_traversals": 2,
            "evaluation_interval": 1,
            "checkpoint_schedule": (1, 2),
            "policy_network_layers": (8, 8),
            "advantage_network_layers": (8, 8),
            "batch_size_advantage": 2,
            "batch_size_strategy": 2,
            "memory_capacity": 256,
            "policy_network_train_steps": 1,
            "advantage_network_train_steps": 1,
            "policy_network_train_every": 1,
        }
    )
    validate_config(config)
    return config


validate_config(DEFAULT_CONFIG)

__all__ = [
    "CHECKPOINT_SCHEDULE",
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "smoke_config",
    "validate_config",
]
