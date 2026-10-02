"""Frozen scientific contract: selected uniform SD-CFR, not paper schedule."""
from deep_cfr_poker.single_solver import SELECTED_SD_CFR_KWARGS

EXPERIMENT_NAME = "exp2_sd_cfr_24h"
ALGORITHM_ID = "optimised_uniform_sd_cfr"
REPORT_ID = "sd_cfr_exp2"
FEATURE_ENCODER_METADATA = None
SEEDS = (0, 1, 2)
HOURS = (6, 12, 18, 24)
SECONDS = tuple(hour * 3600 for hour in HOURS)
ITERATION_CAP = 1_000_000
BASE_SEED = 20260922
RULE_DEALS = 10000
LBR_DEALS = 1000
LBR_SHARD_DEALS = 10
LBR_ROLLOUTS = 4096
CROSSPLAY_DEALS = 50000
REFERENCE_VM = dict(machine_type="n2-standard-8", vcpus=8, memory_gib=32,
                    disk_gib=200, provisioning="STANDARD")


def solver_config(smoke=False):
    config = dict(SELECTED_SD_CFR_KWARGS)
    config["num_iterations"] = ITERATION_CAP
    if smoke:
        config.update(num_traversals=4, advantage_network_train_steps=2,
                      batch_size_advantage=8, memory_capacity=64, evaluation_interval=1)
    return config


def task_name(seed):
    if seed not in SEEDS:
        raise ValueError(f"Expected one of {SEEDS}")
    return f"task_{seed:03d}_{ALGORITHM_ID}_seed_{seed}"
