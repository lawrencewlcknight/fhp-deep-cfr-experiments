"""Preserve Experiment 5's game, learner, VM, seeds, and 24-hour protocol."""
from experiments.fhp.exp5_sd_cfr_parallel_24h.config import (
    FEATURE_ENCODER_METADATA, SEEDS, HOURS, SECONDS, ITERATION_CAP, REFERENCE_VM,
    TRAVERSAL_WORKERS, solver_config, execution_config as baseline_execution,
)

EXPERIMENT_NAME = "exp7_sd_cfr_distributed_fitting_24h"
ALGORITHM_ID = "distributed_fitting_structured_uniform_sd_cfr"
REPORT_ID = "sd_cfr_exp7_distributed_fitting8"


def execution_config(seed):
    return dict(baseline_execution(seed), distributed_fitting=True)


def task_name(seed):
    if seed not in SEEDS:
        raise ValueError("Unexpected seed")
    return f"task_{seed:03d}_{ALGORITHM_ID}_seed_{seed}"
