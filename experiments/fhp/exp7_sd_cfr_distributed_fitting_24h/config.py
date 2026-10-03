"""Preserve Experiment 5's game, learner, VM, seeds, and 24-hour protocol."""
from experiments.fhp.exp5_sd_cfr_parallel_24h.config import (
    FEATURE_ENCODER_METADATA, SEEDS, HOURS, SECONDS, ITERATION_CAP, REFERENCE_VM,
    TRAVERSAL_WORKERS, solver_config, execution_config as baseline_execution,
)

EXPERIMENT_NAME = "exp7_sd_cfr_distributed_fitting_24h"
ALGORITHM_ID = "distributed_fitting_structured_uniform_sd_cfr"
REPORT_ID = "sd_cfr_exp7_distributed_fitting8"
EXPERIMENT_METADATA = dict(
    purpose="24h quality/efficiency comparison with Experiment 5, not output-equivalence certification",
    fitting_validation="strict_single_update_correctness; accumulated_full_fit_drift_is_reported",
    comparison_baseline="exp5_sd_cfr_parallel_24h",
    primary_quality_endpoint="24h same-seed two-seat head-to-head value",
    policy_checkpoints_retained=True,
)


def execution_config(seed):
    return dict(baseline_execution(seed), distributed_fitting=True)


def task_name(seed):
    if seed not in SEEDS:
        raise ValueError("Unexpected seed")
    return f"task_{seed:03d}_{ALGORITHM_ID}_seed_{seed}"
