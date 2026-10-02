"""Experiment 5 unchanged, except horizon and final training-state retention."""
from experiments.fhp.exp5_sd_cfr_parallel_24h.config import (
    FEATURE_ENCODER_METADATA, SEEDS, ITERATION_CAP, REFERENCE_VM,
    TRAVERSAL_WORKERS, solver_config, execution_config,
)

EXPERIMENT_NAME = "exp6_sd_cfr_parallel_48h"
ALGORITHM_ID = "parallel_structured_uniform_sd_cfr_48h"
REPORT_ID = "sd_cfr_exp6_parallel8_48h"
HOURS = tuple(range(6, 49, 6))
SECONDS = tuple(hour * 3600 for hour in HOURS)
RETAIN_FINAL_TRAINING_STATE = True


def task_name(seed):
    if seed not in SEEDS:
        raise ValueError("Unexpected seed")
    return f"task_{seed:03d}_{ALGORITHM_ID}_seed_{seed}"


def checkpoint_hours(manifest):
    """Validate both a fresh 48h run and explicit six-hour-step continuations."""
    hours = manifest["checkpoints_hours"]
    continuation = manifest.get("continuation")
    if continuation is None:
        if hours != list(HOURS):
            raise ValueError("Incorrect 48-hour schedule")
    else:
        previous = continuation["source_hours"]
        extra = continuation["additional_hours"]
        if (not previous or previous != list(range(6, previous[-1] + 1, 6))
                or previous[:len(HOURS)] != list(HOURS) or extra not in range(6, 49, 6)
                or hours != list(range(6, previous[-1] + extra + 1, 6))):
            raise ValueError("Invalid continuation schedule")
    return tuple(hours)
