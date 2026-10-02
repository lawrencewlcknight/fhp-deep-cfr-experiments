"""Same scientific contract as Experiment 2; only player inputs change."""
from deep_cfr_poker.fhp_features import FHPFeatureEncoder
from experiments.fhp.exp2_sd_cfr_24h.config import (
    SEEDS, HOURS, SECONDS, ITERATION_CAP, REFERENCE_VM, solver_config,
)

EXPERIMENT_NAME = "exp3_sd_cfr_structured_24h"
ALGORITHM_ID = "structured_uniform_sd_cfr"
REPORT_ID = "sd_cfr_exp3_structured"
FEATURE_ENCODER_METADATA = FHPFeatureEncoder().metadata()


def task_name(seed):
    if seed not in SEEDS:
        raise ValueError(f"Expected one of {SEEDS}")
    return f"task_{seed:03d}_{ALGORITHM_ID}_seed_{seed}"
