"""Hardware-only counterpart to Experiment 3; no new learning settings."""
from experiments.fhp.exp3_sd_cfr_structured_24h.config import (
    ALGORITHM_ID, FEATURE_ENCODER_METADATA, SEEDS, HOURS, SECONDS,
    ITERATION_CAP, solver_config, task_name, REFERENCE_VM as BASELINE_VM,
)

EXPERIMENT_NAME = "exp4_sd_cfr_structured_n2_standard16"
REPORT_ID = "sd_cfr_exp4_structured_vm16"
REFERENCE_VM = dict(BASELINE_VM, machine_type="n2-standard-16", vcpus=16, memory_gib=64)
