"""Freeze Experiment 4's learner and budgets; change traversal execution only."""
from experiments.fhp.exp4_sd_cfr_structured_n2_standard16.config import (
    FEATURE_ENCODER_METADATA, SEEDS, HOURS, SECONDS, ITERATION_CAP, REFERENCE_VM,
    solver_config as baseline_solver_config,
)

EXPERIMENT_NAME = "exp5_sd_cfr_parallel_24h"
ALGORITHM_ID = "parallel_structured_uniform_sd_cfr"
REPORT_ID = "sd_cfr_exp5_parallel8"
TRAVERSAL_WORKERS = 8


def solver_config(smoke=False):
    config = baseline_solver_config(smoke)
    if smoke:
        # Exercise ALL eight actors rather than leave four idle.
        config["num_traversals"] = TRAVERSAL_WORKERS
    return config


def execution_config(seed):
    if seed not in SEEDS:
        raise ValueError("Unexpected seed")
    return dict(parallel_num_workers=TRAVERSAL_WORKERS, parallel_run_seed=seed,
                parallel_backend="ray", parallel_chunk_rows=4096,
                parallel_max_rows_per_worker=1_000_000, parallel_timeout_seconds=300,
                parallel_ray_object_store_memory=512 * 1024 * 1024,
                parallel_inference_cache_entries=4096)


def task_name(seed):
    if seed not in SEEDS:
        raise ValueError("Unexpected seed")
    return f"task_{seed:03d}_{ALGORITHM_ID}_seed_{seed}"
