#!/usr/bin/env python3
"""Same Experiment 5 resources; traversal AND fitting on eight CPU actors."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcp import exp2_sd_cfr_24h_batch as base

EXPERIMENT = dict(number=7, module="experiments.fhp.exp7_sd_cfr_distributed_fitting_24h",
    algorithm_id="distributed_fitting_structured_uniform_sd_cfr",
    batch_script="gcp/exp7_sd_cfr_distributed_fitting_24h_batch.py",
    test_file="tests/test_exp7_sd_cfr_distributed_fitting.py",
    extra_test_files=("tests/test_sd_cfr_distributed.py", "tests/test_sd_cfr_parallel.py", "tests/test_exp7_comparison.py"),
    parallel_smoke=True,
    fitting_benchmark=True,
    allow_trajectory_drift=True,
    default_lbr="0",
    comparison=dict(module="experiments.fhp.exp5_sd_cfr_parallel_24h",
                    algorithm_id="parallel_structured_uniform_sd_cfr",
                    default_run_id="sdcfr5-par8-20261002-102757"),
    training_resources=dict(machine_type="n2-standard-16", cpu_milli=16000, memory_mib=60000))
STAGES = base.STAGES


def build_job(args, stage):
    args.experiment = EXPERIMENT
    return base.build_job(args, stage)


if __name__ == "__main__":
    base.main(experiment=EXPERIMENT)
