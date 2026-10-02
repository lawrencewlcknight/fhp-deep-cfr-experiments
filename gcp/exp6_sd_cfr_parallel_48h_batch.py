#!/usr/bin/env python3
"""Experiment 5 extended to 48 hours; retain final training states only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcp import exp2_sd_cfr_24h_batch as base

EXPERIMENT = dict(number=6, hours=48, module="experiments.fhp.exp6_sd_cfr_parallel_48h",
                  algorithm_id="parallel_structured_uniform_sd_cfr_48h",
                  batch_script="gcp/exp6_sd_cfr_parallel_48h_batch.py",
                  test_file="tests/test_exp6_sd_cfr_parallel_48h.py",
                  extra_test_files=("tests/test_sd_cfr_training_state.py",
                                    "tests/test_exp5_sd_cfr_parallel_24h.py",
                                    "tests/test_sd_cfr_parallel_efficiency.py"),
                  parallel_smoke=True, final_training_state=True,
                  train_max_seconds=72 * 3600,
                  training_resources=dict(machine_type="n2-standard-16", cpu_milli=16000,
                                          memory_mib=60000))
STAGES = base.STAGES


def build_job(args, stage):
    args.experiment = EXPERIMENT
    return base.build_job(args, stage)


if __name__ == "__main__":
    base.main(experiment=EXPERIMENT)
