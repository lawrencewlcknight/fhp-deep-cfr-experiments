#!/usr/bin/env python3
"""Experiment 4's hardware and protocol, with eight traversal workers."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcp import exp2_sd_cfr_24h_batch as base

EXPERIMENT = dict(number=5, module="experiments.fhp.exp5_sd_cfr_parallel_24h",
                  algorithm_id="parallel_structured_uniform_sd_cfr",
                  batch_script="gcp/exp5_sd_cfr_parallel_24h_batch.py",
                  test_file="tests/test_exp5_sd_cfr_parallel_24h.py",
                  extra_test_files=("tests/test_exp3_sd_cfr_structured.py",
                                    "tests/test_sd_cfr_parallel.py",
                                    "tests/test_sd_cfr_parallel_efficiency.py"),
                  parallel_smoke=True,
                  training_resources=dict(machine_type="n2-standard-16", cpu_milli=16000,
                                          memory_mib=60000))
STAGES = base.STAGES


def build_job(args, stage):
    args.experiment = EXPERIMENT
    return base.build_job(args, stage)


if __name__ == "__main__":
    base.main(experiment=EXPERIMENT)
