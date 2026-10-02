#!/usr/bin/env python3
"""Experiment 3 uses exactly the baseline Batch resources and stage ordering."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gcp import exp2_sd_cfr_24h_batch as base

EXPERIMENT = dict(number=3, module="experiments.fhp.exp3_sd_cfr_structured_24h",
                  algorithm_id="structured_uniform_sd_cfr",
                  batch_script="gcp/exp3_sd_cfr_structured_24h_batch.py",
                  test_file="tests/test_exp3_sd_cfr_structured.py")
STAGES = base.STAGES


def build_job(args, stage):
    args.experiment = EXPERIMENT
    return base.build_job(args, stage)


if __name__ == "__main__":
    base.main(experiment=EXPERIMENT)
