"""Shared analysis schema under a dedicated Experiment 7 identity."""
from experiments.fhp.exp2_sd_cfr_24h.report import main
from . import config

if __name__ == "__main__":
    main(experiment=config)
