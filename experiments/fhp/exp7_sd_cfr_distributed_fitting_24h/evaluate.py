"""Unchanged rule-agent, LBR and temporal head-to-head evaluation."""
from experiments.fhp.exp2_sd_cfr_24h.evaluate import main
from . import config

if __name__ == "__main__":
    main(experiment=config)
