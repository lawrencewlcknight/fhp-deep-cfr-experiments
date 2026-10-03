"""Routine evaluation plus all-seed same-budget play against saved Experiment 5."""
from experiments.fhp.exp2_sd_cfr_24h.evaluate import main
from . import config, comparison

if __name__ == "__main__":
    main(experiment=config, comparison=comparison)
