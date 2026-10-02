"""Identical standalone rule/LBR/temporal protocol, versioned policy loading."""
from experiments.fhp.exp2_sd_cfr_24h.evaluate import main
from . import config

if __name__ == "__main__":
    main(experiment=config)
