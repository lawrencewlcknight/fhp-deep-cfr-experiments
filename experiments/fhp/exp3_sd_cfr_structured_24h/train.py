"""Reuse the baseline clock/checkpoint loop with the v1 observable encoder."""
from deep_cfr_poker.sd_cfr_structured import StructuredSingleDeepCFRSolver
from experiments.fhp.exp2_sd_cfr_24h.train import main
from . import config

if __name__ == "__main__":
    main(experiment=config, solver_class=StructuredSingleDeepCFRSolver)
