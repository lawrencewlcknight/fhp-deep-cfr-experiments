"""Unchanged Experiment 3 solver and clock; separate hardware/run metadata."""
from deep_cfr_poker.sd_cfr_structured import StructuredSingleDeepCFRSolver
from experiments.fhp.exp2_sd_cfr_24h.train import main
from . import config

if __name__ == "__main__":
    main(experiment=config, solver_class=StructuredSingleDeepCFRSolver)
