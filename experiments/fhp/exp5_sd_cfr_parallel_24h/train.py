"""Identical training/checkpoint protocol, using eight traversal actors."""
from deep_cfr_poker.sd_cfr_parallel import ParallelStructuredSingleDeepCFRSolver
from experiments.fhp.exp2_sd_cfr_24h.train import main
from . import config

if __name__ == "__main__":
    main(experiment=config, solver_class=ParallelStructuredSingleDeepCFRSolver)
