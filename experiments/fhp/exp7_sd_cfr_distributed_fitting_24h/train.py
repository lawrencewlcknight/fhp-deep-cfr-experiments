"""Shared 24-hour runner, now using synchronous distributed fitting."""
from deep_cfr_poker.sd_cfr_distributed import DistributedFittingSingleDeepCFRSolver
from experiments.fhp.exp2_sd_cfr_24h.train import main
from . import config

if __name__ == "__main__":
    main(experiment=config, solver_class=DistributedFittingSingleDeepCFRSolver)
