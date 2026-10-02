"""Experiment 5's training protocol with final-state saving and continuation."""
from deep_cfr_poker.sd_cfr_parallel import ParallelStructuredSingleDeepCFRSolver
from experiments.fhp.exp2_sd_cfr_24h.train import main
from . import config

if __name__ == "__main__":
    main(experiment=config, solver_class=ParallelStructuredSingleDeepCFRSolver)
