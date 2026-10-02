"""Unchanged structured replay and historical archive capacity checks."""
from experiments.fhp.exp2_sd_cfr_24h.stress import main
from deep_cfr_poker.sd_cfr_structured import StructuredSingleDeepCFRSolver

if __name__ == "__main__":
    main(solver_class=StructuredSingleDeepCFRSolver)
