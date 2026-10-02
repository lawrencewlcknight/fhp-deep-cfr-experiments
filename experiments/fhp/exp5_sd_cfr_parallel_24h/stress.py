"""Central replay/archive capacity gate; actual actors are tested separately."""
from deep_cfr_poker.sd_cfr_structured import StructuredSingleDeepCFRSolver
from experiments.fhp.exp2_sd_cfr_24h.stress import main

if __name__ == "__main__":
    main(solver_class=StructuredSingleDeepCFRSolver)
