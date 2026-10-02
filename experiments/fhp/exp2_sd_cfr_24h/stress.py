"""Cloud-only mature replay/archive memory smoke; synthetic data are discarded."""
import argparse
from pathlib import Path
import resource
import sys
import tempfile

import numpy as np
import torch

from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive, DiskArchiveReader, DiskSampledPolicy, write_json
from deep_cfr_poker.sd_cfr_optimised import OptimisedSingleDeepCFRSolver
from deep_cfr_poker.seeding import set_seed
from .config import solver_config


def run(output, iterations=10000, capacity=5000000, *, solver_class=OptimisedSingleDeepCFRSolver):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    set_seed(77)
    config = solver_config()
    config["memory_capacity"] = capacity
    solver = solver_class(**config)
    allocated = 0
    for memory in solver._advantage_memories:
        # Fill with an actual encoded input, so structured fractions remain valid.
        state = solver._game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        packed = memory._pack(solver._information_state(state, state.current_player()))
        for array in (memory._info_states, memory._iterations, memory._targets):
            array[...] = packed if array is memory._info_states else 1
            allocated += array.nbytes
        memory._size = capacity
        memory._add_calls = capacity
        batch = memory.sample_batch(min(2048, capacity))
        if batch["info_states"].dtype != np.float32:
            raise RuntimeError("Replay decoding changed precision")
    with tempfile.TemporaryDirectory(prefix="sdcfr-archive-stress-") as temporary:
        root = Path(temporary)
        archive = DiskSDCFRArchive(solver, root)
        for iteration in range(1, iterations + 1):
            for player in (0, 1):
                archive.capture_from_solver(solver, player, iteration)
        checkpoint = archive.checkpoint(root / "stress.json")
        reader = DiskArchiveReader(checkpoint, solver._game)
        policy = DiskSampledPolicy(reader)
        policy.begin_episode(seed=77)
        state = solver._game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        if not np.isclose(sum(policy.action_probabilities(state).values()), 1):
            raise RuntimeError("Stress policy failed reload")
        archive_bytes = sum(c["size_bytes"] for c in archive.chunks)
    write_json(output, dict(passed=True, replay_rows_per_player=capacity,
                            replay_allocated_bytes=allocated, archive_iterations=iterations,
                            synthetic_archive_bytes=archive_bytes, synthetic_artifacts_retained=False,
                            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            * (1 if sys.platform == "darwin" else 1024)))


def main(*, solver_class=OptimisedSingleDeepCFRSolver):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, solver_class=solver_class)


if __name__ == "__main__":
    main()
