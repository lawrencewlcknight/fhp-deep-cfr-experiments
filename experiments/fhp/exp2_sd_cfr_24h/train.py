"""Continuous training, four durable policy prefixes, no full replay dumps."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from deep_cfr_poker.game import serialisable_game_definition
from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive, DiskArchiveReader, DiskSampledPolicy, sha256, write_json
from deep_cfr_poker.sd_cfr_optimised import OptimisedSingleDeepCFRSolver
from deep_cfr_poker.seeding import set_seed
from . import config as default_experiment


class ActiveClock:
    def __init__(self, clock=time.perf_counter):
        self.clock, self.excluded = clock, 0.0
        self.started = clock()

    def __call__(self):
        return self.clock() - self.started - self.excluded

    @contextmanager
    def paused(self):
        start = self.clock()
        try:
            yield
        finally:
            self.excluded += self.clock() - start


def sync(root, uri):
    if uri:
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", "--exclude=\\.tmp$", str(root), uri], check=True)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty result table")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_worker(output_root, seed, *, smoke=False, remote_uri=None,
               experiment=default_experiment, solver_class=OptimisedSingleDeepCFRSolver):
    if seed not in experiment.SEEDS or (smoke and seed != 0):
        raise ValueError("Unexpected seed")
    root = Path(output_root) / "workers" / experiment.task_name(seed)
    # Never pretend policy-only checkpoints can resume a training trajectory.
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Non-empty worker directory: {root}. Use a fresh run; no replay state is retained.")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    set_seed(seed)
    config = experiment.solver_config(smoke)
    solver = solver_class(pack_replay=True, **config)
    archive = DiskSDCFRArchive(solver, root / "archive", chunk_iterations=2 if smoke else 128)
    solver.archive = archive
    schedule = (0.01, 0.02, 0.03, 0.04) if smoke else experiment.SECONDS
    hours = experiment.HOURS
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = dict(experiment_name=experiment.EXPERIMENT_NAME, algorithm_id=experiment.ALGORITHM_ID,
                    seed=seed, smoke=smoke, config=config, game=serialisable_game_definition(),
                    checkpoints_hours=list(hours), target_seconds=list(schedule),
                    feature_encoder=archive.metadata.get("feature_encoder"),
                    replay_encoding=solver._advantage_memories[0].feature_encoding,
                    reference_vm=experiment.REFERENCE_VM, torch_threads=1, interop_threads=1,
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    python_version=platform.python_version(), repository_commit=commit,
                    strategy_weighting="uniform", full_training_states_retained=False,
                    checkpoint_boundary="first_completed_outer_iteration_crossing_threshold",
                    time_excludes="checkpoint_serialization_reload_validation_and_upload",
                    archive_capture_included_in_training_time=True,
                    node_definition="calls_to_external_sampling_traversal_including_terminal_states",
                    exact_exploitability=False)
    write_json(root / "run_manifest.json", manifest)
    records, telemetry = [], []
    clock = ActiveClock()
    completed_iteration_seconds = 0.0

    def observe(active_solver, iteration):
        nonlocal completed_iteration_seconds
        active_seconds = clock()
        completed_iteration_seconds = active_seconds
        usage = resource.getrusage(resource.RUSAGE_SELF)
        row = dict(seed=seed, iteration=iteration, active_seconds=active_seconds,
                   elapsed_seconds=time.perf_counter() - clock.started,
                   nodes_touched=active_solver._nodes_touched,
                   replay_rows_p0=len(active_solver._advantage_memories[0]),
                   replay_rows_p1=len(active_solver._advantage_memories[1]),
                   peak_rss_bytes=int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
                   archive_iterations=archive.count,
                   archive_bytes=sum(chunk["size_bytes"] for chunk in archive.chunks),
                   checkpoint_overhead_seconds=clock.excluded)
        telemetry.append(row)
        if iteration % 25 == 0 or smoke:
            print(json.dumps(row), flush=True)
        due = [index for index in range(len(records), 4) if active_seconds >= schedule[index]]
        if not due:
            return
        with clock.paused():
            for index in due:
                path = archive.checkpoint(root / "archive" / f"time_{hours[index]:02d}h.json")
                reader = DiskArchiveReader(path, active_solver._game)
                policy = DiskSampledPolicy(reader)
                policy.begin_episode(seed=123456)
                state = active_solver._game.new_initial_state()
                while state.is_chance_node():
                    state.apply_action(state.chance_outcomes()[0][0])
                probabilities = policy.action_probabilities(state)
                if not np.isclose(sum(probabilities.values()), 1.0):
                    raise RuntimeError("Reloaded policy invalid")
                records.append(dict(seed=seed, checkpoint_index=index,
                                    checkpoint_id=f"time_{hours[index]:02d}h",
                                    checkpoint_target_hours=hours[index],
                                    checkpoint_target_seconds=schedule[index],
                                    actual_training_elapsed_seconds=active_seconds,
                                    wall_clock_seconds=time.perf_counter() - clock.started,
                                    outer_iteration=iteration, nodes_touched=active_solver._nodes_touched,
                                    path=str(path.relative_to(root)), sha256=sha256(path),
                                    archive_bytes=sum(c["size_bytes"] for c in archive.chunks)))
            write_json(root / "checkpoint_manifest.json", records)
            write_csv(root / "checkpoint_manifest.csv", records)
            write_csv(root / "training_trajectory.csv", telemetry)
            sync(root, remote_uri)

    try:
        result = solver.solve(post_iteration_callback=observe,
                              max_training_seconds=schedule[-1],
                              # Stop and checkpoint must use the SAME boundary
                              # observation: logging could cross the deadline
                              # after the callback checked it, otherwise ending
                              # the run without its final playable checkpoint.
                              training_clock=lambda: completed_iteration_seconds)
        if len(records) != 4:
            raise RuntimeError("Iteration safety cap reached before all time checkpoints; run is incomplete")
        write_csv(root / "training_trajectory.csv", telemetry)
        # Per-iteration regression losses and standard solver diagnostics.
        loss_rows = [dict(iteration=index + 1, player=player,
                         advantage_loss=float(value) if np.isfinite(value) else None)
                     for player, values in result.advantage_losses.items()
                     for index, value in enumerate(values)]
        write_csv(root / "advantage_losses.csv", loss_rows)
        diagnostics = [dict(checkpoint_index=i, **{key: (None if isinstance(values[i], float)
                        and not np.isfinite(values[i]) else values[i])
                        for key, values in result.diagnostics.items()})
                       for i in range(len(result.diagnostics["iteration"]))]
        write_csv(root / "solver_diagnostics.csv", diagnostics)
        write_json(root / "SUCCESS.json", dict(seed=seed, checkpoints=4,
                    active_seconds=clock(), elapsed_seconds=time.perf_counter() - clock.started,
                    checkpoint_overhead_seconds=clock.excluded,
                    final_nodes=solver._nodes_touched, completed_iterations=archive.count,
                    full_training_states_retained=False))
        sync(root, remote_uri)
    except Exception as error:
        write_json(root / "FAILURE.json", dict(error=repr(error), active_seconds=clock()))
        raise
    return root


def main(*, experiment=default_experiment, solver_class=OptimisedSingleDeepCFRSolver):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=experiment.SEEDS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--remote-uri")
    args = parser.parse_args()
    run_worker(args.output_root, args.seed, smoke=args.smoke, remote_uri=args.remote_uri,
               experiment=experiment, solver_class=solver_class)


if __name__ == "__main__":
    main()
