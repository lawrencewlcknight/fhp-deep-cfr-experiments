"""Continuous training with durable policy prefixes and opt-in final resume state."""
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
    def __init__(self, clock=time.perf_counter, *, active=0.0, elapsed=0.0):
        self.clock, self.excluded = clock, elapsed - active
        self.elapsed_offset = elapsed
        self.started = clock()

    def __call__(self):
        return self.elapsed() - self.excluded

    def elapsed(self):
        return self.elapsed_offset + self.clock() - self.started

    @contextmanager
    def paused(self):
        start = self.clock()
        try:
            yield
        finally:
            self.excluded += self.clock() - start


def sync(root, uri):
    if uri:
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", "--exclude=.*[.]tmp$", str(root), uri], check=True)


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
               experiment=default_experiment, solver_class=OptimisedSingleDeepCFRSolver,
               resume_state=None, additional_hours=None):
    if seed not in experiment.SEEDS or (smoke and seed != 0):
        raise ValueError("Unexpected seed")
    root = Path(output_root) / "workers" / experiment.task_name(seed)
    retain_state = getattr(experiment, "RETAIN_FINAL_TRAINING_STATE", False)
    if (resume_state is None) != (additional_hours is None):
        raise ValueError("Resume requires both --resume-state and --additional-hours")
    if resume_state and (not retain_state or additional_hours not in range(6, 49, 6)):
        raise ValueError("Only resumable experiments accept an additional 6..48 hours in six-hour steps")
    # Policy-only checkpoints cannot resume a training trajectory. Full states
    # are restored into a new output directory, never over the source run.
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Non-empty worker directory: {root}. Use a fresh output directory.")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    set_seed(seed)
    config = experiment.solver_config(smoke)
    execution = (experiment.execution_config(seed) if hasattr(experiment, "execution_config") else {})
    previous = {}
    continuation = None
    hours = list(experiment.HOURS)
    if resume_state:
        from deep_cfr_poker.sd_cfr_training_state import inspect_training_state, load_training_state
        saved = inspect_training_state(resume_state, expected_config=config, expected_execution=execution)
        if saved["seed"] != seed or saved["smoke"] != smoke:
            raise ValueError("Resume seed/smoke mode differs from the saved run")
        previous_hours = saved["hours"]
        hours = list(range(6, previous_hours[-1] + additional_hours + 1, 6))
        if previous_hours != hours[:len(previous_hours)] or previous_hours[:len(experiment.HOURS)] != list(experiment.HOURS):
            raise ValueError("Resume checkpoint schedule differs from the experiment")
        continuation = dict(source_state_sha256=sha256(resume_state), source_hours=previous_hours,
                            additional_hours=additional_hours)
        solver, previous = load_training_state(resume_state, root / "archive", expected_config=config,
                                              expected_execution=execution, solver_class=solver_class)
        archive = solver.archive
    else:
        solver = solver_class(pack_replay=True, **config, **execution)
        archive = DiskSDCFRArchive(solver, root / "archive", chunk_iterations=2 if smoke else 128)
        solver.archive = archive
    schedule = [0.01 * (index + 1) for index in range(len(hours))] if smoke else [h * 3600 for h in hours]
    # Smoke continuation needs a fresh tiny budget even if the first real
    # iteration greatly overshot its artificial thresholds.
    if resume_state and smoke:
        schedule = [r["checkpoint_target_seconds"] for r in previous["records"]] + [
            previous["active_seconds"] + 0.01 * (i + 1) for i in range(additional_hours // 6)]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = dict(experiment_name=experiment.EXPERIMENT_NAME, algorithm_id=experiment.ALGORITHM_ID,
                    seed=seed, smoke=smoke, config=config, game=serialisable_game_definition(),
                    checkpoints_hours=list(hours), target_seconds=list(schedule),
                    feature_encoder=archive.metadata.get("feature_encoder"),
                    replay_encoding=solver._advantage_memories[0].feature_encoding,
                    reference_vm=experiment.REFERENCE_VM, torch_threads=1, interop_threads=1,
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    python_version=platform.python_version(), repository_commit=commit,
                    strategy_weighting="uniform", full_training_states_retained=retain_state,
                    checkpoint_boundary="first_completed_outer_iteration_crossing_threshold",
                    time_excludes="checkpoint_serialization_reload_validation_and_upload",
                    archive_capture_included_in_training_time=True,
                    node_definition="calls_to_external_sampling_traversal_including_terminal_states",
                    exact_exploitability=False)
    if continuation:
        manifest["continuation"] = continuation
    if execution:
        manifest.update(execution=execution, parallel_execution=archive.metadata["parallel_execution"],
                        parallel_startup_included_in_training_time=True,
                        peak_rss_scope="central_learner_only_excludes_ray_and_actors")
    write_json(root / "run_manifest.json", manifest)
    records, telemetry = previous.get("records", []), previous.get("telemetry", [])
    clock = ActiveClock(active=previous.get("active_seconds", 0.0),
                        elapsed=previous.get("elapsed_seconds", 0.0))
    completed_iteration_seconds = previous.get("active_seconds", 0.0)
    remaining_seconds = schedule[-1] - completed_iteration_seconds
    first_iteration = solver._iteration

    def observe(active_solver, iteration):
        nonlocal completed_iteration_seconds
        active_seconds = clock()
        completed_iteration_seconds = active_seconds
        usage = resource.getrusage(resource.RUSAGE_SELF)
        row = dict(seed=seed, iteration=iteration, active_seconds=active_seconds,
                   elapsed_seconds=clock.elapsed(),
                   nodes_touched=active_solver._nodes_touched,
                   replay_rows_p0=len(active_solver._advantage_memories[0]),
                   replay_rows_p1=len(active_solver._advantage_memories[1]),
                   peak_rss_bytes=int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
                   archive_iterations=archive.count,
                   archive_bytes=sum(chunk["size_bytes"] for chunk in archive.chunks),
                   checkpoint_overhead_seconds=clock.excluded)
        if execution:
            phase = active_solver.last_parallel_collection
            # This is the final player phase, not a whole-iteration timing.
            row.update(traversal_workers=active_solver.parallel_num_workers,
                       last_phase_player=phase["player"],
                       last_phase_traversals=phase["traversals"],
                       last_phase_collection_seconds=phase["seconds"],
                       last_phase_cache_hits=phase["inference_cache_hits"],
                       last_phase_cache_misses=phase["inference_cache_misses"])
        telemetry.append(row)
        if iteration % 25 == 0 or smoke:
            print(json.dumps(row), flush=True)
        due = [index for index in range(len(records), len(hours)) if active_seconds >= schedule[index]]
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
                                    wall_clock_seconds=clock.elapsed(),
                                    outer_iteration=iteration, nodes_touched=active_solver._nodes_touched,
                                    path=str(path.relative_to(root)), sha256=sha256(path),
                                    archive_bytes=sum(c["size_bytes"] for c in archive.chunks)))
            write_json(root / "checkpoint_manifest.json", records)
            write_csv(root / "checkpoint_manifest.csv", records)
            write_csv(root / "training_trajectory.csv", telemetry)
            sync(root, remote_uri)

    try:
        if remaining_seconds <= 0:
            raise ValueError("Saved training already exceeds the requested continuation endpoint")
        result = solver.solve(post_iteration_callback=observe,
                              max_training_seconds=remaining_seconds,
                              # Stop and checkpoint must use the SAME boundary
                              # observation: logging could cross the deadline
                              # after the callback checked it, otherwise ending
                              # the run without its final playable checkpoint.
                              training_clock=lambda: completed_iteration_seconds)
        if len(records) != len(hours):
            raise RuntimeError("Iteration safety cap reached before all time checkpoints; run is incomplete")
        write_csv(root / "training_trajectory.csv", telemetry)
        # Per-iteration regression losses and standard solver diagnostics.
        loss_rows = previous.get("loss_rows", []) + [dict(iteration=index + first_iteration, player=player,
                         advantage_loss=float(value) if np.isfinite(value) else None)
                     for player, values in result.advantage_losses.items()
                     for index, value in enumerate(values)]
        write_csv(root / "advantage_losses.csv", loss_rows)
        prior_diagnostics = previous.get("diagnostics", [])
        diagnostics = prior_diagnostics + [dict(checkpoint_index=i + len(prior_diagnostics), **{key: (None if isinstance(values[i], float)
                        and not np.isfinite(values[i]) else values[i])
                        for key, values in result.diagnostics.items()})
                       for i in range(len(result.diagnostics["iteration"]))]
        for row in diagnostics[len(prior_diagnostics):]:
            row["wall_clock_seconds"] += previous.get("elapsed_seconds", 0.0)
        write_csv(root / "solver_diagnostics.csv", diagnostics)
        state_metadata = {}
        if retain_state:
            from deep_cfr_poker.sd_cfr_training_state import save_training_state
            # Save AFTER final diagnostics: these can sample replay and advance
            # the learner RNG. Snapshot/validation I/O is not active training.
            context = dict(seed=seed, smoke=smoke, hours=hours, records=records, telemetry=telemetry,
                           loss_rows=loss_rows, diagnostics=diagnostics,
                           active_seconds=clock(), elapsed_seconds=clock.elapsed())
            with clock.paused():
                state_path = save_training_state(solver, root, root / records[-1]["path"],
                                                 config=config, execution=execution, context=context)
                state_metadata = dict(training_state_path=str(state_path.relative_to(root)),
                                      training_state_sha256=sha256(state_path),
                                      training_state_bytes=sum(p.stat().st_size for p in state_path.parent.iterdir()))
        write_json(root / "SUCCESS.json", dict(seed=seed, checkpoints=len(hours),
                    active_seconds=clock(), elapsed_seconds=clock.elapsed(),
                    checkpoint_overhead_seconds=clock.excluded,
                    final_nodes=solver._nodes_touched, completed_iterations=archive.count,
                    full_training_states_retained=retain_state, **state_metadata))
        sync(root, remote_uri)
    except Exception as error:
        write_json(root / "FAILURE.json", dict(error=repr(error), active_seconds=clock()))
        raise
    finally:
        close = getattr(solver, "close", None)
        if close is not None:
            close()
    return root


def main(*, experiment=default_experiment, solver_class=OptimisedSingleDeepCFRSolver):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=experiment.SEEDS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--remote-uri")
    parser.add_argument("--resume-state", type=Path, help="Trusted final training_state/manifest.json; requires its adjacent archive")
    parser.add_argument("--additional-hours", type=int, help="Extra active hours after the source nominal endpoint (6..48, step 6)")
    args = parser.parse_args()
    run_worker(args.output_root, args.seed, smoke=args.smoke, remote_uri=args.remote_uri,
               experiment=experiment, solver_class=solver_class,
               resume_state=args.resume_state, additional_hours=args.additional_hours)


if __name__ == "__main__":
    main()
