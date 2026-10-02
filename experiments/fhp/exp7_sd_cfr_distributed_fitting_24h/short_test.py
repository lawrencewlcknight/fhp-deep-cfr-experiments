"""Standalone ~30-minute engineering screen; never launches long training.

Incorrect updates/non-finite values fail. Long-fit numerical drift is reported,
not suppressed and not confused with an infrastructure failure. The separate
24-hour experiment's stricter gate is not changed by this diagnostic.
"""
import argparse
import copy
import json
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time

import numpy as np
import torch

from deep_cfr_poker.sd_cfr import regret_matching_probabilities
from deep_cfr_poker.sd_cfr_distributed import DistributedFittingSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_parallel import ParallelStructuredSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive, DiskArchiveReader, write_json
from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.solver import DeepCFRSolver
from experiments.fhp.exp1_sd_cfr_efficiency.run import compare_trees, fingerprint, require_finite
from experiments.fhp.exp2_sd_cfr_24h.train import write_csv
from experiments.fhp.exp5_sd_cfr_parallel_24h import config as baseline
from . import config
from .benchmark import probes


ARMS = ("central", "distributed")
BUDGETS = dict(warmup=300.0, fitting=1200.0, end_to_end=300.0)


def order(pair):
    return ARMS if pair % 2 == 0 else ARMS[::-1]


def rng_digest():
    return fingerprint((random.getstate(), np.random.get_state(), torch.get_rng_state()))


def fit(solver, player, arm):
    if arm == "central":
        return DeepCFRSolver._learn_advantage_network(solver, player)
    if arm == "distributed":
        return solver._learn_advantage_network(player)
    raise ValueError(arm)


def learner_state(solver):
    return copy.deepcopy(([net.state_dict() for net in solver._advantage_networks],
                          [opt.state_dict() for opt in solver._optimizer_advantages]))


def restore_learner(solver, state):
    for net, weights in zip(solver._advantage_networks, state[0]):
        net.load_state_dict(weights)
    for opt, saved in zip(solver._optimizer_advantages, state[1]):
        opt.load_state_dict(copy.deepcopy(saved))


def check_update(solver, player, seed):
    """Hard gate: same input state/data, one step, actual all-reduced gradient."""
    original, steps = learner_state(solver), solver._advantage_network_train_steps
    solver._advantage_network_train_steps = 1
    solver._capture_fit_gradients = True
    results = {}
    try:
        for arm in ARMS:
            restore_learner(solver, original)
            set_seed(seed)
            loss = fit(solver, player, arm)
            gradient = (np.concatenate([p.grad.numpy().reshape(-1)
                        for p in solver._advantage_networks[player].parameters()])
                        if arm == "central" else solver._last_fit_gradient)
            state = learner_state(solver)
            require_finite((state, gradient, loss))
            results[arm] = dict(state=state, gradient=gradient.copy(), loss=loss, rng=rng_digest())
        a, b = (results[arm] for arm in ARMS)
        state = compare_trees(a["state"], b["state"], atol=3e-6, rtol=3e-5)
        gradient = compare_trees(a["gradient"], b["gradient"], atol=2e-6, rtol=3e-5)
        passed = (state["near"] and gradient["near"] and a["rng"] == b["rng"]
                  and bool(np.isclose(a["loss"], b["loss"], atol=3e-6, rtol=2e-5)))
        return dict(player=player, passed=passed, state=state, gradient=gradient,
                    identical_rng=a["rng"] == b["rng"], central_loss=a["loss"], distributed_loss=b["loss"])
    finally:
        restore_learner(solver, original)
        solver._advantage_network_train_steps = steps
        solver._capture_fit_gradients = False


def pair_metrics(rows):
    """Aggregate matched work, never treat timing repeats as training seeds."""
    grouped = {}
    for row in rows:
        if row["arm"] in grouped.get(row["pair"], {}):
            raise ValueError("Duplicate arm within timing pair")
        if not np.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("Timing must be positive and finite")
        grouped.setdefault(row["pair"], {})[row["arm"]] = row
    if not grouped or any(set(pair) != set(ARMS) for pair in grouped.values()):
        raise ValueError("Expected complete central/distributed pairs")
    ratios = []
    for pair in grouped.values():
        a, b = (pair[arm] for arm in ARMS)
        for key in ("updates", "examples", "outer_iterations", "root_traversals"):
            if key in a and a[key] != b[key]:
                raise ValueError(f"Unmatched work: {key}")
        ratios.append(a["seconds"] / b["seconds"])
    totals = {arm: sum(r["seconds"] for r in rows if r["arm"] == arm) for arm in ARMS}
    result = dict(pairs=len(grouped), central_seconds=totals["central"],
                  distributed_seconds=totals["distributed"],
                  matched_work_speedup=totals["central"] / totals["distributed"],
                  median_paired_speedup=float(np.median(ratios)))
    if "nodes" in rows[0]:
        rates = {arm: sum(r["nodes"] for r in rows if r["arm"] == arm) / totals[arm] for arm in ARMS}
        result.update(nodes_per_second=rates, node_throughput_ratio=rates["distributed"] / rates["central"])
    return result


def prepare(solver, output, warmup_seconds, smoke):
    start = time.perf_counter()
    write_json(output / "progress.json", dict(stage="worker_startup"))
    solver._start_fitting()
    startup = time.perf_counter() - start
    minimum = solver._batch_size_advantage
    target = 64 if smoke else 32768
    # Use a fixed-policy fixture with independently seeded collection phases.
    # Never repeat/tile a handful of rows to pretend the reservoir is full.
    for _ in range(10000):
        for player in (0, 1):
            solver._collect_traversals_for_player(player)
        sizes = [len(b) for b in solver.advantage_buffers]
        write_json(output / "progress.json", dict(stage="replay_collection", rows_per_player=sizes,
                                                  elapsed_seconds=time.perf_counter() - start))
        if min(sizes) >= target or time.perf_counter() - start >= 0.8 * warmup_seconds:
            break
        solver._iteration += 1
    if min(sizes) < minimum:
        raise RuntimeError("Warm-up did not collect a full global minibatch for both players")
    steps = solver._advantage_network_train_steps
    try:
        solver._advantage_network_train_steps = 2 if smoke else 5
        for player in (0, 1):
            fit(solver, player, "central")
            fit(solver, player, "distributed")
    finally:
        solver._advantage_network_train_steps = steps
    checks = [check_update(solver, player, 19000 + player) for player in (0, 1)]
    write_json(output / "update_checks.json", checks)
    if not all(row["passed"] for row in checks):
        raise RuntimeError("Single-update gradient/Adam correctness gate failed")
    return dict(seconds=time.perf_counter() - start, worker_startup_seconds=startup,
                rows_per_player=sizes, capacity_per_player=solver.advantage_buffers[0].capacity,
                target_rows_per_player=target, collection_phases=solver._iteration,
                fixture="independent frozen-policy traversals; common warm-start fits",
                represents_full_reservoir=False, update_checks_passed=True)


def fitting_screen(solver, output, seconds):
    initial = learner_state(solver)
    source_hash = fingerprint([b.state_dict() for b in solver.advantage_buffers])
    # Probes include player-observable information only; they are not an
    # exploitability estimate. Probe both player's fitted outputs consistently.
    probe_sets = [probes(solver, player) for player in (0, 1)]
    rows, differences = [], []
    start = time.perf_counter()
    pair = 0
    while pair < 4 or time.perf_counter() - start < seconds:
        # Each player receives both execution orders; do not confound player
        # identity with always going first or second.
        player, outcomes = (pair // 2) % 2, {}
        x, legal = probe_sets[player]
        for arm in order(pair):
            restore_learner(solver, initial)
            set_seed(27000 + pair)
            tick = time.perf_counter()
            loss = fit(solver, player, arm)
            duration = time.perf_counter() - tick
            with torch.no_grad():
                logits = solver._advantage_networks[player](x).numpy()
            probabilities = np.stack([regret_matching_probabilities(v, mask, 3)
                                      for v, mask in zip(logits, legal)])
            state = learner_state(solver)
            require_finite((state, logits, probabilities, loss))
            outcomes[arm] = dict(state=state, logits=logits, probabilities=probabilities,
                                 rng=rng_digest())
            row = dict(pair=pair, arm=arm, player=player, seconds=duration, loss=loss,
                       updates=solver._advantage_network_train_steps,
                       examples=solver._advantage_network_train_steps * solver._batch_size_advantage)
            if arm == "distributed":
                phase = solver.last_distributed_fit
                row.update(preparation_seconds=phase["preparation_seconds"],
                           mean_worker_compute_seconds=float(np.mean(phase["worker_compute_seconds"])),
                           mean_worker_communication_seconds=float(np.mean(phase["worker_communication_seconds"])))
            rows.append(row)
        a, b = (outcomes[arm] for arm in ARMS)
        if a["rng"] != b["rng"]:
            raise RuntimeError("Fitting arms consumed different sampling/RNG streams")
        state_diff = compare_trees(a["state"], b["state"], atol=2e-4, rtol=2e-3)
        logit_diff = compare_trees(a["logits"], b["logits"], atol=2e-4, rtol=2e-3)
        probability_delta = float(np.max(np.abs(a["probabilities"] - b["probabilities"])))
        differences.append(dict(pair=pair, player=player, same_sampling_rng=True,
            near_full_fit=bool(state_diff["near"] and logit_diff["near"] and probability_delta <= 0.002),
            state_max_abs_difference=state_diff["max_abs_difference"],
            logit_max_abs_difference=logit_diff["max_abs_difference"],
            max_action_probability_difference=probability_delta))
        pair += 1
        write_csv(output / "fitting_timings.csv", rows)
        write_csv(output / "fitting_differences.csv", differences)
        write_json(output / "progress.json", dict(stage="fitting", complete_pairs=pair,
                                                  elapsed_seconds=time.perf_counter() - start))
        print(json.dumps(dict(stage="fitting", pair=pair, near_full_fit=differences[-1]["near_full_fit"])), flush=True)
    if source_hash != fingerprint([b.state_dict() for b in solver.advantage_buffers]):
        raise RuntimeError("Frozen replay mutated during fitting comparison")
    return dict(**pair_metrics(rows), elapsed_seconds=time.perf_counter() - start,
                all_full_fits_near=all(d["near_full_fit"] for d in differences),
                worst_action_probability_difference=max(d["max_action_probability_difference"] for d in differences))


def end_to_end_arm(arm, iterations, pair, seed, smoke):
    """Fresh production traversal-fit-archive loop; no overlapping arm pools."""
    cls = (ParallelStructuredSingleDeepCFRSolver if arm == "central"
           else DistributedFittingSingleDeepCFRSolver)
    learner = dict(config.solver_config(smoke), num_iterations=iterations)
    execution = baseline.execution_config(seed) if arm == "central" else config.execution_config(seed)
    set_seed(seed)
    with cls(**learner, **execution) as solver, tempfile.TemporaryDirectory(prefix="sdcfr7-short-archive-") as temporary:
        tick = time.perf_counter()
        if arm == "central":
            solver._start_workers()
        else:
            solver._start_fitting()
        startup = time.perf_counter() - tick
        # Actual production archive-capture path, discarded after validation.
        solver.archive = DiskSDCFRArchive(solver, temporary, chunk_iterations=128)
        ticks = []
        tick = time.perf_counter()
        solver.solve(post_iteration_callback=lambda s, i: ticks.append(
            dict(iteration=i, seconds=time.perf_counter() - tick, nodes=s._nodes_touched)))
        duration = time.perf_counter() - tick
        require_finite(learner_state(solver))
        checkpoint = solver.archive.checkpoint(Path(temporary) / "final.json")
        reader = DiskArchiveReader(checkpoint, solver._game)
        if reader.count != iterations:
            raise RuntimeError("End-to-end history capture was incomplete")
        return dict(pair=pair, arm=arm, seconds=duration, startup_seconds=startup,
                    outer_iterations=iterations, nodes=solver._nodes_touched,
                    traversal_collection_seconds=solver._cumulative_traversal_collection_seconds,
                    root_traversals=iterations * 2 * learner["num_traversals"],
                    nodes_per_second=solver._nodes_touched / duration,
                    actual_replay_rows_p0=len(solver.advantage_buffers[0]),
                    actual_replay_rows_p1=len(solver.advantage_buffers[1]),
                    archive_bytes=sum(c["size_bytes"] for c in solver.archive.chunks),
                    archive_reload_valid=True), ticks


def end_to_end_screen(output, seconds, seed, smoke, runner=end_to_end_arm):
    rows, progress, pair, iterations = [], [], 0, 1
    while True:
        for arm in order(pair):
            row, ticks = runner(arm, iterations, pair, seed, smoke)
            rows.append(row)
            progress.extend(dict(pair=pair, arm=arm, **t) for t in ticks)
            write_csv(output / "end_to_end_timings.csv", rows)
            write_csv(output / "end_to_end_trajectory.csv", progress)
            write_json(output / "progress.json", dict(stage="end_to_end", complete_arms=len(rows)))
        spent = sum(row["seconds"] for row in rows)
        if pair >= 1 or (smoke and spent >= seconds):
            break
        # First matched pair is an actual one-iteration calibration. The second
        # pair reverses execution order and uses the SAME work in both arms.
        estimated_pair = sum(row["seconds"] for row in rows[-2:]) / iterations
        iterations = max(1, min(100, int(max(0.0, seconds - spent) / estimated_pair)))
        pair += 1
    return dict(**pair_metrics(rows), startup_seconds=sum(r["startup_seconds"] for r in rows),
                interpretation="Early-run throughput, matched root traversals; actual node counts may differ.")


def run(output, *, seed=0, smoke=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty short-test output directory")
    if seed not in config.SEEDS:
        raise ValueError("Unsupported seed")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    budgets = dict(warmup=120.0, fitting=0.001, end_to_end=0.001) if smoke else dict(BUDGETS)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    learner = config.solver_config(smoke)
    manifest = dict(experiment="exp7_short_distributed_fitting_screen", seed=seed, smoke=smoke,
        budgets_seconds=budgets, solver_config=learner, execution=config.execution_config(seed),
        machine_type="n2-standard-16 (cloud launcher)", local_platform=platform.platform(),
        machine=platform.machine(), torch_version=torch.__version__, numpy_version=np.__version__,
        repository_commit=commit, repository_dirty=bool(dirty),
        arms_run_concurrently=False, long_run_authorized=False,
        retention="Analysis, diagnostics and metadata only; no replay or model states",
        full_fit_drift_is_diagnostic=True, production_equivalence_gate_unchanged=True,
        timing="Complete matched pairs can overrun stage targets; startups separately reported")
    write_json(output / "manifest.json", manifest)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    set_seed(seed)
    try:
        with DistributedFittingSingleDeepCFRSolver(**learner, **config.execution_config(seed)) as solver:
            warmup = prepare(solver, output, budgets["warmup"], smoke)
            write_json(output / "warmup.json", warmup)
            fitting = fitting_screen(solver, output, budgets["fitting"])
            write_json(output / "fitting_summary.json", fitting)
        # The fitting actor pool is CLOSED before either end-to-end arm starts.
        end_to_end = end_to_end_screen(output, budgets["end_to_end"], seed, smoke)
        summary = dict(completed=True, correctness_checks_passed=True,
            numerical_drift_requires_review=not fitting["all_full_fits_near"],
            long_run_authorized=False, warmup=warmup, fitting=fitting, end_to_end=end_to_end,
            elapsed_seconds=time.perf_counter() - started,
            recommendation="Manual review required; this engineering screen does not establish poker strength or convergence.")
        write_json(output / "summary.json", summary)
        write_json(output / "SUCCESS.json", dict(completed=True, long_run_authorized=False,
                                                 numerical_drift_requires_review=summary["numerical_drift_requires_review"]))
        print(json.dumps(summary), flush=True)
        return summary
    except Exception as error:
        write_json(output / "FAILURE.json", dict(error=repr(error), elapsed_seconds=time.perf_counter() - started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=config.SEEDS, default=0)
    parser.add_argument("--smoke", action="store_true", help="Tiny integration check, not a performance result")
    args = parser.parse_args()
    run(args.output, seed=args.seed, smoke=args.smoke)


if __name__ == "__main__":
    main()
