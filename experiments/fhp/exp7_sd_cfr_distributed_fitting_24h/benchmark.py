"""Same frozen replay/minibatches: central vs eight-worker fitting.

Timings include sampling, preparation, transfer, reduction and state sync,
but exclude one-time Ray/Gloo startup (reported separately). Not an end-to-end
node-throughput benchmark. Correctness is mandatory. Accumulated drift is
diagnostic only when explicitly enabling the comparative-training experiment.
"""
import argparse
import copy
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from deep_cfr_poker.sd_cfr import regret_matching_probabilities
from deep_cfr_poker.sd_cfr_distributed import DistributedFittingSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_disk import write_json
from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.solver import DeepCFRSolver
from experiments.fhp.exp1_sd_cfr_efficiency.run import compare_trees, fingerprint, require_finite
from . import config


def probes(solver, player=0):
    rng = np.random.default_rng(517)
    features, legal = [], []
    for _ in range(64):
        state = solver._game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                action = int(rng.choice(actions, p=probs))
            else:
                actions = state.legal_actions()
                if state.current_player() == player:
                    features.append(solver._information_state(state, player))
                    legal.append(tuple(actions))
                action = int(rng.choice(actions))
            state.apply_action(action)
    return torch.from_numpy(np.asarray(features, dtype=np.float32)), legal


def validation_decision(*, correctness_passed, fits_near, allow_trajectory_drift):
    return dict(correctness_checks_passed=bool(correctness_passed),
                full_fit_equivalence_passed=bool(fits_near),
                numerical_drift_requires_review=not fits_near,
                validation_mode="comparative_training" if allow_trajectory_drift else "strict_equivalence",
                approved_for_comparative_run=bool(correctness_passed and (fits_near or allow_trajectory_drift)))


def run(output, *, repeats=3, updates=200, seed=0, allow_trajectory_drift=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("Benchmark output already exists")
    if repeats < 1 or updates < 1:
        raise ValueError("Positive repeats and updates required")
    torch.set_num_threads(1)
    set_seed(seed)
    learner = dict(config.solver_config(), memory_capacity=50000,
                   advantage_network_train_steps=updates, num_iterations=1)
    rows, checks = [], []
    with DistributedFittingSingleDeepCFRSolver(**learner, **config.execution_config(seed)) as solver:
        tick = time.perf_counter()
        solver._start_fitting()
        startup_seconds = time.perf_counter() - tick
        # Build enough independent frozen-policy observations to exercise the
        # full 2048 batch for BOTH players, not the early-run full-buffer fallback. This is
        # untimed fixture generation, NOT a change to production traversals.
        for _ in range(20):
            for player in (0, 1):
                solver._collect_traversals_for_player(player)
            if min(len(b) for b in solver.advantage_buffers) >= learner["batch_size_advantage"]:
                break
            solver._iteration += 1
        if min(len(b) for b in solver.advantage_buffers) < learner["batch_size_advantage"]:
            raise RuntimeError("Frozen source too small for production-sized minibatches")
        # Common short warm-start produces NONEMPTY Adam history in both arms.
        solver._advantage_network_train_steps = 5
        for player in (0, 1):
            DeepCFRSolver._learn_advantage_network(solver, player)
            # Warm communication/training before resetting both measured arms.
            solver._learn_advantage_network(player)
        solver._advantage_network_train_steps = updates
        # Reuse the short screen's tight actual-gradient/Adam/RNG checks. The
        # import is deferred because that module also uses probes() above.
        from .short_test import check_update
        update_checks = [check_update(solver, player, 19000 + player) for player in (0, 1)]
        write_json(output / "update_checks.json", update_checks)
        if not all(row["passed"] for row in update_checks):
            write_json(output / "FAILED.json", dict(reason="Single-update correctness failed"))
            raise RuntimeError("Single-update gradient/Adam correctness gate failed")
        source_digest = fingerprint([b.state_dict() for b in solver.advantage_buffers])
        initial = [copy.deepcopy((n.state_dict(), o.state_dict())) for n, o in
                   zip(solver._advantage_networks, solver._optimizer_advantages)]
        probe_sets = [probes(solver, player) for player in (0, 1)]
        for pair in range(2 * repeats):
            player, repeat = pair % 2, pair // 2
            network, optimizer = solver._advantage_networks[player], solver._optimizer_advantages[player]
            x, legal = probe_sets[player]
            outcomes = {}
            for arm in (("central", "distributed") if (repeat + player) % 2 == 0 else ("distributed", "central")):
                network.load_state_dict(initial[player][0])
                optimizer.load_state_dict(copy.deepcopy(initial[player][1]))
                random.seed(8100 + pair)
                tick = time.perf_counter()
                loss = (DeepCFRSolver._learn_advantage_network(solver, player) if arm == "central"
                        else solver._learn_advantage_network(player))
                seconds = time.perf_counter() - tick
                with torch.no_grad():
                    values = network(x).numpy()
                probabilities = np.stack([regret_matching_probabilities(v, mask, 3)
                                          for v, mask in zip(values, legal)])
                outcomes[arm] = dict(state=copy.deepcopy((network.state_dict(), optimizer.state_dict())),
                                     logits=values, probabilities=probabilities,
                                     rng=random.getstate(), loss=loss)
                require_finite((outcomes[arm]["state"], values, probabilities, loss))
                rows.append(dict(repeat=repeat, player=player, arm=arm, seconds=seconds, loss=loss,
                                 updates=updates, examples=updates * learner["batch_size_advantage"]))
            left, right = outcomes["central"], outcomes["distributed"]
            # A separate single-step gradient/Adam gate validates the learning
            # rule. This stricter full-fit gate also checks trajectory drift;
            # tiny FP32 differences can amplify over 200 nonlinear updates.
            state = compare_trees(left["state"], right["state"], atol=2e-4, rtol=2e-3)
            logits = compare_trees(left["logits"], right["logits"], atol=2e-4, rtol=2e-3)
            policy_delta = float(np.max(np.abs(left["probabilities"] - right["probabilities"])))
            same_rng = left["rng"] == right["rng"]
            same_replay = source_digest == fingerprint([b.state_dict() for b in solver.advantage_buffers])
            if not same_rng or not same_replay:
                write_json(output / "FAILED.json", dict(reason="Sampling or frozen replay changed"))
                raise RuntimeError("Fitting arms changed replay or consumed different sampling streams")
            passed = state["near"] and logits["near"] and policy_delta <= 0.002
            write_json(output / f"equivalence_p{player}_{repeat}.json", dict(passed=passed, state=state,
                logits=logits, max_action_probability_difference=policy_delta,
                identical_sampling_rng=left["rng"] == right["rng"]))
            checks.append(passed)
        central = [r["seconds"] for r in rows if r["arm"] == "central"]
        distributed = [r["seconds"] for r in rows if r["arm"] == "distributed"]
        summary = dict(passed=all(checks), seed=seed, repeats_per_player=repeats, updates=updates,
            **validation_decision(correctness_passed=True, fits_near=all(checks),
                                  allow_trajectory_drift=allow_trajectory_drift),
            global_batch_size=learner["batch_size_advantage"], workers=solver.parallel_num_workers,
            frozen_rows_per_player=[len(b) for b in solver.advantage_buffers], startup_seconds=startup_seconds,
            central_median_seconds=float(np.median(central)),
            distributed_median_seconds=float(np.median(distributed)),
            fitting_speedup=float(np.median(central) / np.median(distributed)),
            torch_version=torch.__version__, machine=platform.machine(), platform=platform.platform(),
            interpretation="Fitting only; production node throughput must be measured separately.",
            rows=rows)
        write_json(output / "summary.json", summary)
        if not summary["approved_for_comparative_run"]:
            write_json(output / "FAILED.json", dict(reason="Frozen-fit trajectory equivalence gate failed"))
            raise RuntimeError("Full-fit trajectories differ beyond tolerance; review before paid training")
        write_json(output / "SUCCESS.json", dict(status="comparative_preflight_complete",
            correctness_checks_passed=True, full_fit_equivalence_passed=all(checks),
            validation_mode=summary["validation_mode"]))
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--seed", type=int, choices=config.SEEDS, default=0)
    parser.add_argument("--allow-trajectory-drift", action="store_true",
                        help="Allow the approved quality-comparison arm; never bypass correctness checks")
    args = parser.parse_args()
    print(run(args.output, repeats=args.repeats, updates=args.updates, seed=args.seed,
              allow_trajectory_drift=args.allow_trajectory_drift))


if __name__ == "__main__":
    main()
