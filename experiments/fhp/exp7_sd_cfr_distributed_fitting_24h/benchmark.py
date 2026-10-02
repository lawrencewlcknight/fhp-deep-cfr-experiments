"""Same frozen replay/minibatches: central vs eight-worker fitting.

Timings include sampling, preparation, transfer, reduction and state sync,
but exclude one-time Ray/Gloo startup (reported separately). Not an end-to-end
node-throughput benchmark. Equivalence is a gate; faster fitting is not assumed.
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
from experiments.fhp.exp1_sd_cfr_efficiency.run import compare_trees, fingerprint
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


def run(output, *, repeats=3, updates=200, seed=0):
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
        # full 2048 batch, not the early-run full-buffer fallback. This is
        # untimed fixture generation, NOT a change to production traversals.
        for _ in range(20):
            solver._collect_traversals_for_player(0)
            if len(solver.advantage_buffers[0]) >= learner["batch_size_advantage"]:
                break
            solver._iteration += 1
        if len(solver.advantage_buffers[0]) < learner["batch_size_advantage"]:
            raise RuntimeError("Frozen source too small for production-sized minibatches")
        network, optimizer = solver._advantage_networks[0], solver._optimizer_advantages[0]
        x, legal = probes(solver)
        # Common short warm-start produces NONEMPTY Adam history in both arms.
        solver._advantage_network_train_steps = 5
        DeepCFRSolver._learn_advantage_network(solver, 0)
        # Warm communication/training too, before resetting both measured arms.
        solver._learn_advantage_network(0)
        solver._advantage_network_train_steps = updates
        initial = copy.deepcopy((network.state_dict(), optimizer.state_dict()))
        source_digest = fingerprint(solver.advantage_buffers[0].state_dict())
        for repeat in range(repeats):
            outcomes = {}
            for arm in (("central", "distributed") if repeat % 2 == 0 else ("distributed", "central")):
                network.load_state_dict(initial[0])
                optimizer.load_state_dict(copy.deepcopy(initial[1]))
                random.seed(8100 + repeat)
                tick = time.perf_counter()
                loss = (DeepCFRSolver._learn_advantage_network(solver, 0) if arm == "central"
                        else solver._learn_advantage_network(0))
                seconds = time.perf_counter() - tick
                with torch.no_grad():
                    values = network(x).numpy()
                probabilities = np.stack([regret_matching_probabilities(v, mask, 3)
                                          for v, mask in zip(values, legal)])
                outcomes[arm] = dict(state=copy.deepcopy((network.state_dict(), optimizer.state_dict())),
                                     logits=values, probabilities=probabilities,
                                     rng=random.getstate(), loss=loss)
                rows.append(dict(repeat=repeat, arm=arm, seconds=seconds, loss=loss,
                                 updates=updates, examples=updates * learner["batch_size_advantage"]))
            left, right = outcomes["central"], outcomes["distributed"]
            # A separate single-step gradient/Adam gate validates the learning
            # rule. This stricter full-fit gate also checks trajectory drift;
            # tiny FP32 differences can amplify over 200 nonlinear updates.
            state = compare_trees(left["state"], right["state"], atol=2e-4, rtol=2e-3)
            logits = compare_trees(left["logits"], right["logits"], atol=2e-4, rtol=2e-3)
            policy_delta = float(np.max(np.abs(left["probabilities"] - right["probabilities"])))
            passed = (state["near"] and logits["near"] and policy_delta <= 0.002
                      and left["rng"] == right["rng"]
                      and source_digest == fingerprint(solver.advantage_buffers[0].state_dict()))
            write_json(output / f"equivalence_{repeat}.json", dict(passed=passed, state=state,
                logits=logits, max_action_probability_difference=policy_delta,
                identical_sampling_rng=left["rng"] == right["rng"]))
            checks.append(passed)
        central = [r["seconds"] for r in rows if r["arm"] == "central"]
        distributed = [r["seconds"] for r in rows if r["arm"] == "distributed"]
        summary = dict(passed=all(checks), seed=seed, repeats=repeats, updates=updates,
            global_batch_size=learner["batch_size_advantage"], workers=solver.parallel_num_workers,
            frozen_rows=len(solver.advantage_buffers[0]), startup_seconds=startup_seconds,
            central_median_seconds=float(np.median(central)),
            distributed_median_seconds=float(np.median(distributed)),
            fitting_speedup=float(np.median(central) / np.median(distributed)),
            torch_version=torch.__version__, machine=platform.machine(), platform=platform.platform(),
            interpretation="Fitting only; production node throughput must be measured separately.",
            rows=rows)
        write_json(output / "summary.json", summary)
        if not all(checks):
            write_json(output / "FAILED.json", dict(reason="Frozen-fit trajectory equivalence gate failed"))
            raise RuntimeError("Full-fit trajectories differ beyond tolerance; review before paid training")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--seed", type=int, choices=config.SEEDS, default=0)
    args = parser.parse_args()
    print(run(args.output, repeats=args.repeats, updates=args.updates, seed=args.seed))


if __name__ == "__main__":
    main()
