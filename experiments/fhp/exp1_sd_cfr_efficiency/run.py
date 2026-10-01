"""Run each arm in a fresh process, sequentially on the same host.

Example: python -m experiments.fhp.exp1_sd_cfr_efficiency.run --smoke
No cloud resources are submitted by this Python entry point.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from deep_cfr_poker.sd_cfr import SDCFRArchive, _build_snapshot_network, regret_matching_probabilities
from deep_cfr_poker.sd_cfr_optimised import OptimisedSingleDeepCFRSolver
from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.single_solver import SingleDeepCFRSolver
from .config import ARMS, ATOL, RTOL, DEFAULT_REPEATS, DEFAULT_SEEDS, EXPERIMENT_NAME, solver_config


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(value):
    """Stable content digest, not pickle/zip serialisation metadata."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def require_finite(value):
    """NaNs are allowed only in explicitly unavailable diagnostics, not here."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite learner/policy tensor")
    elif isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise RuntimeError("Non-finite learner/policy array")
    elif isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite(item)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise RuntimeError("Non-finite learner/policy scalar")


def compare_trees(left, right, *, atol=ATOL, rtol=RTOL):
    """Numerical tolerance only for floats; structure/integers must match."""
    report = {"exact": True, "near": True, "max_abs_difference": 0.0,
              "failures": [], "arrays_compared": 0}

    def fail(path):
        report["exact"] = report["near"] = False
        if len(report["failures"]) < 20:
            report["failures"].append(path)

    def visit(a, b, path):
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            report["arrays_compared"] += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                fail(path + ": shape/dtype")
                return
            if a.dtype.kind in "fc":
                # Diagnostic NaNs must be in the same positions. Infinities
                # are never accepted; all training arrays are checked below.
                finite = np.isfinite(a) & np.isfinite(b)
                delta = float(np.max(np.abs(a[finite].astype(np.float64) -
                                           b[finite].astype(np.float64)), initial=0))
                report["max_abs_difference"] = max(report["max_abs_difference"], delta)
                equal = np.array_equal(a, b, equal_nan=True)
                near = (not np.isinf(a).any() and not np.isinf(b).any()
                        and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
            else:
                equal = near = np.array_equal(a, b)
            report["exact"] &= bool(equal)
            if not near:
                fail(path)
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                fail(path + ": keys")
            else:
                for key in a:
                    visit(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, type(a)):
            if len(a) != len(b):
                fail(path + ": length")
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, f"{path}/{i}")
        elif isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            visit(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), path)
        elif type(a) is not type(b) or a != b:
            fail(path)

    visit(left, right, "root")
    return report


def policy_probes(solver):
    """Public-observation probes for every historical network; no tree scan.

    The returned arrays are a behavioural regression test, NOT an unweighted
    approximation to SD-CFR's behavioural average or an exploitability metric.
    """
    rng = np.random.default_rng(707)
    states = []
    for _ in range(24):
        state = solver._game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                action = int(rng.choice(actions, p=probs))
            else:
                states.append(state.clone())
                action = int(rng.choice(state.legal_actions()))
            state.apply_action(action)
    rows = {}
    for player, entries in solver.archive.entries_by_player.items():
        observations = [s for s in states if s.current_player() == player]
        inputs = torch.tensor(np.asarray([s.information_state_tensor(player)
                                         for s in observations]), dtype=torch.float32)
        for entry in entries:
            network = _build_snapshot_network(solver.archive, entry)
            with torch.inference_mode():
                raw = network(inputs).numpy()
            probabilities = np.stack([regret_matching_probabilities(
                values, state.legal_actions(), solver._num_actions)
                for values, state in zip(raw, observations)])
            rows[f"player{player}_iteration{entry.iteration}"] = {
                "raw_advantages": raw, "action_probabilities": probabilities,
            }
    # Check the actual deployed trajectory sampler too, with one archive draw
    # per player/hand. Compare identical deals/actions and returns across arms.
    policy = solver.make_policy(seed=991)
    trajectories = []
    for _ in range(16):
        policy.resample_episode()
        record = {"selected": dict(policy.selected_iterations), "decisions": []}
        state = solver._game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
            else:
                distribution = policy.action_probabilities(state)
                actions, probs = zip(*sorted(distribution.items()))
                record["decisions"].append((state.current_player(), tuple(actions),
                                             np.asarray(probs)))
            state.apply_action(int(rng.choice(actions, p=probs)))
        record["returns"] = np.asarray(state.returns())
        trajectories.append(record)
    return {"historical": rows, "sampled_play": trajectories}


def run_worker(args):
    source_before = source_hashes()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.ones(64, 64).square().sum().item()
    config = solver_config(args.smoke)
    set_seed(args.seed)
    started = time.perf_counter()
    solver = (SingleDeepCFRSolver(**config) if args.arm == "reference" else
              OptimisedSingleDeepCFRSolver(pack_replay=args.arm == "scripted_packed", **config))
    init_seconds = time.perf_counter() - started
    phase_seconds = {"collection": 0.0, "fitting": 0.0, "archive": 0.0}

    def timed(obj, name, label):
        original = getattr(obj, name)

        def wrapped(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                phase_seconds[label] += time.perf_counter() - start
        setattr(obj, name, wrapped)

    timed(solver, "_collect_traversals_for_player", "collection")
    timed(solver, "_learn_advantage_network", "fitting")
    timed(solver.archive, "capture_from_solver", "archive")
    boundaries = []
    started = time.perf_counter()

    def completed(active, iteration):
        boundaries.append({"iteration": iteration, "seconds": time.perf_counter() - started,
                           "nodes": active._nodes_touched})

    result = solver.solve(post_iteration_callback=completed)
    train_seconds = time.perf_counter() - started
    # Warm-up is the first complete CFR iteration, not a separate training run.
    steady_seconds = train_seconds - boundaries[0]["seconds"]
    steady_nodes = solver._nodes_touched - boundaries[0]["nodes"]
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss *= 1 if sys.platform == "darwin" else 1024
    # Capture RNG before post-training validation (which is outside all timing).
    rng_state = {"python": random.getstate(), "numpy": np.random.get_state(),
                 "torch": torch.get_rng_state()}
    archive = {p: [{"iteration": e.iteration, "weights": dict(e.state_dict)} for e in es]
               for p, es in solver.archive.entries_by_player.items()}
    replay = [{"capacity": b.capacity, "add_calls": b.add_calls, **b.as_batch(copy=True)}
              for b in solver.advantage_buffers]
    diagnostics = {k: v for k, v in result.diagnostics.items() if "seconds" not in k}
    validation = {"archive": archive, "replay": replay,
                  "optimisers": [o.state_dict() for o in solver._optimizer_advantages],
                  "rng": rng_state, "losses": result.advantage_losses,
                  "diagnostics": diagnostics,
                  "nodes_by_iteration": [b["nodes"] for b in boundaries],
                  "policy": policy_probes(solver)}
    # Any non-finite learner values invalidate the run, even if both arms agree.
    for name in ("archive", "optimisers", "replay", "losses", "policy"):
        require_finite(validation[name])
    temporary = Path(args.validation)
    # Exercise the normal portable archive format, not scripted serialisation.
    archive_path = temporary.with_suffix(".archive.pt")
    solver.save_archive(archive_path)
    restored = SDCFRArchive.load(archive_path)
    restored_weights = {p: [{"iteration": e.iteration, "weights": dict(e.state_dict)} for e in es]
                        for p, es in restored.entries_by_player.items()}
    if not compare_trees(archive, restored_weights)["exact"]:
        raise RuntimeError("Archive save/reload changed weights")
    torch.save(validation, temporary)
    arrays = [arr for b in solver.advantage_buffers
              for arr in (b._info_states, b._iterations, b._targets)]
    row_bytes = sum(b._info_states.shape[1] * b._info_states.dtype.itemsize
                    + b._iterations.dtype.itemsize
                    + b._targets.shape[1] * b._targets.dtype.itemsize
                    for b in solver.advantage_buffers)
    record = {"arm": args.arm, "seed": args.seed, "repeat": args.repeat,
              "initialisation_seconds": init_seconds, "training_seconds": train_seconds,
              "initialisation_plus_training_seconds": init_seconds + train_seconds,
              "steady_training_seconds": steady_seconds, "steady_nodes": steady_nodes,
              "nodes": solver._nodes_touched, "nodes_per_second": solver._nodes_touched / train_seconds,
              "steady_nodes_per_second": steady_nodes / steady_seconds,
              "phase_seconds": phase_seconds, "iteration_boundaries": boundaries,
              "peak_training_process_rss_bytes": peak_rss,
              "replay_allocated_bytes": sum(x.nbytes for x in arrays),
              "replay_used_rows": [len(b) for b in solver.advantage_buffers],
              "projected_replay_array_bytes_at_5m_per_player": 5_000_000 * row_bytes,
              "fingerprints": {key: fingerprint(value) for key, value in validation.items()},
              "portable_archive_roundtrip_exact": True}
    if source_hashes() != source_before:
        raise RuntimeError("Source files changed while a benchmark fit was running")
    record["source_sha256"] = source_before
    write_json(args.result, record)


def validate_pair(reference, candidate):
    if reference.keys() != candidate.keys():
        return {"passed": False, "bit_identical": False, "sampling_and_work_exact": False,
                "groups": {}, "error": "Missing or extra validation groups"}
    groups = {key: compare_trees(value, candidate[key]) for key, value in reference.items()}
    # Features, iterations, sampler state and work cannot merely be "close".
    structural = groups["rng"]["exact"] and groups["nodes_by_iteration"]["exact"]
    for a, b in zip(reference["replay"], candidate["replay"]):
        structural &= compare_trees({k: v for k, v in a.items() if k != "targets"},
                                    {k: v for k, v in b.items() if k != "targets"})["exact"]
    return {"passed": bool(structural and all(g["near"] for g in groups.values())),
            "bit_identical": all(g["exact"] for g in groups.values()),
            "sampling_and_work_exact": bool(structural), "groups": groups}


def summarise(records, comparisons):
    output = {"equivalence_passed": all(row["passed"] for row in comparisons),
              "all_bit_identical": all(row["bit_identical"] for row in comparisons),
              "atol": ATOL, "rtol": RTOL, "arms": {},
              "repeat_fingerprints_identical": all(
                  len({json.dumps(r["fingerprints"], sort_keys=True) for r in records
                       if r["seed"] == seed and r["arm"] == arm}) == 1
                  for seed in {r["seed"] for r in records} for arm in ARMS)}
    for arm in ARMS[1:]:
        pairs = []
        for row in records:
            if row["arm"] != arm:
                continue
            ref = next(x for x in records if x["arm"] == "reference"
                       and x["seed"] == row["seed"] and x["repeat"] == row["repeat"])
            pairs.append({"seed": row["seed"], "repeat": row["repeat"],
                          "speedup": ref["training_seconds"] / row["training_seconds"],
                          "time_saved_percent": 100 * (1 - row["training_seconds"] / ref["training_seconds"]),
                          "including_initialisation_speedup": ref["initialisation_plus_training_seconds"]
                          / row["initialisation_plus_training_seconds"],
                          "steady_speedup": ref["steady_training_seconds"] / row["steady_training_seconds"]})
        seed_summaries = [{"seed": seed, "median_speedup": float(np.median(
            [p["speedup"] for p in pairs if p["seed"] == seed]))}
            for seed in sorted({p["seed"] for p in pairs})]
        output["arms"][arm] = {
            "pairs": pairs, "by_seed": seed_summaries,
            "median_paired_speedup": float(np.median([p["speedup"] for p in pairs])),
            "median_time_saved_percent": float(np.median([p["time_saved_percent"] for p in pairs])),
            "median_including_initialisation_speedup": float(np.median(
                [p["including_initialisation_speedup"] for p in pairs])),
            "median_steady_speedup": float(np.median([p["steady_speedup"] for p in pairs])),
            "minimum_paired_speedup": min(p["speedup"] for p in pairs),
            "maximum_paired_speedup": max(p["speedup"] for p in pairs),
            "speedup_valid_for_matched_work": all(c["sampling_and_work_exact"]
                for c in comparisons if c["arm"] == arm),
            "eligible_for_promotion": all(c["passed"] for c in comparisons if c["arm"] == arm),
        }
    return output


def render_report(output, records, summary):
    lines = ["# SD-CFR implementation benchmark", "",
             f"Equivalence passed: **{summary['equivalence_passed']}**. "
             f"All compared numerical outputs identical: **{summary['all_bit_identical']}**.", "",
             "| Arm | Median speedup | Median training time saved | Speedup incl. initialisation |",
             "| --- | ---: | ---: | ---: |"]
    for arm, stats in summary["arms"].items():
        lines.append(f"| {arm} | {stats['median_paired_speedup']:.3f}x | "
                     f"{stats['median_time_saved_percent']:.2f}% | "
                     f"{stats['median_including_initialisation_speedup']:.3f}x |")
    lines += ["", "Speedup below 1 means slower. Failed equivalence invalidates promotion; "
              "unequal node counts invalidate an equal-work speed claim.", "",
              "Fresh processes run sequentially with counterbalanced order. Repeats measure timing "
              "noise, not additional independent training seeds. Timed training includes traversal, "
              "fitting, archive capture and unchanged diagnostics. Initialisation includes compilation; "
              "policy probes, full-state comparisons and disk I/O are outside training time. "
              "Steady timing excludes the first complete iteration.", "",
              "Replay allocation and process peak RSS are different quantities. The 5m-row memory "
              "figure is an exact array-size projection, not a measured mature-reservoir RSS. "
              "This short 100k-capacity screen does not establish 24-hour speedup, large-history "
              "performance, poker strength or exploitability. Re-run on the intended GCP hardware."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for x, arm in enumerate(ARMS[1:]):
        ratios = [p["speedup"] for p in summary["arms"][arm]["pairs"]]
        axes[0].scatter(np.linspace(x - .1, x + .1, len(ratios)), ratios, alpha=.6)
        axes[0].plot([x - .2, x + .2], [np.median(ratios)] * 2, color="black")
    axes[0].axhline(1, color="gray", linestyle="--")
    axes[0].set(xticks=[0, 1], xticklabels=ARMS[1:], ylabel="Reference time / candidate time",
                title="Paired training speedup (line = median)")
    memory = [next(r["replay_allocated_bytes"] for r in records if r["arm"] == arm)
              / 1e6 for arm in ARMS]
    axes[1].bar(ARMS, memory)
    axes[1].set(ylabel="Allocated replay arrays (decimal MB)", title="Benchmark replay storage")
    fig.tight_layout()
    fig.savefig(output / "speed_and_replay_memory.png", dpi=180)
    plt.close(fig)


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "deep_cfr_poker").glob("*.py")) + sorted(Path(__file__).parent.glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def provenance():
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "source_sha256": source_hashes(),
            "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "python": sys.version,
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "open_spiel")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--threads", type=int, default=1)
    # Internal worker switches. Each fit gets independent process/RNG state.
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    parser.add_argument("--validation", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.repeats < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive threads/repeats and distinct seeds are required")
    if args.arm:
        run_worker(args)
        return 0
    output = args.output_dir or Path("outputs") / EXPERIMENT_NAME / time.strftime("%Y%m%d-%H%M%S")
    # Never mix old results with a new run, including an interrupted one.
    output.mkdir(parents=True, exist_ok=False)
    seeds = args.seeds[:1] if args.smoke else args.seeds
    repeats = 1 if args.smoke else args.repeats
    manifest = {**provenance(), "experiment": EXPERIMENT_NAME, "smoke": args.smoke,
                "seeds": seeds, "repeats": repeats, "threads": args.threads,
                "solver_config": solver_config(args.smoke), "atol": ATOL, "rtol": RTOL,
                "arms": ARMS, "execution": "fresh sequential subprocesses, rotating arm order"}
    write_json(output / "manifest.json", manifest)
    records, comparisons, schedule = [], [], []
    for seed_index, seed in enumerate(seeds):
        for repeat in range(repeats):
            shift = (seed_index + repeat) % len(ARMS)
            order = ARMS[shift:] + ARMS[:shift]
            schedule.append({"seed": seed, "repeat": repeat, "order": order})
            write_json(output / "schedule.json", schedule)
            with tempfile.TemporaryDirectory(prefix="sdcfr-equivalence-") as temporary:
                paths = {}
                for arm in order:
                    label = f"seed{seed}_repeat{repeat}_{arm}"
                    paths[arm] = Path(temporary) / f"{arm}.pt"
                    result_path = output / f"{label}.json"
                    command = [sys.executable, "-m", "experiments.fhp.exp1_sd_cfr_efficiency.run",
                               "--arm", arm, "--seed", str(seed), "--repeat", str(repeat),
                               "--threads", str(args.threads), "--result", str(result_path),
                               "--validation", str(paths[arm])]
                    if args.smoke:
                        command.append("--smoke")
                    print(f"Running {label}", flush=True)
                    environment = {**os.environ, "PYTHONHASHSEED": str(seed),
                                   "OMP_NUM_THREADS": str(args.threads),
                                   "MKL_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": "1"}
                    with (output / f"{label}.log").open("w") as log:
                        subprocess.run(command, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    record = json.loads(result_path.read_text())
                    if record["source_sha256"] != manifest["source_sha256"]:
                        raise RuntimeError("Source changed between benchmark arms; start a fresh run")
                    records.append(record)
                reference = torch.load(paths["reference"], weights_only=False)
                for arm in ARMS[1:]:
                    candidate = torch.load(paths[arm], weights_only=False)
                    check = validate_pair(reference, candidate)
                    comparisons.append({"seed": seed, "repeat": repeat, "arm": arm, **check})
                    print(f"  {arm}: passed={check['passed']}, identical={check['bit_identical']}", flush=True)
                    del candidate
                del reference
                # Only this experiment's temporary validation files are removed
                # by TemporaryDirectory; compact reports/fingerprints are kept.
            write_json(output / "comparisons.json", comparisons)
            write_json(output / "runs.json", records)
    summary = summarise(records, comparisons)
    write_json(output / "summary.json", summary)
    render_report(output, records, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {output}", flush=True)
    return 0 if summary["equivalence_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
