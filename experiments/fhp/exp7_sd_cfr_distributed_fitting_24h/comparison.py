"""Prespecified saved-policy comparison: distributed fitting versus Experiment 5.

Reference policies are read-only. All matches use the shared duplicate-hand
evaluator; the nine cross-seed cells are never treated as nine replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from deep_cfr_poker.game import serialisable_game_definition
from deep_cfr_poker.sd_cfr_disk import sha256, write_json
from experiments.fhp.exp2_sd_cfr_24h.config import BASE_SEED, CROSSPLAY_DEALS
from experiments.fhp.exp2_sd_cfr_24h.report import summary
from experiments.fhp.exp2_sd_cfr_24h.train import write_csv
from experiments.fhp.exp5_sd_cfr_parallel_24h import config as baseline
from . import config

PROTOCOL = dict(
    primary_endpoint="24h same-seed two-seat EV: Experiment 7 minus Experiment 5",
    primary_inferential_units="three training-seed pairs",
    secondary_endpoints="6h/12h/18h paired EV and matched-active-time node throughput",
    cross_seed_matrix="all nine pairings per checkpoint; descriptive, not nine independent replicates",
    deals_per_pair=CROSSPLAY_DEALS, seats="both", seed_layout="split",
    common_random_numbers="same deal/action seed across checkpoint and seed pairings",
    uncertainty="pointwise training-seed t intervals; no multiplicity adjustment",
    comparison_type="retrospective saved Experiment 5 controls; not a contemporaneous randomised trial",
    exact_exploitability=False,
)


def normalised(value):
    return json.loads(json.dumps(value, sort_keys=True))


def validate_metadata(root, *, smoke=False, experiment=baseline):
    """Cheap fail-fast validation; full archive hashes are checked by checkpoint_index."""
    provenance, seeds = [], []
    for path in sorted(Path(root).glob("workers/*/run_manifest.json")):
        worker = path.parent
        manifest = json.loads(path.read_text())
        seed = manifest["seed"]
        if seed not in experiment.SEEDS or seed in seeds:
            raise ValueError("Unexpected/duplicate reference training seed")
        seeds.append(seed)
        expected = dict(
            experiment_name=experiment.EXPERIMENT_NAME, algorithm_id=experiment.ALGORITHM_ID,
            smoke=smoke, config=experiment.solver_config(smoke),
            execution=experiment.execution_config(seed), game=serialisable_game_definition(),
            feature_encoder=experiment.FEATURE_ENCODER_METADATA, reference_vm=experiment.REFERENCE_VM,
            checkpoints_hours=list(experiment.HOURS), strategy_weighting="uniform",
            torch_threads=1, interop_threads=1,
            checkpoint_boundary="first_completed_outer_iteration_crossing_threshold",
            time_excludes="checkpoint_serialization_reload_validation_and_upload",
            archive_capture_included_in_training_time=True,
            node_definition="calls_to_external_sampling_traversal_including_terminal_states",
            target_seconds=[.01 * (i + 1) for i in range(len(experiment.HOURS))] if smoke
                           else [h * 3600 for h in experiment.HOURS],
        )
        for key, value in expected.items():
            if manifest.get(key) != normalised(value):
                raise ValueError(f"Incompatible comparison source {path}: {key}")
        success = worker / "SUCCESS.json"
        if not success.is_file() or (worker / "FAILURE.json").exists():
            raise ValueError("Incomplete comparison training worker")
        completed = json.loads(success.read_text())
        rows_path = worker / "checkpoint_manifest.json"
        rows = json.loads(rows_path.read_text())
        if (completed.get("seed") != seed or completed.get("checkpoints") != len(experiment.HOURS)
                or [r["checkpoint_target_hours"] for r in rows] != list(experiment.HOURS)):
            raise ValueError("Incomplete/duplicate comparison checkpoint schedule")
        for row, seconds in zip(rows, expected["target_seconds"]):
            if (row["checkpoint_target_seconds"] != seconds
                    or not np.isfinite(row["actual_training_elapsed_seconds"])
                    or row["actual_training_elapsed_seconds"] < seconds
                    or row["nodes_touched"] <= 0 or row["outer_iteration"] <= 0):
                raise ValueError("Invalid comparison training budget/counters")
        provenance.append(dict(seed=seed, repository_commit=manifest.get("repository_commit"),
            torch_version=manifest.get("torch_version"), python_version=manifest.get("python_version"),
            numpy_version=manifest.get("numpy_version"), manifest_sha256=sha256(path),
            checkpoints_sha256=sha256(rows_path), success_sha256=sha256(success)))
    if sorted(seeds) != list((0,) if smoke else experiment.SEEDS):
        raise ValueError("Missing comparison training seeds")
    return provenance


def indexes(records, experiment, smoke):
    by_key = {(r["seed"], r["training_hours"]): r for r in records}
    seeds = (0,) if smoke else config.SEEDS
    if (len(by_key) != len(records) or set(by_key) != set(itertools.product(seeds, config.HOURS))
            or any(r["experiment"] != experiment.REPORT_ID for r in records)):
        raise ValueError("Wrong/incomplete comparison checkpoint index")
    return by_key


def extend_tasks(routine, candidate, reference, *, smoke=False):
    a, b = indexes(candidate, config, smoke), indexes(reference, baseline, smoke)
    seeds = (0,) if smoke else config.SEEDS
    hours = (config.HOURS[0], config.HOURS[-1]) if smoke else config.HOURS
    tasks = [dict(t) for t in routine]
    if not tasks:
        raise ValueError("Missing routine evaluation protocol")
    protocol = {k: tasks[0][k] for k in ("evaluation_protocol", "lbr_enabled", "lbr_backend", "lbr_device")}
    sources = [Path(__file__), Path(config.__file__), Path(baseline.__file__)]
    implementation = hashlib.sha256(json.dumps(
        [tasks[0]["implementation_sha256"], *[sha256(p) for p in sources]], sort_keys=True).encode()).hexdigest()
    for hour, seed_a, seed_b in itertools.product(hours, seeds, seeds):
        left, right = a[seed_a, hour], b[seed_b, hour]
        tasks.append(dict(protocol, task_id=f"cross_exp7_s{seed_a}_exp5_s{seed_b}_{hour}h",
            kind="cross_experiment", training_seed=seed_a, reference_seed=seed_b,
            same_seed=seed_a == seed_b, training_hours=hour,
            path_a=left["path"], sha_a=left["sha256"], path_b=right["path"], sha_b=right["sha256"],
            nodes_touched=left["nodes_touched"], reference_nodes_touched=right["nodes_touched"],
            candidate_experiment=config.REPORT_ID, reference_experiment=baseline.REPORT_ID,
            num_deals=2 if smoke else CROSSPLAY_DEALS, evaluation_seed=BASE_SEED + 3000000))
    for task in tasks:
        task["implementation_sha256"] = implementation
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate evaluation task")
    return tasks


def report(results, candidate, reference, output, *, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    a, b = indexes(candidate, config, smoke), indexes(reference, baseline, smoke)
    seeds = (0,) if smoke else config.SEEDS
    hours = (config.HOURS[0], config.HOURS[-1]) if smoke else config.HOURS
    rows, keys = [], set()
    for item in results:
        task = item["task"]
        if task["kind"] != "cross_experiment":
            continue
        key = (task["training_hours"], task["training_seed"], task["reference_seed"])
        hour, sa, sb = key
        if key in keys or key not in set(itertools.product(hours, seeds, seeds)):
            raise ValueError("Unexpected/duplicate cross-experiment result")
        if (task["sha_a"] != a[sa, hour]["sha256"] or task["sha_b"] != b[sb, hour]["sha256"]
                or task["num_deals"] != (2 if smoke else CROSSPLAY_DEALS)
                or item["result"]["num_deal_pairs"] != task["num_deals"]
                or not np.isfinite(item["result"]["mean_mbb_per_hand"])):
            raise ValueError("Incomplete/incompatible cross-experiment result")
        keys.add(key)
        rows.append(dict(**{k: v for k, v in task.items() if not k.startswith("path_")},
                         **{k: v for k, v in item["result"].items() if k not in task},
                         evaluation_seconds=item["elapsed_seconds"]))
    if keys != set(itertools.product(hours, seeds, seeds)):
        raise ValueError("Missing cross-experiment results")
    write_csv(output / "exp7_vs_exp5_by_pair.csv", rows)
    paired, descriptive, throughput, aggregates = [], [], [], []
    for hour in hours:
        selected = [r for r in rows if r["training_hours"] == hour]
        paired.append(dict(training_hours=hour, primary_endpoint=hour == 24,
            **summary([r["mean_mbb_per_hand"] for r in selected if r["training_seed"] == r["reference_seed"]])))
        descriptive.append(dict(training_hours=hour, pairings=len(selected), training_seeds_per_arm=len(seeds),
            mean_mbb_per_hand=float(np.mean([r["mean_mbb_per_hand"] for r in selected])),
            positive_pairings=sum(r["mean_mbb_per_hand"] > 0 for r in selected),
            interpretation="descriptive; correlated cells, not independent replicates"))
    for hour, seed in itertools.product(config.HOURS, seeds):
        left, right = a[seed, hour], b[seed, hour]
        if min(left["active_seconds"], right["active_seconds"], left["nodes_touched"], right["nodes_touched"]) <= 0:
            raise ValueError("Invalid throughput denominator")
        throughput.append(dict(training_hours=hour, seed=seed,
            candidate_nodes=left["nodes_touched"], reference_nodes=right["nodes_touched"],
            candidate_active_seconds=left["active_seconds"], reference_active_seconds=right["active_seconds"],
            node_gain=left["nodes_touched"] - right["nodes_touched"],
            rate_ratio=(left["nodes_touched"] / left["active_seconds"]) /
                       (right["nodes_touched"] / right["active_seconds"])))
    for hour in config.HOURS:
        selected = [r for r in throughput if r["training_hours"] == hour]
        for metric in ("candidate_nodes", "reference_nodes", "node_gain", "rate_ratio"):
            aggregates.append(dict(training_hours=hour, metric=metric, **summary([r[metric] for r in selected])))
    write_csv(output / "exp7_vs_exp5_paired_head_to_head.csv", paired)
    write_csv(output / "exp7_vs_exp5_cross_seed_descriptive.csv", descriptive)
    write_csv(output / "exp7_vs_exp5_throughput_by_seed.csv", throughput)
    write_csv(output / "exp7_vs_exp5_throughput_aggregate.csv", aggregates)
    write_json(output / "comparison_summary.json", dict(status="complete", smoke=smoke,
        protocol=PROTOCOL, primary_24h_paired_mbb_per_hand=paired[-1],
        rate_ratio_at_24h=next(r for r in aggregates if r["training_hours"] == 24 and r["metric"] == "rate_ratio"),
        candidate_checkpoints=candidate, reference_checkpoints=reference))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.errorbar(hours, [r["mean"] for r in paired], yerr=[r["se"] or 0 for r in paired],
                  marker="o", label=f"{len(seeds)} same-seed pairs: mean ± one SE")
    axis.plot(hours, [r["mean_mbb_per_hand"] for r in descriptive], "--", label="All cross-seed cells (descriptive)")
    axis.axhline(0, color="grey", linewidth=1)
    axis.set(xlabel="Nominal active training hours", ylabel="mbb/hand; positive favours distributed fitting")
    axis.legend()
    if smoke:
        axis.set_title("SMOKE TEST — nominal labels, not production performance")
    fig.tight_layout()
    fig.savefig(output / "exp7_vs_exp5_head_to_head.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    for metric, label in (("candidate_nodes", "Distributed fitting (Exp7)"), ("reference_nodes", "Central fitting (Exp5)")):
        selected = [r for r in aggregates if r["metric"] == metric]
        axis.errorbar([r["training_hours"] for r in selected], [r["mean"] / 1e6 for r in selected],
                      yerr=[(r["se"] or 0) / 1e6 for r in selected], marker="o", label=label)
    axis.set(xlabel="Nominal active training hours", ylabel="Training nodes (millions); mean ± one SE")
    axis.legend()
    if smoke:
        axis.set_title("SMOKE TEST — nominal labels, not production throughput")
    fig.tight_layout()
    fig.savefig(output / "exp7_vs_exp5_nodes_by_training_time.png", dpi=180)
    plt.close(fig)
    (output / "comparison_interpretation.txt").write_text(
        "Primary quality endpoint: 24h Experiment 7 versus Experiment 5 same-seed two-seat EV.\n"
        "Positive values favour distributed fitting. Three training-seed pairs are the inferential units.\n"
        "Pointwise t intervals are exploratory with n=3, unadjusted for secondary comparisons.\n"
        "All nine cross-seed matchups are descriptive, correlated cells, not nine replicates.\n"
        "Historical controls use the same specified learner, VM class and active-time accounting;\n"
        "machine contention, software revisions and numerical training paths may nevertheless differ.\n"
        "Throughput includes fitting, traversal, startup and archive capture, but excludes checkpoint overhead.\n"
        "Actual checkpoint overshoot is retained in the tables and in node-rate ratios.\n"
        "Head-to-head/rule-agent strength is not exploitability or proof of Nash convergence.\n"
        "No automatic model promotion is made. Source policy hashes and complete archives are retained.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-reference", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_metadata(args.validate_reference), indent=2))
