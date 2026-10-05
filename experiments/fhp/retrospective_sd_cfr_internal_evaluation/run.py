"""Prespecified internal league for the saved FHP SD-CFR experiments.

This is evaluation only. Every policy is the complete uniform historical
trajectory mixture saved by its source experiment. There is no retraining,
archive thinning, behavioural averaging, LBR, or automatic model promotion.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
from importlib.metadata import version
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np

SEEDS = (0, 1, 2)
SHARD_PAIRS = 5_000
PAIRS_PER_CELL = 50_000
PROTOCOL = "fhp_sd_cfr_internal_cross_seed_v1"

# The exact successful production cohorts. Source commits are validated before
# any matchup is run; a similarly named rerun cannot be substituted silently.
SOURCE_SPECS = OrderedDict((
    ("exp2", dict(
        run_id="sdcfr2-24h-20261002-003338",
        experiment_name="exp2_sd_cfr_24h",
        algorithm_id="optimised_uniform_sd_cfr",
        repository_commit="fbf3c398ab5804d35515effe3a29b5a6607d44b5",
        hours=(6, 12, 18, 24),
    )),
    ("exp3", dict(
        run_id="sdcfr3-24h-20261002-010643",
        experiment_name="exp3_sd_cfr_structured_24h",
        algorithm_id="structured_uniform_sd_cfr",
        repository_commit="0a6228671886446d56255fb249c3c6aa07ad7416",
        hours=(6, 12, 18, 24),
    )),
    ("exp4", dict(
        run_id="sdcfr4-vm16-20261002-095614",
        experiment_name="exp4_sd_cfr_structured_n2_standard16",
        algorithm_id="structured_uniform_sd_cfr",
        repository_commit="0a6228671886446d56255fb249c3c6aa07ad7416",
        hours=(6, 12, 18, 24),
    )),
    ("exp5", dict(
        run_id="sdcfr5-par8-20261002-102757",
        experiment_name="exp5_sd_cfr_parallel_24h",
        algorithm_id="parallel_structured_uniform_sd_cfr",
        repository_commit="e1118a1e1fd316d40f71c7c898868b5432873151",
        hours=(6, 12, 18, 24),
    )),
    ("exp6", dict(
        run_id="sdcfr6-48h-20261002-161544",
        experiment_name="exp6_sd_cfr_parallel_48h",
        algorithm_id="parallel_structured_uniform_sd_cfr_48h",
        repository_commit="bb689252d6b322adb3de6102d095a49c3ed87250",
        hours=(6, 12, 18, 24, 30, 36, 42, 48),
    )),
    ("exp7", dict(
        run_id="sdcfr7-distfit-20261003-172011",
        experiment_name="exp7_sd_cfr_distributed_fitting_24h",
        algorithm_id="distributed_fitting_structured_uniform_sd_cfr",
        repository_commit="dca6d2463792dc452d6aeb0854c181060c7a13a8",
        hours=(6, 12, 18, 24),
    )),
))

# id, scientific question, policy A, A hours, policy B, B hours. Reported value
# is always A minus B. All comparisons use every 3x3 training-seed pairing.
COMPARISONS = (
    ("representation_24h", "Does the structured representation improve SD-CFR?", "exp3", 24, "exp2", 24),
    ("parallel_traversal_24h", "Does parallel traversal preserve/improve quality at equal time?", "exp5", 24, "exp4", 24),
    ("repeatability_24h", "Does the repeated parallel configuration reproduce 24h strength?", "exp6", 24, "exp5", 24),
    ("distributed_equal_time", "Does distributed fitting improve quality at equal time?", "exp7", 24, "exp6", 24),
    ("distributed_node_bracket", "How does Exp7 24h compare with the next Exp6 node bracket?", "exp7", 24, "exp6", 30),
    ("long_horizon_36h", "Does Exp6 36h beat Exp7 24h?", "exp6", 36, "exp7", 24),
    ("long_horizon_42h", "Does Exp6 42h beat Exp7 24h?", "exp6", 42, "exp7", 24),
    ("long_horizon_48h", "Does Exp6 48h beat Exp7 24h?", "exp6", 48, "exp7", 24),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items() if k != "path"}
    if isinstance(value, list):
        return [portable(v) for v in value]
    return value


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Source path escapes root: {relative}")
    return path


def source_configs():
    from experiments.fhp.exp2_sd_cfr_24h import config as exp2
    from experiments.fhp.exp3_sd_cfr_structured_24h import config as exp3
    from experiments.fhp.exp4_sd_cfr_structured_n2_standard16 import config as exp4
    from experiments.fhp.exp5_sd_cfr_parallel_24h import config as exp5
    from experiments.fhp.exp6_sd_cfr_parallel_48h import config as exp6
    from experiments.fhp.exp7_sd_cfr_distributed_fitting_24h import config as exp7
    return dict(exp2=exp2, exp3=exp3, exp4=exp4, exp5=exp5, exp6=exp6, exp7=exp7)


def implementation_digest():
    root = Path(__file__).resolve().parents[3]
    files = {str(p.relative_to(root)): p for directory in
             (root / "deep_cfr_poker", root / "fhp_evaluation", Path(__file__).parent)
             for p in directory.rglob("*.py")}
    from deep_cfr_poker.sd_cfr_disk import sha256
    return digest({name: sha256(path) for name, path in sorted(files.items())})


def _normal(value):
    return json.loads(json.dumps(value, sort_keys=True))


def validate_sources(root):
    from deep_cfr_poker.game import FHP_GAME_PARAMETERS, load_fhp_game
    from deep_cfr_poker.sd_cfr_disk import DiskArchiveReader, sha256

    root = Path(root)
    game = load_fhp_game()
    configs = source_configs()
    records, verified_chunks = [], {}
    for experiment, spec in SOURCE_SPECS.items():
        source = root / experiment
        workers = sorted((source / "workers").glob("task_*"))
        if len(workers) != 3:
            raise ValueError(f"{experiment}: expected exactly three successful workers")
        seen_seeds = set()
        config = configs[experiment]
        for worker in workers:
            contained(source, worker.relative_to(source))
            manifest_path = worker / "run_manifest.json"
            checkpoint_manifest = worker / "checkpoint_manifest.json"
            success_path = worker / "SUCCESS.json"
            if not all(p.is_file() for p in (manifest_path, checkpoint_manifest, success_path)):
                raise ValueError(f"Incomplete source worker: {worker}")
            if (worker / "FAILURE.json").exists():
                raise ValueError(f"Failed source worker: {worker}")
            manifest, success = read_json(manifest_path), read_json(success_path)
            seed = int(manifest.get("seed", -1))
            if seed not in SEEDS or seed in seen_seeds or manifest.get("smoke") is not False:
                raise ValueError(f"Unexpected/duplicate/smoke source seed: {worker}")
            seen_seeds.add(seed)
            expected = dict(
                experiment_name=spec["experiment_name"],
                algorithm_id=spec["algorithm_id"],
                repository_commit=spec["repository_commit"],
                config=config.solver_config(),
                feature_encoder=config.FEATURE_ENCODER_METADATA,
                reference_vm=config.REFERENCE_VM,
                checkpoints_hours=list(spec["hours"]),
                strategy_weighting="uniform",
                torch_threads=1,
                interop_threads=1,
                checkpoint_boundary="first_completed_outer_iteration_crossing_threshold",
                time_excludes="checkpoint_serialization_reload_validation_and_upload",
                archive_capture_included_in_training_time=True,
                node_definition="calls_to_external_sampling_traversal_including_terminal_states",
            )
            if experiment in {"exp5", "exp6", "exp7"}:
                expected["execution"] = config.execution_config(seed)
            for key, value in expected.items():
                if _normal(manifest.get(key)) != _normal(value):
                    raise ValueError(f"Unexpected {key} in {manifest_path}")
            if manifest.get("game", {}).get("parameters") != dict(FHP_GAME_PARAMETERS):
                raise ValueError(f"Unexpected FHP game contract in {manifest_path}")
            rows = read_json(checkpoint_manifest)
            if (success.get("seed") != seed or success.get("checkpoints") != len(spec["hours"])
                    or [int(r["checkpoint_target_hours"]) for r in rows] != list(spec["hours"])):
                raise ValueError(f"Incomplete checkpoint schedule: {worker}")
            for row in rows:
                hour = int(row["checkpoint_target_hours"])
                path = contained(worker, row["path"])
                if sha256(path) != row["sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch: {path}")
                elapsed = float(row["actual_training_elapsed_seconds"])
                if (float(row["checkpoint_target_seconds"]) != hour * 3600
                        or not math.isfinite(elapsed) or elapsed < hour * 3600
                        or int(row["nodes_touched"]) <= 0 or int(row["outer_iteration"]) <= 0):
                    raise ValueError(f"Invalid checkpoint budget/counters: {path}")
                checkpoint = read_json(path)
                for chunk in checkpoint["chunks"]:
                    chunk_path = contained(path.parent, chunk["path"])
                    if chunk_path not in verified_chunks:
                        verified_chunks[chunk_path] = sha256(chunk_path)
                    if verified_chunks[chunk_path] != chunk["sha256"]:
                        raise ValueError(f"Archive chunk hash mismatch: {chunk_path}")
                reader = DiskArchiveReader(path, game, verify=False)
                metadata = checkpoint.get("metadata", {})
                if (reader.count != int(row["outer_iteration"])
                        or digest(metadata.get("solver_config")) != digest(manifest["config"])
                        or _normal(metadata.get("feature_encoder")) != _normal(manifest["feature_encoder"])
                        or ("parallel_execution" in manifest
                            and _normal(metadata.get("parallel_execution")) != _normal(manifest["parallel_execution"]))):
                    raise ValueError(f"Archive metadata mismatch: {path}")
                records.append(dict(
                    experiment=experiment, experiment_name=spec["experiment_name"],
                    algorithm_id=spec["algorithm_id"], seed=seed, training_hours=hour,
                    active_seconds=elapsed, nodes_touched=int(row["nodes_touched"]),
                    outer_iteration=int(row["outer_iteration"]), path=str(path), sha256=row["sha256"],
                    source_commit=spec["repository_commit"],
                    run_manifest_sha256=sha256(manifest_path),
                    checkpoint_manifest_sha256=sha256(checkpoint_manifest),
                ))
        if seen_seeds != set(SEEDS):
            raise ValueError(f"Missing source seed for {experiment}")
    validate_index(records)
    return records


def validate_index(records):
    actual = [(r["experiment"], r["seed"], r["training_hours"]) for r in records]
    expected = [(experiment, seed, hour) for experiment, spec in SOURCE_SPECS.items()
                for seed in SEEDS for hour in spec["hours"]]
    if len(actual) != len(set(actual)) or sorted(actual) != sorted(expected):
        raise ValueError("Expected six complete SD-CFR source cohorts and their frozen checkpoints")


def build_tasks(records, implementation, *, stage="production"):
    validate_index(records)
    if stage not in {"production", "profile", "smoke"}:
        raise ValueError(stage)
    index = {(r["experiment"], r["seed"], r["training_hours"]): r for r in records}
    total = PAIRS_PER_CELL if stage == "production" else (128 if stage == "profile" else 2)
    tasks = []
    for comparison_index, (comparison, question, a_exp, a_hour, b_exp, b_hour) in enumerate(COMPARISONS):
        for a_seed, b_seed in itertools.product(SEEDS, repeat=2):
            cell = f"{comparison}_a{a_seed}_b{b_seed}"
            for shard, start in enumerate(range(0, total, SHARD_PAIRS)):
                evaluation_seed = int(np.random.SeedSequence([
                    20261005, comparison_index, a_seed, b_seed, shard,
                    {"production": 0, "profile": 1, "smoke": 2}[stage],
                ]).generate_state(1)[0])
                tasks.append(dict(
                    task_id=f"{cell}_{shard:03d}", cell_id=cell, comparison=comparison,
                    question=question, policy_a=index[a_exp, a_seed, a_hour],
                    policy_b=index[b_exp, b_seed, b_hour],
                    num_deals=min(SHARD_PAIRS, total - start), evaluation_seed=evaluation_seed,
                    stage=stage, protocol=PROTOCOL, implementation=implementation,
                ))
    return tasks


_POLICIES = OrderedDict()


def initialise_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def loaded_policy(record, game):
    from deep_cfr_poker.sd_cfr_disk import DiskArchiveReader, DiskSampledPolicy, sha256
    key = record["sha256"]
    if key not in _POLICIES:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Checkpoint changed after validation")
        _POLICIES[key] = DiskSampledPolicy(DiskArchiveReader(record["path"], game, verify=False))
    _POLICIES.move_to_end(key)
    while len(_POLICIES) > 4:
        _POLICIES.popitem(last=False)
    return _POLICIES[key]


def execute_task(task):
    from deep_cfr_poker.game import load_fhp_game
    from fhp_evaluation.duplicate import evaluate_duplicate_match
    started = time.perf_counter()
    game = load_fhp_game()
    a = loaded_policy(task["policy_a"], game)
    b = loaded_policy(task["policy_b"], game)
    label_a = f"SD-CFR {task['policy_a']['experiment'].upper()}"
    label_b = f"SD-CFR {task['policy_b']['experiment'].upper()}"
    result = evaluate_duplicate_match(
        game, a, b, num_deals=task["num_deals"], seed=task["evaluation_seed"],
        seed_layout="split", policy_a_name=label_a, policy_b_name=label_b,
    ).to_dict()
    row = dict(task=portable(task), result=result, elapsed_seconds=time.perf_counter() - started)
    row["result_sha256"] = digest(row)
    return row


def validate_result(row, task):
    body = {k: v for k, v in row.items() if k != "result_sha256"}
    if (digest(body) != row.get("result_sha256") or row.get("task") != portable(task)
            or row["result"].get("num_deal_pairs") != task["num_deals"]
            or row["result"].get("num_games") != 2 * task["num_deals"]):
        raise ValueError("Corrupt or incompatible cached match shard")
    for key in ("mean_chips_per_hand", "std_chips_per_pair", "se_chips_per_hand",
                "mean_mbb_per_hand", "policy_a_player0_mean_chips", "policy_a_player1_mean_chips"):
        if not math.isfinite(row["result"][key]):
            raise ValueError("Non-finite match result")


def run_tasks(tasks, output, *, workers, deadline=None):
    from deep_cfr_poker.sd_cfr_disk import write_json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pending, results = [], []
    for task in tasks:
        file = output / (task["task_id"] + ".json")
        if file.exists():
            row = read_json(file)
            validate_result(row, task)
            results.append(row)
        else:
            pending.append(task)
    if not pending:
        return results
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Evaluation budget reached; completed shards retained for resume")
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialise_worker) as pool:
        iterator, futures = iter(pending), {}

        def submit_one():
            task = next(iterator, None)
            if task is not None:
                futures[pool.submit(execute_task, task)] = task

        for _ in range(workers):
            submit_one()
        while futures:
            done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                row = future.result()
                validate_result(row, task)
                write_json(output / (task["task_id"] + ".json"), row)
                results.append(row)
                if deadline is None or time.monotonic() < deadline:
                    submit_one()
                print(f"{task['stage']}: {len(results)}/{len(tasks)} shards complete", flush=True)
            if deadline is not None and time.monotonic() >= deadline and not futures:
                break
    if len(results) != len(tasks):
        raise TimeoutError("Evaluation budget reached; completed shards retained for resume")
    return results


def cost_estimate(probes, tasks, workers):
    rates = {r["task"]["cell_id"]: r["elapsed_seconds"] / r["task"]["num_deals"] for r in probes}
    if any(t["cell_id"] not in rates for t in tasks):
        raise ValueError("Missing real-checkpoint timing pilot")
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates.values()):
        raise ValueError("Invalid pilot timing")
    seconds = 2 * sum(t["num_deals"] * rates[t["cell_id"]] for t in tasks) / workers + 600
    return dict(predicted_hours_with_2x_margin=seconds / 3600, workers=workers,
                remaining_duplicate_pairs=sum(t["num_deals"] for t in tasks),
                per_cell_seconds_per_pair=rates, timing_is_estimate_not_guarantee=True)


def pool_shards(rows):
    n = sum(r["result"]["num_deal_pairs"] for r in rows)
    mean = sum(r["result"]["num_deal_pairs"] * r["result"]["mean_chips_per_hand"] for r in rows) / n
    ss = sum((r["result"]["num_deal_pairs"] - 1) * r["result"]["std_chips_per_pair"] ** 2
             + r["result"]["num_deal_pairs"] * (r["result"]["mean_chips_per_hand"] - mean) ** 2
             for r in rows)
    se = math.sqrt(ss / (n - 1) / n) * 10
    return dict(duplicate_pairs=n, hands=2*n, mean_mbb_per_hand=mean*10,
                mc_se_mbb_per_hand=se, mc_ci95_low=mean*10-1.96*se, mc_ci95_high=mean*10+1.96*se,
                policy_a_as_player0_mean_mbb=10*sum(r["result"]["num_deal_pairs"] * r["result"]["policy_a_player0_mean_chips"] for r in rows)/n,
                policy_a_as_player1_mean_mbb=10*sum(r["result"]["num_deal_pairs"] * r["result"]["policy_a_player1_mean_chips"] for r in rows)/n)


def cluster_interval(matrix, *, draws=10_000, seed=20261005):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Bootstrap requires the complete 3x3 training-seed matrix")
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, (draws, 3))
    b = rng.integers(0, 3, (draws, 3))
    means = matrix[a[:, :, None], b[:, None, :]].mean(axis=(1, 2))
    return tuple(float(x) for x in np.quantile(means, (.025, .975)))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(results, output):
    from deep_cfr_poker.sd_cfr_disk import write_json
    output = Path(output)
    cells = []
    for cell_id in sorted({r["task"]["cell_id"] for r in results}):
        rows = [r for r in results if r["task"]["cell_id"] == cell_id]
        task = rows[0]["task"]
        a, b = task["policy_a"], task["policy_b"]
        cells.append(dict(
            cell_id=cell_id, comparison=task["comparison"], question=task["question"],
            policy_a=a["experiment"], policy_a_hours=a["training_hours"], policy_a_seed=a["seed"],
            policy_b=b["experiment"], policy_b_hours=b["training_hours"], policy_b_seed=b["seed"],
            policy_a_nodes=a["nodes_touched"], policy_b_nodes=b["nodes_touched"],
            policy_a_active_seconds=a["active_seconds"], policy_b_active_seconds=b["active_seconds"],
            relative_node_excess_a=a["nodes_touched"] / b["nodes_touched"] - 1,
            **pool_shards(rows),
        ))
    summaries = []
    for comparison_index, (comparison, question, a_exp, a_hour, b_exp, b_hour) in enumerate(COMPARISONS):
        selected = [r for r in cells if r["comparison"] == comparison]
        if sorted((r["policy_a_seed"], r["policy_b_seed"]) for r in selected) != list(itertools.product(SEEDS, repeat=2)):
            raise ValueError("Incomplete comparison; refusing aggregate")
        if any(r["duplicate_pairs"] != PAIRS_PER_CELL for r in selected):
            raise ValueError("Incomplete production deal budget; probe results cannot be reported")
        matrix = np.empty((3, 3))
        for row in selected:
            matrix[row["policy_a_seed"], row["policy_b_seed"]] = row["mean_mbb_per_hand"]
        lo, hi = cluster_interval(matrix, seed=20261005 + comparison_index)
        diagonal = np.diag(matrix)
        diagonal_se = float(np.std(diagonal, ddof=1) / math.sqrt(3))
        summaries.append(dict(
            comparison=comparison, question=question,
            policy_a=a_exp, policy_a_hours=a_hour, policy_b=b_exp, policy_b_hours=b_hour,
            positive_favours=a_exp, mean_mbb_per_hand=float(matrix.mean()),
            cluster_ci95_low=lo, cluster_ci95_high=hi,
            conditional_mc_se_mbb=math.sqrt(sum(r["mc_se_mbb_per_hand"]**2 for r in selected))/9,
            positive_cells=int((matrix > 0).sum()), cross_seed_cells=9,
            same_seed_mean_mbb=float(diagonal.mean()), same_seed_se_mbb=diagonal_se,
            policy_a_nodes_mean=float(np.mean([r["policy_a_nodes"] for r in selected])),
            policy_b_nodes_mean=float(np.mean([r["policy_b_nodes"] for r in selected])),
            node_excess_min=min(r["relative_node_excess_a"] for r in selected),
            node_excess_max=max(r["relative_node_excess_a"] for r in selected),
            duplicate_pairs=sum(r["duplicate_pairs"] for r in selected),
        ))
    write_csv(output / "matchups.csv", cells)
    write_csv(output / "comparison_summary.csv", summaries)
    write_json(output / "summary.json", dict(
        status="complete", protocol=PROTOCOL, comparisons=summaries, units="mbb/hand",
        inference=("Exploratory pointwise percentile bootstrap with 10,000 independent row/column "
                   "training-seed resamples; the nine cross-seed cells are not nine independent runs."),
        uncertainty=("Cluster intervals condition on matchup means; independent aggregate Monte Carlo "
                     "SE is reported separately. Same-seed summaries are descriptive."),
        caveats=[
            "Head-to-head value is not exploitability or proof of convergence.",
            "Historical selected cohorts; no multiplicity-adjusted confirmatory claim.",
            "Approximate node brackets are neither exact node matches nor equal compute.",
            "A policy can perform well head-to-head while remaining exploitable by another strategy.",
        ],
    ))
    plots(cells, summaries, output)
    (output / "interpretation.txt").write_text(
        "Every estimate is policy A minus policy B; positive values favour policy A.\n"
        "All nine cross-seed cells are evaluated, with duplicate deals and both seats.\n"
        "Training-seed uncertainty resamples the three row and three column seeds independently.\n"
        "Finite-hand Monte Carlo error is reported separately from training-seed uncertainty.\n"
        "The complete uniform historical SD-CFR mixture is used for every saved policy.\n"
        "No LBR, exploitability estimate, retraining, archive thinning or automatic promotion occurs.\n"
    )
    return summaries


def plots(cells, summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["comparison"] for row in summaries]
    means = np.array([row["mean_mbb_per_hand"] for row in summaries])
    low = np.array([row["cluster_ci95_low"] for row in summaries])
    high = np.array([row["cluster_ci95_high"] for row in summaries])
    y = np.arange(len(summaries))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(means, y, xerr=np.vstack((means-low, high-means)), fmt="o", capsize=3)
    ax.axvline(0, color="black", lw=.8)
    ax.set(yticks=y, yticklabels=labels, xlabel="Policy A minus policy B (mbb/hand)",
           title="SD-CFR internal head-to-head comparisons")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "internal_comparison_forest.png", dpi=180)
    plt.close(fig)

    long_rows = [row for row in summaries if row["comparison"].startswith("long_horizon_")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hours = [row["policy_a_hours"] for row in long_rows]
    ax.plot(hours, [row["mean_mbb_per_hand"] for row in long_rows], "o-")
    ax.vlines(hours, [row["cluster_ci95_low"] for row in long_rows],
              [row["cluster_ci95_high"] for row in long_rows])
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="Experiment 6 active training hours", ylabel="Exp6 minus Exp7 24h (mbb/hand)",
           title="Long-horizon central fitting versus Exp7 distributed fitting")
    fig.tight_layout()
    fig.savefig(output / "exp6_long_horizon_vs_exp7.png", dpi=180)
    plt.close(fig)

    final_cells = [row for row in cells if row["comparison"] == "long_horizon_48h"]
    matrix = np.empty((3, 3))
    for row in final_cells:
        matrix[row["policy_a_seed"], row["policy_b_seed"]] = row["mean_mbb_per_hand"]
    bound = max(float(np.abs(matrix).max()), .001)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
    ax.set(xticks=SEEDS, yticks=SEEDS, xlabel="Experiment 7 training seed",
           ylabel="Experiment 6 training seed", title="Exp6 48h minus Exp7 24h")
    fig.colorbar(image, ax=ax, label="mbb/hand (positive favours Exp6)")
    fig.tight_layout()
    fig.savefig(output / "exp6_48h_vs_exp7_24h_heatmap.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-root", type=Path, required=True)
    parser.add_argument("--source-uri", action="append", default=[], metavar="EXPERIMENT=URI")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--stage", choices=("smoke", "profile", "run"), default="run")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 0 < args.max_hours <= 12:
        parser.error("Use 1–8 workers and an evaluation limit up to 12 hours")
    if (args.output.resolve().is_relative_to(args.sources_root.resolve())
            or args.sources_root.resolve().is_relative_to(args.output.resolve())):
        parser.error("Output and source paths must be disjoint")
    source_uris = {}
    for item in args.source_uri:
        if "=" not in item:
            parser.error("--source-uri must be EXPERIMENT=URI")
        key, value = item.split("=", 1)
        if key not in SOURCE_SPECS or key in source_uris:
            parser.error(f"Unexpected/duplicate source URI key: {key}")
        source_uris[key] = value
    if source_uris and set(source_uris) != set(SOURCE_SPECS):
        parser.error("Provide all six source URIs or none")

    import torch
    torch.set_num_threads(1)
    from deep_cfr_poker.sd_cfr_disk import write_json
    args.output.mkdir(parents=True, exist_ok=True)
    records = validate_sources(args.sources_root)
    implementation = implementation_digest()
    tasks = build_tasks(records, implementation)
    environment = dict(python=platform.python_version(),
                       **{name: version(name) for name in ("torch", "numpy", "open_spiel")})
    manifest = dict(
        protocol=PROTOCOL, sources=portable(records), source_uris=source_uris,
        tasks_sha256=digest(portable(tasks)), implementation_sha256=implementation,
        environment=environment, duplicate_pairs=sum(t["num_deals"] for t in tasks),
        workers=args.workers, seed_layout="split",
        policy_representation="complete_uniform_historical_trajectory_mixture",
        comparisons=[dict(comparison=c, question=q, policy_a=a, policy_a_hours=ah,
                          policy_b=b, policy_b_hours=bh, duplicate_pairs_per_cell=PAIRS_PER_CELL)
                     for c, q, a, ah, b, bh in COMPARISONS],
    )
    manifest_path = args.output / "evaluation_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Resume would mix different sources, code, runtime or protocol")
    write_json(manifest_path, manifest)
    write_csv(args.output / "checkpoint_index.csv", portable(records))
    if args.stage == "smoke":
        smoke = build_tasks(records, implementation, stage="smoke")
        run_tasks(smoke, args.output / "smoke_tasks", workers=args.workers)
        write_json(args.output / "SMOKE_SUCCESS.json", dict(passed=True, tested_cells=len(smoke)))
        return

    started = time.monotonic()
    probe_dir = args.output / "profile_tasks" / str(time.time_ns())
    probes = run_tasks(build_tasks(records, implementation, stage="profile"), probe_dir,
                       workers=args.workers)
    task_dir = args.output / "task_results"
    remaining = []
    for task in tasks:
        cached = task_dir / (task["task_id"] + ".json")
        if cached.exists():
            validate_result(read_json(cached), task)
        else:
            remaining.append(task)
    estimate = cost_estimate(probes, remaining, args.workers)
    estimate["limit_hours"] = args.max_hours
    estimate["passed"] = estimate["predicted_hours_with_2x_margin"] + (time.monotonic()-started)/3600 <= args.max_hours
    write_json(args.output / "timing_pilot.json", estimate)
    if not estimate["passed"]:
        raise RuntimeError("Timing pilot exceeds safety budget. No production matches launched; review timing_pilot.json")
    if args.stage == "profile":
        return
    results = run_tasks(tasks, task_dir, workers=args.workers,
                        deadline=started + args.max_hours * 3600)
    report(results, args.output)
    write_json(args.output / "SUCCESS.json", dict(
        status="complete", protocol=PROTOCOL, duplicate_pairs=sum(t["num_deals"] for t in tasks),
        hands=2*sum(t["num_deals"] for t in tasks), shards=len(results),
        evaluation_manifest_sha256=digest(manifest),
    ))


if __name__ == "__main__":
    main()
