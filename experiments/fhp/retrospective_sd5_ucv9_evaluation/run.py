"""Prespecified SD-CFR Exp5 versus UCV-ESCHER Exp9 cross-seed league.

The only SD-CFR deployment representation used here is the full uniform
historical trajectory mixture. No LBR, fitting, archive thinning or retraining.
The UCV repository is a separately pinned, read-only loader dependency.
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
import sys
import time

import numpy as np

SEEDS = (0, 1, 2)
HOURS = (6, 12, 18, 24)
# kind, SD-CFR hours, UCV hours, duplicate-deal pairs per cross-seed cell.
COMPARISONS = (
    ("primary", 24, 24, 100_000),
    ("same_time", 6, 6, 50_000),
    ("same_time", 12, 12, 50_000),
    ("same_time", 18, 18, 50_000),
    ("approximate_nodes", 6, 12, 50_000),
    ("approximate_nodes", 12, 24, 50_000),
)
SHARD_PAIRS = 5_000
PROTOCOL = "fhp_sd5_ucv9_cross_seed_v1"
SD_SOURCE_COMMIT = "e1118a1e1fd316d40f71c7c898868b5432873151"
UCV_SOURCE_COMMIT = "3342ae096f81194f0d7a1c46fa5aec9fc26bb5ba"
# Frozen production Exp9 configuration; avoids importing its conflicting
# `experiments` package into this repository's Python namespace.
UCV_CONFIG_SHA256 = "fcd04bd63c0a15ad81ec24d327f09c35c8a3a2ed1e8976b64d944e8cb88ae9db"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items() if k not in {"path", "ucv_repo"}}
    if isinstance(value, list):
        return [portable(v) for v in value]
    return value


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Source path escapes root: {relative}")
    return path


def setup_ucv(repo):
    repo = Path(repo).resolve()
    if not (repo / "fhp_escher/checkpointing.py").is_file():
        raise ValueError("UCV loader repository is missing")
    # SD repository remains first: never install two overlapping `experiments`
    # or `fhp_evaluation` packages into the same environment.
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    import fhp_escher.checkpointing as loader
    if Path(loader.__file__).resolve() != repo / "fhp_escher/checkpointing.py":
        raise ValueError("A different UCV loader is already imported")
    return loader


def implementation_digest(ucv_repo):
    root = Path(__file__).resolve().parents[3]
    files = {"sd/" + str(p.relative_to(root)): p for directory in
             (root / "deep_cfr_poker", root / "fhp_evaluation", Path(__file__).parent)
             for p in directory.rglob("*.py")}
    for directory in ("fhp_escher", "vr_deep_cfr"):
        files.update({"ucv/" + str(p.relative_to(ucv_repo)): p
                      for p in (Path(ucv_repo) / directory).rglob("*.py")})
    from deep_cfr_poker.sd_cfr_disk import sha256
    return digest({k: sha256(p) for k, p in sorted(files.items())})


def validate_sources(sd_root, ucv_root, ucv_repo):
    from deep_cfr_poker.game import load_fhp_game, FHP_GAME_PARAMETERS
    from deep_cfr_poker.sd_cfr_disk import DiskArchiveReader, sha256
    from experiments.fhp.exp5_sd_cfr_parallel_24h import config as sd_config
    loader = setup_ucv(ucv_repo)
    game = load_fhp_game()
    records, verified_chunks = [], {}
    for algorithm, root in (("sd", Path(sd_root)), ("ucv", Path(ucv_root))):
        workers = sorted((root / "workers").glob("task_*"))
        if len(workers) != 3:
            raise ValueError(f"{algorithm}: expected exactly three successful workers")
        for worker in workers:
            contained(root, worker.resolve())
            manifest = read_json(worker / "run_manifest.json")
            success = read_json(worker / "SUCCESS.json")
            if (worker / "FAILURE.json").exists() or manifest.get("smoke") is not False:
                raise ValueError("Incomplete/smoke source worker")
            expected_commit = SD_SOURCE_COMMIT if algorithm == "sd" else UCV_SOURCE_COMMIT
            expected_name = "exp5_sd_cfr_parallel_24h" if algorithm == "sd" else "exp9_fhp_cached_parallel_24h"
            expected_id = "parallel_structured_uniform_sd_cfr" if algorithm == "sd" else "cached_parallel_structured_ucv_escher"
            if (manifest.get("repository_commit") != expected_commit
                    or manifest.get("experiment_name") != expected_name
                    or manifest.get("algorithm_id") != expected_id
                    or manifest.get("game", {}).get("parameters") != dict(FHP_GAME_PARAMETERS)
                    or manifest.get("feature_encoder") != sd_config.FEATURE_ENCODER_METADATA
                    or manifest.get("reference_vm", {}).get("machine_type") != "n2-standard-16"):
                raise ValueError(f"Unexpected source experiment/game/encoder/VM: {worker}")
            seed = int(manifest["seed"])
            if seed not in SEEDS:
                raise ValueError("Unexpected training seed")
            if algorithm == "sd":
                if (digest(manifest["config"]) != digest(sd_config.solver_config())
                        or manifest.get("execution") != sd_config.execution_config(seed)
                        or success.get("checkpoints") != 4):
                    raise ValueError("Unexpected SD-CFR training contract")
            else:
                if (digest(manifest["training_config"]) != UCV_CONFIG_SHA256
                        or success.get("status") != "complete"):
                    raise ValueError("Unexpected UCV configuration/completion")
                runtime = read_json(worker / "runtime_manifest.json")
                if (runtime.get("frozen_critic_target_cache") is not True
                        or runtime.get("traversal_execution") != "ray_parallel"):
                    raise ValueError("Unexpected UCV execution contract")
            rows = read_json(worker / "checkpoint_manifest.json")
            if sorted(r["checkpoint_target_hours"] for r in rows) != list(HOURS):
                raise ValueError("Missing/duplicate checkpoint schedule")
            for row in rows:
                path = contained(worker, row["path"])
                if sha256(path) != row["sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch: {path}")
                hour = int(row["checkpoint_target_hours"])
                elapsed = float(row["actual_training_elapsed_seconds"])
                if (not math.isfinite(elapsed) or elapsed < hour * 3600
                        or int(row["nodes_touched"]) <= 0 or int(row["outer_iteration"]) <= 0):
                    raise ValueError("Checkpoint precedes its declared training boundary")
                if algorithm == "sd":
                    # Verify each immutable shared chunk once, not once per
                    # checkpoint prefix. Reader still validates every shape.
                    c = read_json(path)
                    for chunk in c["chunks"]:
                        file = contained(path.parent, chunk["path"])
                        if file not in verified_chunks:
                            verified_chunks[file] = sha256(file)
                        if verified_chunks[file] != chunk["sha256"]:
                            raise ValueError("SD-CFR archive chunk hash mismatch")
                    reader = DiskArchiveReader(path, game, verify=False)
                    if (reader.count != row["outer_iteration"]
                            or digest(c["metadata"].get("solver_config")) != digest(manifest["config"])
                            or c["metadata"].get("feature_encoder") != manifest["feature_encoder"]
                            or c["metadata"].get("parallel_execution") != manifest["parallel_execution"]):
                        raise ValueError("SD-CFR archive metadata mismatch")
                else:
                    policy = loader.LoadedFHPPolicy(game, path)
                    payload = policy.checkpoint
                    expected = dict(seed=seed, experiment_name=expected_name, algorithm_id=expected_id,
                                    checkpoint_target_seconds=hour * 3600,
                                    nodes_touched=row["nodes_touched"], outer_iteration=row["outer_iteration"],
                                    feature_encoder=manifest["feature_encoder"],
                                    training_config=manifest["training_config"])
                    # Pickle preserves tuples and float-valued thresholds;
                    # JSON manifests use lists. Numeric int/float equivalence
                    # is legitimate metadata equality, unlike a string change.
                    if any(json.loads(json.dumps(payload.get(k))) != json.loads(json.dumps(v))
                           for k, v in expected.items()):
                        raise ValueError("UCV checkpoint metadata mismatch")
                records.append(dict(algorithm=algorithm, seed=seed, training_hours=hour,
                                    active_seconds=elapsed, nodes_touched=int(row["nodes_touched"]),
                                    outer_iteration=int(row["outer_iteration"]), path=str(path), sha256=row["sha256"],
                                    source_commit=expected_commit,
                                    run_manifest_sha256=sha256(worker / "run_manifest.json"),
                                    checkpoint_manifest_sha256=sha256(worker / "checkpoint_manifest.json")))
            ordered = sorted((r for r in records if r["algorithm"] == algorithm and r["seed"] == seed),
                             key=lambda r: r["training_hours"])
            for earlier, later in zip(ordered, ordered[1:]):
                if any(later[key] <= earlier[key] for key in ("nodes_touched", "outer_iteration", "active_seconds")):
                    raise ValueError("Source training trajectory does not advance between checkpoints")
    validate_index(records)
    return records


def validate_index(records):
    if sorted((r["algorithm"], r["seed"], r["training_hours"]) for r in records) != list(
            itertools.product(("sd", "ucv"), SEEDS, HOURS)):
        raise ValueError("Expected both algorithms, three distinct seeds, four checkpoints")


def build_tasks(records, implementation, *, stage="production"):
    validate_index(records)
    if stage not in {"production", "profile", "smoke"}:
        raise ValueError(stage)
    index = {(r["algorithm"], r["seed"], r["training_hours"]): r for r in records}
    tasks = []
    for comparison, (kind, sd_hour, ucv_hour, deals) in enumerate(COMPARISONS):
        total = deals if stage == "production" else (128 if stage == "profile" else 2)
        for a, b in itertools.product(SEEDS, repeat=2):
            cell = f"{kind}_sd{sd_hour:02d}_ucv{ucv_hour:02d}_s{a}_u{b}"
            for shard, start in enumerate(range(0, total, SHARD_PAIRS)):
                # Independent streams across cells and shards; paired chance
                # within each seat-swapped pair. Probe streams never enter results.
                seed = int(np.random.SeedSequence([915039, comparison, a, b, shard,
                           {"production": 0, "profile": 1, "smoke": 2}[stage]]).generate_state(1)[0])
                tasks.append(dict(task_id=f"{cell}_{shard:03d}", cell_id=cell, kind=kind,
                                  sd=index["sd", a, sd_hour], ucv=index["ucv", b, ucv_hour],
                                  num_deals=min(SHARD_PAIRS, total - start), evaluation_seed=seed,
                                  stage=stage, protocol=PROTOCOL, implementation=implementation))
    return tasks


_POLICIES = OrderedDict()


def initialise_worker(ucv_repo):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    setup_ucv(ucv_repo)


def loaded_policy(record, game):
    from deep_cfr_poker.sd_cfr_disk import DiskSampledPolicy, DiskArchiveReader, sha256
    key = record["algorithm"], record["sha256"]
    if key not in _POLICIES:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Checkpoint changed after validation")
        if record["algorithm"] == "sd":
            value = DiskSampledPolicy(DiskArchiveReader(record["path"], game, verify=False))
        else:
            from fhp_escher.checkpointing import LoadedFHPPolicy
            value = LoadedFHPPolicy(game, record["path"])
        _POLICIES[key] = value
    _POLICIES.move_to_end(key)
    while len(_POLICIES) > 4:
        _POLICIES.popitem(last=False)
    return _POLICIES[key]


def execute_task(task):
    from deep_cfr_poker.game import load_fhp_game
    from fhp_evaluation.duplicate import evaluate_duplicate_match
    started = time.perf_counter()
    game = load_fhp_game()
    a, b = loaded_policy(task["sd"], game), loaded_policy(task["ucv"], game)
    result = evaluate_duplicate_match(game, a, b, num_deals=task["num_deals"],
                                      seed=task["evaluation_seed"], seed_layout="split",
                                      policy_a_name="SD-CFR Exp5", policy_b_name="UCV-ESCHER Exp9").to_dict()
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


def run_tasks(tasks, output, *, workers, ucv_repo, deadline=None):
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
    # Only workers tasks are in flight, so timeout waits for at most one shard
    # per worker, not hours of queued work. Atomic files are uploaded periodically.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialise_worker, initargs=(str(ucv_repo),)) as pool:
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
    if any(not math.isfinite(r) or r <= 0 for r in rates.values()):
        raise ValueError("Invalid pilot timing")
    # Count cold checkpoint hydration in rates; double measured work and allow
    # an additional ten minutes. Full archives, not shortened pilot archives.
    seconds = 2 * sum(t["num_deals"] * rates[t["cell_id"]] for t in tasks) / workers + 600
    return dict(predicted_hours_with_2x_margin=seconds / 3600,
                workers=workers, remaining_duplicate_pairs=sum(t["num_deals"] for t in tasks),
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
                sd_as_player0_mean_mbb=10*sum(r["result"]["num_deal_pairs"] * r["result"]["policy_a_player0_mean_chips"] for r in rows)/n,
                sd_as_player1_mean_mbb=10*sum(r["result"]["num_deal_pairs"] * r["result"]["policy_a_player1_mean_chips"] for r in rows)/n)


def cluster_interval(matrix, *, draws=10_000, seed=590031):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Bootstrap requires the complete 3x3 training-seed matrix")
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 3, (draws, 3)), rng.integers(0, 3, (draws, 3))
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
    for cell in sorted({r["task"]["cell_id"] for r in results}):
        rows = [r for r in results if r["task"]["cell_id"] == cell]
        t = rows[0]["task"]
        cells.append(dict(cell_id=cell, kind=t["kind"], sd_hours=t["sd"]["training_hours"],
                          ucv_hours=t["ucv"]["training_hours"], sd_seed=t["sd"]["seed"], ucv_seed=t["ucv"]["seed"],
                          sd_nodes=t["sd"]["nodes_touched"], ucv_nodes=t["ucv"]["nodes_touched"],
                          sd_active_seconds=t["sd"]["active_seconds"], ucv_active_seconds=t["ucv"]["active_seconds"],
                          relative_node_excess_sd=t["sd"]["nodes_touched"]/t["ucv"]["nodes_touched"]-1,
                          **pool_shards(rows)))
    summaries = []
    for kind, a, b, pairs in COMPARISONS:
        rows = [r for r in cells if r["kind"] == kind and r["sd_hours"] == a and r["ucv_hours"] == b]
        if sorted((r["sd_seed"], r["ucv_seed"]) for r in rows) != list(itertools.product(SEEDS, repeat=2)):
            raise ValueError("Incomplete comparison; refusing aggregate")
        if any(r["duplicate_pairs"] != pairs for r in rows):
            raise ValueError("Incomplete production deal budget; probe results cannot be reported")
        matrix = np.empty((3, 3))
        for r in rows:
            matrix[r["sd_seed"], r["ucv_seed"]] = r["mean_mbb_per_hand"]
        lo, hi = cluster_interval(matrix)
        summaries.append(dict(kind=kind, sd_hours=a, ucv_hours=b,
                              mean_mbb_per_hand=float(matrix.mean()), cluster_ci95_low=lo, cluster_ci95_high=hi,
                              conditional_mc_se_mbb=math.sqrt(sum(r["mc_se_mbb_per_hand"]**2 for r in rows))/9,
                              positive_cells=int((matrix > 0).sum()), cross_seed_cells=9,
                              sd_nodes_mean=float(np.mean([r["sd_nodes"] for r in rows])),
                              ucv_nodes_mean=float(np.mean([r["ucv_nodes"] for r in rows])),
                              node_excess_min=min(r["relative_node_excess_sd"] for r in rows),
                              node_excess_max=max(r["relative_node_excess_sd"] for r in rows),
                              duplicate_pairs=sum(r["duplicate_pairs"] for r in rows)))
    write_csv(output / "matchups.csv", cells)
    write_csv(output / "comparison_summary.csv", summaries)
    write_json(output / "summary.json", dict(comparisons=summaries, units="mbb/hand", positive_favours="SD-CFR Exp5",
        inference="Exploratory pointwise percentile bootstrap: 10,000 independent row/column training-seed resamples; three seeds per method, not nine independent runs.",
        uncertainty="Cluster intervals condition on estimated matchup means; independent Monte Carlo SE is reported separately, not added to bootstrap variance.",
        caveats=["Head-to-head value is not exploitability or proof of convergence.",
                 "Historical selected cohorts; no multiplicity-adjusted confirmatory claim.",
                 "Active-time clocks differ: UCV excludes policy fitting; SD archive/deployment costs differ.",
                 "Node matches are approximate and algorithm node definitions need not imply equal computation."]))
    plots(cells, summaries, output)
    return summaries


def plots(cells, summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.empty((3, 3))
    for r in cells:
        if r["kind"] == "primary":
            matrix[r["sd_seed"], r["ucv_seed"]] = r["mean_mbb_per_hand"]
    fig, ax = plt.subplots(figsize=(6, 5))
    bound = max(float(np.abs(matrix).max()), .001)
    im = ax.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound)
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black",
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
    ax.set(xticks=SEEDS, yticks=SEEDS, xlabel="UCV-ESCHER training seed", ylabel="SD-CFR training seed",
           title="24-hour saved policies: SD-CFR minus UCV-ESCHER")
    fig.colorbar(im, ax=ax, label="mbb/hand (positive favours SD-CFR)")
    fig.tight_layout(); fig.savefig(output / "final_head_to_head_heatmap.png", dpi=180); plt.close(fig)
    for node_view in (False, True):
        rows = sorted([r for r in summaries if (r["kind"] == "approximate_nodes") == node_view], key=lambda r:r["sd_hours"])
        x = np.arange(len(rows)) if node_view else np.array([r["sd_hours"] for r in rows])
        means = [r["mean_mbb_per_hand"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, means, "o-", label="Mean over nine cross-seed matchups")
        ax.vlines(x, [r["cluster_ci95_low"] for r in rows], [r["cluster_ci95_high"] for r in rows], color="C0")
        ax.axhline(0, color="black", lw=.8)
        ax.set(ylabel="mbb/hand (positive favours SD-CFR)", title="Pointwise 95% training-seed cluster intervals")
        if node_view:
            ax.set_xticks(x, [f"SD {r['sd_hours']}h vs UCV {r['ucv_hours']}h\n{r['sd_nodes_mean']/1e6:.1f}m vs {r['ucv_nodes_mean']/1e6:.1f}m nodes" for r in rows])
            ax.set_xlabel("Approximate node matches (not equal compute)")
        else:
            ax.set(xticks=HOURS, xlabel="Nominal active training hours (different clock exclusions)")
        fig.tight_layout(); fig.savefig(output / ("approximate_nodes.png" if node_view else "head_to_head_by_training_time.png"), dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd-root", type=Path, required=True)
    parser.add_argument("--ucv-root", type=Path, required=True)
    parser.add_argument("--ucv-repo", type=Path, required=True)
    parser.add_argument("--sd-source-uri", help="Original cloud run prefix, recorded for provenance")
    parser.add_argument("--ucv-source-uri", help="Original cloud run prefix, recorded for provenance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--stage", choices=("smoke", "profile", "run"), default="run")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 0 < args.max_hours <= 12:
        parser.error("Use 1–8 workers and an evaluation limit up to 12 hours")
    for source in (args.sd_root, args.ucv_root, args.ucv_repo):
        if args.output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(args.output.resolve()):
            parser.error("Output and source paths must be disjoint")
    import torch
    torch.set_num_threads(1)
    from deep_cfr_poker.sd_cfr_disk import write_json
    args.output.mkdir(parents=True, exist_ok=True)
    records = validate_sources(args.sd_root, args.ucv_root, args.ucv_repo)
    implementation = implementation_digest(args.ucv_repo)
    tasks = build_tasks(records, implementation)
    environment = dict(python=platform.python_version(), **{k: version(k) for k in ("torch", "numpy", "open_spiel")})
    manifest = dict(protocol=PROTOCOL, sources=portable(records), tasks_sha256=digest(portable(tasks)),
                    implementation_sha256=implementation, environment=environment,
                    duplicate_pairs=sum(t["num_deals"] for t in tasks), workers=args.workers,
                    source_uris=dict(sd=args.sd_source_uri, ucv=args.ucv_source_uri),
                    comparisons=[dict(kind=k, sd_hours=a, ucv_hours=b, duplicate_pairs_per_cell=n)
                                 for k,a,b,n in COMPARISONS], seed_layout="split",
                    policy_representations=dict(sd="full_uniform_historical_trajectory_mixture",
                                                ucv="saved_deployed_average_policy_network"))
    file = args.output / "evaluation_manifest.json"
    if file.exists() and read_json(file) != manifest:
        raise ValueError("Resume would mix different sources, code, runtime or protocol")
    write_json(file, manifest)
    write_csv(args.output / "checkpoint_index.csv", portable(records))
    if args.stage == "smoke":
        run_tasks(build_tasks(records, implementation, stage="smoke"), args.output / "smoke_tasks",
                  workers=args.workers, ucv_repo=args.ucv_repo)
        write_json(args.output / "SMOKE_SUCCESS.json", dict(passed=True, tested_cells=54))
        return
    started = time.monotonic()
    # Re-profile after a VM restart; throughput on a previous host is not a
    # substitute for a real pilot here. These are not included in match results.
    probe_dir = args.output / "profile_tasks" / str(time.time_ns())
    probes = run_tasks(build_tasks(records, implementation, stage="profile"), probe_dir,
                       workers=args.workers, ucv_repo=args.ucv_repo)
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
    results = run_tasks(tasks, task_dir, workers=args.workers, ucv_repo=args.ucv_repo,
                        deadline=started + args.max_hours * 3600)
    report(results, args.output)
    write_json(args.output / "SUCCESS.json", dict(status="complete", protocol=PROTOCOL,
               duplicate_pairs=sum(t["num_deals"] for t in tasks), shards=len(results),
               evaluation_manifest_sha256=digest(manifest)))


if __name__ == "__main__":
    main()
