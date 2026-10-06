"""Prespecified SD-CFR versus UCV-ESCHER duration cross-seed league.

The 24-hour cohort compares SD-CFR Experiment 5 with UCV-ESCHER Experiment 10.
The 48-hour cohort compares SD-CFR Experiment 6 with UCV-ESCHER Experiment 16,
the exact continuation of Experiment 10. SD-CFR is always deployed as its full
uniform historical trajectory mixture. There is no training, LBR, refitting or
archive thinning. The UCV repository is a separately pinned read-only loader.
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
import shutil
import sys
import time

import numpy as np

SEEDS = (0, 1, 2)
COHORT_HOURS = {"24h": (6, 12, 18, 24), "48h": tuple(range(6, 49, 6))}
# comparison id, kind, SD hours, UCV hours, pairs per cross-seed cell, primary.
COMPARISONS = {
    "24h": (
        ("primary_24h", "primary", 24, 24, 100_000, True),
        ("same_time_06h", "same_time", 6, 6, 50_000, False),
        ("same_time_12h", "same_time", 12, 12, 50_000, False),
        ("same_time_18h", "same_time", 18, 18, 50_000, False),
        ("nodes_sd06_ucv12", "approximate_nodes", 6, 12, 50_000, False),
        ("nodes_sd12_ucv24", "approximate_nodes", 12, 24, 50_000, False),
    ),
    "48h": (
        ("lineage_24h", "lineage_bridge", 24, 24, 50_000, False),
        ("same_time_30h", "same_time", 30, 30, 50_000, False),
        ("same_time_36h", "same_time", 36, 36, 50_000, False),
        ("same_time_42h", "same_time", 42, 42, 50_000, False),
        ("primary_48h", "primary", 48, 48, 100_000, True),
        ("nodes_sd24_ucv48", "approximate_nodes", 24, 48, 50_000, False),
    ),
}
SHARD_PAIRS = 5_000
PROTOCOL = "fhp_sd_ucv_duration_cross_seed_v2"
SD_SOURCE_COMMITS = {"24h": "e1118a1e1fd316d40f71c7c898868b5432873151",
                     "48h": "bb689252d6b322adb3de6102d095a49c3ed87250"}
UCV_SOURCE_COMMIT = "e66d4da515eb212e5026a965ac5c39c86144c901"
# Frozen Experiment 10 configuration, also used unchanged by Experiment 16.
UCV_CONFIG_SHA256 = "065734572e18f4597dd8f494ea39230c1f193b9d68aa775138c4a8a3e39eddad"
SOURCE_NAMES = {
    "24h": {"sd": ("exp5_sd_cfr_parallel_24h", "parallel_structured_uniform_sd_cfr", "SD-CFR Exp5"),
            "ucv": ("exp10_fhp_hand_board_features", "hand_board_cached_parallel_ucv_escher", "UCV-ESCHER Exp10")},
    "48h": {"sd": ("exp6_sd_cfr_parallel_48h", "parallel_structured_uniform_sd_cfr_48h", "SD-CFR Exp6"),
            "ucv": ("exp10_fhp_hand_board_features", "hand_board_cached_parallel_ucv_escher", "UCV-ESCHER Exp16")},
}


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


def validate_sources(sd_root, ucv_root, ucv_repo, *, cohort, ucv_source10_root=None):
    from deep_cfr_poker.game import load_fhp_game, FHP_GAME_PARAMETERS
    from deep_cfr_poker.sd_cfr_disk import DiskArchiveReader, sha256
    if cohort == "24h":
        from experiments.fhp.exp5_sd_cfr_parallel_24h import config as sd_config
    elif cohort == "48h":
        from experiments.fhp.exp6_sd_cfr_parallel_48h import config as sd_config
    else:
        raise ValueError(f"Unknown cohort: {cohort}")
    loader = setup_ucv(ucv_repo)
    from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
    game = load_fhp_game()
    hours = COHORT_HOURS[cohort]
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
            expected_commit = SD_SOURCE_COMMITS[cohort] if algorithm == "sd" else UCV_SOURCE_COMMIT
            expected_name, expected_id, label = SOURCE_NAMES[cohort][algorithm]
            expected_encoder = (sd_config.FEATURE_ENCODER_METADATA if algorithm == "sd"
                                else FHPHandBoardFeatureEncoder().metadata())
            if (manifest.get("repository_commit") != expected_commit
                    or manifest.get("experiment_name") != expected_name
                    or manifest.get("algorithm_id") != expected_id
                    or manifest.get("game", {}).get("parameters") != dict(FHP_GAME_PARAMETERS)
                    or manifest.get("feature_encoder") != expected_encoder
                    or manifest.get("reference_vm", {}).get("machine_type") != "n2-standard-16"):
                raise ValueError(f"Unexpected source experiment/game/encoder/VM: {worker}")
            seed = int(manifest["seed"])
            if seed not in SEEDS:
                raise ValueError("Unexpected training seed")
            if algorithm == "sd":
                if (digest(manifest["config"]) != digest(sd_config.solver_config())
                        or manifest.get("execution") != sd_config.execution_config(seed)
                        or success.get("checkpoints") != len(hours)):
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
            if sorted(int(r["checkpoint_target_hours"]) for r in rows) != list(hours):
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
                records.append(dict(algorithm=algorithm, source_experiment=label, cohort=cohort,
                                    seed=seed, training_hours=hour,
                                    active_seconds=elapsed, nodes_touched=int(row["nodes_touched"]),
                                    outer_iteration=int(row["outer_iteration"]), path=str(path), sha256=row["sha256"],
                                    source_commit=expected_commit,
                                    run_manifest_sha256=sha256(worker / "run_manifest.json"),
                                    checkpoint_manifest_sha256=sha256(worker / "checkpoint_manifest.json"),
                                    continuation_source_sha256="", source10_run_manifest_sha256="",
                                    source10_checkpoint_manifest_sha256="", source10_success_sha256=""))
            ordered = sorted((r for r in records if r["algorithm"] == algorithm and r["seed"] == seed),
                             key=lambda r: r["training_hours"])
            for earlier, later in zip(ordered, ordered[1:]):
                if any(later[key] <= earlier[key] for key in ("nodes_touched", "outer_iteration", "active_seconds")):
                    raise ValueError("Source training trajectory does not advance between checkpoints")
    validate_index(records, cohort)
    if cohort == "48h":
        validate_ucv_continuation(Path(ucv_root), Path(ucv_source10_root) if ucv_source10_root else None,
                                  records)
    return records


def validate_ucv_continuation(continued_root, source_root, records):
    """Prove Exp16 is the recorded Exp10 continuation and preserves 6--24h."""
    from deep_cfr_poker.sd_cfr_disk import sha256
    if source_root is None:
        raise ValueError("The 48h cohort requires Experiment 10 lineage metadata")
    continued = {(r["seed"], r["training_hours"]): r for r in records if r["algorithm"] == "ucv"}
    for seed in SEEDS:
        source_worker = source_root / "workers" / f"task_{seed:03d}_hand_board_cached_parallel_ucv_escher_seed_{seed}"
        destination = continued_root / "workers" / f"task_{seed:03d}_hand_board_cached_parallel_ucv_escher_seed_{seed}"
        source_manifest = read_json(source_worker / "run_manifest.json")
        source_rows = read_json(source_worker / "checkpoint_manifest.json")
        source_success = read_json(source_worker / "SUCCESS.json")
        lineage = read_json(destination / "continuation_source.json")
        provenance = dict(continuation_source_sha256=sha256(destination / "continuation_source.json"),
                          source10_run_manifest_sha256=sha256(source_worker / "run_manifest.json"),
                          source10_checkpoint_manifest_sha256=sha256(source_worker / "checkpoint_manifest.json"),
                          source10_success_sha256=sha256(source_worker / "SUCCESS.json"))
        for record in (r for r in records if r["algorithm"] == "ucv" and r["seed"] == seed):
            record.update(provenance)
        if (source_manifest.get("repository_commit") != UCV_SOURCE_COMMIT
                or source_manifest.get("experiment_name") != SOURCE_NAMES["24h"]["ucv"][0]
                or source_manifest.get("seed") != seed):
            raise ValueError("Invalid Experiment 10 lineage source")
        final = source_rows[-1]
        expected = dict(seed=seed, source_total_hours=24, total_hours=48,
                        source_commit=UCV_SOURCE_COMMIT,
                        source_state_path=final.get("training_state_path"),
                        source_state_sha256=final.get("training_state_sha256"),
                        source_summary_sha256=source_success.get("summary_sha256"))
        if any(lineage.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Invalid Experiment 10 continuation lineage: seed {seed}")
        if not lineage.get("source_worker", "").endswith(
                f"/exp10-features-20261001-161740/workers/{source_worker.name}"):
            raise ValueError("Experiment 16 used a different source cohort")
        for row in source_rows:
            hour = int(row["checkpoint_target_hours"])
            if hour > 24:
                continue
            record = continued[(seed, hour)]
            if (record["sha256"] != row["sha256"]
                    or record["nodes_touched"] != int(row["nodes_touched"])
                    or record["outer_iteration"] != int(row["outer_iteration"])):
                raise ValueError("Experiment 16 did not preserve Exp10 6--24h policies byte-for-byte")
        if continued[(seed, 48)]["outer_iteration"] <= continued[(seed, 24)]["outer_iteration"]:
            raise ValueError("Experiment 16 contains no training after 24h")


def validate_index(records, cohort):
    if sorted((r["algorithm"], r["seed"], r["training_hours"]) for r in records) != list(
            itertools.product(("sd", "ucv"), SEEDS, COHORT_HOURS[cohort])):
        raise ValueError("Expected both algorithms, three distinct seeds and the complete cohort schedule")


def build_tasks(records, implementation, *, cohort, stage="production"):
    validate_index(records, cohort)
    if stage not in {"production", "profile", "smoke"}:
        raise ValueError(stage)
    index = {(r["algorithm"], r["seed"], r["training_hours"]): r for r in records}
    tasks = []
    for comparison, (comparison_id, kind, sd_hour, ucv_hour, deals, primary) in enumerate(COMPARISONS[cohort]):
        total = deals if stage == "production" else (128 if stage == "profile" else 2)
        for a, b in itertools.product(SEEDS, repeat=2):
            cell = f"{comparison_id}_s{a}_u{b}"
            for shard, start in enumerate(range(0, total, SHARD_PAIRS)):
                # Independent streams across cells and shards; paired chance
                # within each seat-swapped pair. Probe streams never enter results.
                seed = int(np.random.SeedSequence([915039, comparison, a, b, shard,
                           {"production": 0, "profile": 1, "smoke": 2}[stage]]).generate_state(1)[0])
                tasks.append(dict(task_id=f"{cell}_{shard:03d}", cell_id=cell,
                                  comparison_id=comparison_id, kind=kind, primary=primary, cohort=cohort,
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
                                      policy_a_name=task["sd"]["source_experiment"],
                                      policy_b_name=task["ucv"]["source_experiment"]).to_dict()
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


def cluster_interval(matrix, *, draws=10_000, seed=590031, confidence=.95):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Bootstrap requires the complete 3x3 training-seed matrix")
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 3, (draws, 3)), rng.integers(0, 3, (draws, 3))
    means = matrix[a[:, :, None], b[:, None, :]].mean(axis=(1, 2))
    tail = (1 - confidence) / 2
    return tuple(float(x) for x in np.quantile(means, (tail, 1-tail)))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(results, output, *, cohort):
    from deep_cfr_poker.sd_cfr_disk import write_json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cells = []
    for cell in sorted({r["task"]["cell_id"] for r in results}):
        rows = [r for r in results if r["task"]["cell_id"] == cell]
        t = rows[0]["task"]
        cells.append(dict(cell_id=cell, comparison_id=t["comparison_id"], kind=t["kind"],
                          primary=t["primary"], cohort=cohort, sd_hours=t["sd"]["training_hours"],
                          ucv_hours=t["ucv"]["training_hours"], sd_seed=t["sd"]["seed"], ucv_seed=t["ucv"]["seed"],
                          sd_nodes=t["sd"]["nodes_touched"], ucv_nodes=t["ucv"]["nodes_touched"],
                          sd_active_seconds=t["sd"]["active_seconds"], ucv_active_seconds=t["ucv"]["active_seconds"],
                          relative_node_excess_sd=t["sd"]["nodes_touched"]/t["ucv"]["nodes_touched"]-1,
                          **pool_shards(rows)))
    summaries = []
    matrices = {}
    for comparison_id, kind, a, b, pairs, primary in COMPARISONS[cohort]:
        rows = [r for r in cells if r["comparison_id"] == comparison_id]
        if sorted((r["sd_seed"], r["ucv_seed"]) for r in rows) != list(itertools.product(SEEDS, repeat=2)):
            raise ValueError("Incomplete comparison; refusing aggregate")
        if any(r["duplicate_pairs"] != pairs for r in rows):
            raise ValueError("Incomplete production deal budget; probe results cannot be reported")
        matrix = np.empty((3, 3))
        for r in rows:
            matrix[r["sd_seed"], r["ucv_seed"]] = r["mean_mbb_per_hand"]
        lo, hi = cluster_interval(matrix)
        simultaneous_lo, simultaneous_hi = cluster_interval(matrix, confidence=.975)
        matrices[comparison_id] = matrix
        summaries.append(dict(comparison_id=comparison_id, cohort=cohort, kind=kind, primary=primary,
                              sd_hours=a, ucv_hours=b,
                              mean_mbb_per_hand=float(matrix.mean()), cluster_ci95_low=lo, cluster_ci95_high=hi,
                              family_ci97_5_low=simultaneous_lo if primary else "",
                              family_ci97_5_high=simultaneous_hi if primary else "",
                              conditional_mc_se_mbb=math.sqrt(sum(r["mc_se_mbb_per_hand"]**2 for r in rows))/9,
                              positive_cells=int((matrix > 0).sum()), cross_seed_cells=9,
                              sd_nodes_mean=float(np.mean([r["sd_nodes"] for r in rows])),
                              ucv_nodes_mean=float(np.mean([r["ucv_nodes"] for r in rows])),
                              node_excess_min=min(r["relative_node_excess_sd"] for r in rows),
                              node_excess_max=max(r["relative_node_excess_sd"] for r in rows),
                              duplicate_pairs=sum(r["duplicate_pairs"] for r in rows)))
    write_csv(output / "matchups.csv", cells)
    write_csv(output / "comparison_summary.csv", summaries)
    margin_change = None
    if cohort == "48h":
        change = matrices["primary_48h"] - matrices["lineage_24h"]
        lo, hi = cluster_interval(change, seed=590032)
        margin_change = dict(comparison_id="cross_play_margin_change_24h_to_48h",
                             mean_mbb_per_hand=float(change.mean()), cluster_ci95_low=lo,
                             cluster_ci95_high=hi,
                             interpretation=("Change in the SD-minus-UCV cross-play margin along the Exp6/Exp16 "
                                             "lineages; not a universal or causal learning-speed estimate."))
    write_json(output / "summary.json", dict(protocol=PROTOCOL, cohort=cohort, comparisons=summaries,
        cross_play_margin_change=margin_change, units="mbb/hand", positive_favours="SD-CFR",
        inference="Exploratory pointwise percentile bootstrap: 10,000 independent row/column training-seed resamples; three seeds per method, not nine independent runs.",
        primary_family_inference="The two prespecified primary endpoints also receive Bonferroni-compatible 97.5% cluster intervals; aggregation reports them together.",
        uncertainty="Cluster intervals condition on estimated matchup means; independent Monte Carlo SE is reported separately, not added to bootstrap variance.",
        caveats=["Head-to-head value is not exploitability or proof of convergence.",
                 "Historical selected cohorts with only three training seeds per method.",
                 "Active-time clocks differ: UCV excludes policy fitting; SD archive/deployment costs differ.",
                 "Node matches are approximate and algorithm node definitions need not imply equal computation."]))
    plots(cells, summaries, output, cohort=cohort)
    return summaries


def plots(cells, summaries, output, *, cohort):
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
           title=f"{cohort[:-1]}-hour saved policies: SD-CFR minus UCV-ESCHER")
    fig.colorbar(im, ax=ax, label="mbb/hand (positive favours SD-CFR)")
    fig.tight_layout(); fig.savefig(output / f"primary_{cohort}_heatmap.png", dpi=180); plt.close(fig)
    for node_view in (False, True):
        rows = sorted([r for r in summaries if ((r["kind"] == "approximate_nodes") == node_view)
                       and (node_view or r["kind"] in {"primary", "same_time", "lineage_bridge"})],
                      key=lambda r:r["sd_hours"])
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
            ax.set(xticks=sorted({r["sd_hours"] for r in rows}),
                   xlabel="Nominal active training hours (different clock exclusions)")
        name = f"approximate_nodes_{cohort}.png" if node_view else f"head_to_head_by_training_time_{cohort}.png"
        fig.tight_layout(); fig.savefig(output / name, dpi=180); plt.close(fig)


def aggregate(stage24, stage48, output):
    """Combine complete cohort reports without touching policy inputs."""
    from deep_cfr_poker.sd_cfr_disk import sha256, write_json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    rows = []
    for cohort, directory in (("24h", Path(stage24)), ("48h", Path(stage48))):
        success = read_json(directory / "SUCCESS.json")
        summary = read_json(directory / "summary.json")
        if success.get("status") != "complete" or summary.get("protocol") != PROTOCOL or summary.get("cohort") != cohort:
            raise ValueError(f"Incomplete or incompatible {cohort} stage")
        sources[cohort] = dict(success_sha256=sha256(directory / "SUCCESS.json"),
                               manifest_sha256=sha256(directory / "evaluation_manifest.json"))
        rows.extend(summary["comparisons"])
        for path in directory.glob("*.png"):
            shutil.copy2(path, output / path.name)
    primary = [r for r in rows if r["primary"]]
    if {r["comparison_id"] for r in primary} != {"primary_24h", "primary_48h"}:
        raise ValueError("Expected exactly the two prespecified primary endpoints")
    stage48_summary = read_json(Path(stage48) / "summary.json")
    write_csv(output / "comparison_summary.csv", rows)
    combined = dict(protocol=PROTOCOL, status="complete", units="mbb/hand",
                    positive_favours="SD-CFR", primary_endpoints=primary, comparisons=rows,
                    cross_play_margin_change=stage48_summary["cross_play_margin_change"],
                    family_inference=("Pointwise 95% intervals are exploratory. The two primary endpoints also "
                                      "have Bonferroni-compatible 97.5% training-seed cluster intervals."),
                    sources=sources,
                    caveats=stage48_summary["caveats"])
    write_json(output / "summary.json", combined)
    lines = ["# SD-CFR versus UCV-ESCHER duration evaluation", "",
             "Positive values favour SD-CFR. All values are mbb/hand.", "",
             "## Prespecified primary endpoints", "",
             "| Endpoint | Mean | Pointwise 95% CI | Family-compatible 97.5% CI |",
             "|---|---:|---:|---:|"]
    for row in sorted(primary, key=lambda r: r["sd_hours"]):
        lines.append(f"| {row['sd_hours']}h equal active time | {row['mean_mbb_per_hand']:.3f} | "
                     f"[{row['cluster_ci95_low']:.3f}, {row['cluster_ci95_high']:.3f}] | "
                     f"[{row['family_ci97_5_low']:.3f}, {row['family_ci97_5_high']:.3f}] |")
    change = combined["cross_play_margin_change"]
    lines += ["", "## Change along the 48-hour lineages", "",
              f"The SD-minus-UCV margin changed by **{change['mean_mbb_per_hand']:.3f} mbb/hand** "
              f"from 24h to 48h (95% CI [{change['cluster_ci95_low']:.3f}, "
              f"{change['cluster_ci95_high']:.3f}]).", "",
              change["interpretation"], "",
              "Equal active time is not equal node exposure. Consult `comparison_summary.csv` for the "
              "prespecified approximate-node comparisons and actual checkpoint node counts.", ""]
    (output / "analysis_summary.md").write_text("\n".join(lines))
    write_json(output / "SUCCESS.json", dict(status="complete", protocol=PROTOCOL,
               duplicate_pairs=sum(r["duplicate_pairs"] for r in rows), sources=sources))
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd-root", type=Path)
    parser.add_argument("--ucv-root", type=Path)
    parser.add_argument("--ucv-source10-root", type=Path,
                        help="Small Exp10 metadata tree used to prove Exp16 lineage")
    parser.add_argument("--ucv-repo", type=Path)
    parser.add_argument("--cohort", choices=tuple(COHORT_HOURS))
    parser.add_argument("--sd-source-uri", help="Original cloud run prefix, recorded for provenance")
    parser.add_argument("--ucv-source-uri", help="Original cloud run prefix, recorded for provenance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-24", type=Path)
    parser.add_argument("--stage-48", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--stage", choices=("smoke", "profile", "run", "aggregate"), default="run")
    args = parser.parse_args()
    if args.stage == "aggregate":
        if not args.stage_24 or not args.stage_48:
            parser.error("Aggregation requires --stage-24 and --stage-48")
        aggregate(args.stage_24, args.stage_48, args.output)
        return
    if not all((args.sd_root, args.ucv_root, args.ucv_repo, args.cohort)):
        parser.error("Evaluation requires --sd-root, --ucv-root, --ucv-repo and --cohort")
    if args.cohort == "48h" and not args.ucv_source10_root:
        parser.error("The 48h cohort requires --ucv-source10-root")
    if not 1 <= args.workers <= 8 or not 0 < args.max_hours <= 24:
        parser.error("Use 1–8 workers and an evaluation limit up to 24 hours")
    for source in (args.sd_root, args.ucv_root, args.ucv_repo):
        if args.output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(args.output.resolve()):
            parser.error("Output and source paths must be disjoint")
    import torch
    torch.set_num_threads(1)
    from deep_cfr_poker.sd_cfr_disk import write_json
    args.output.mkdir(parents=True, exist_ok=True)
    records = validate_sources(args.sd_root, args.ucv_root, args.ucv_repo, cohort=args.cohort,
                               ucv_source10_root=args.ucv_source10_root)
    implementation = implementation_digest(args.ucv_repo)
    tasks = build_tasks(records, implementation, cohort=args.cohort)
    environment = dict(python=platform.python_version(), **{k: version(k) for k in ("torch", "numpy", "open_spiel")})
    manifest = dict(protocol=PROTOCOL, cohort=args.cohort, sources=portable(records),
                    tasks_sha256=digest(portable(tasks)),
                    implementation_sha256=implementation, environment=environment,
                    duplicate_pairs=sum(t["num_deals"] for t in tasks), workers=args.workers,
                    source_uris=dict(sd=args.sd_source_uri, ucv=args.ucv_source_uri),
                    comparisons=[dict(comparison_id=i, kind=k, sd_hours=a, ucv_hours=b,
                                      duplicate_pairs_per_cell=n, primary=p)
                                 for i,k,a,b,n,p in COMPARISONS[args.cohort]], seed_layout="split",
                    policy_representations=dict(sd="full_uniform_historical_trajectory_mixture",
                                                ucv="saved_deployed_average_policy_network"))
    file = args.output / "evaluation_manifest.json"
    if file.exists() and read_json(file) != manifest:
        raise ValueError("Resume would mix different sources, code, runtime or protocol")
    write_json(file, manifest)
    write_csv(args.output / "checkpoint_index.csv", portable(records))
    if args.stage == "smoke":
        run_tasks(build_tasks(records, implementation, cohort=args.cohort, stage="smoke"),
                  args.output / "smoke_tasks",
                  workers=args.workers, ucv_repo=args.ucv_repo)
        write_json(args.output / "SMOKE_SUCCESS.json", dict(passed=True, tested_cells=54))
        return
    started = time.monotonic()
    # Re-profile after a VM restart; throughput on a previous host is not a
    # substitute for a real pilot here. These are not included in match results.
    probe_dir = args.output / "profile_tasks" / str(time.time_ns())
    probes = run_tasks(build_tasks(records, implementation, cohort=args.cohort, stage="profile"), probe_dir,
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
    report(results, args.output, cohort=args.cohort)
    write_json(args.output / "SUCCESS.json", dict(status="complete", protocol=PROTOCOL,
               duplicate_pairs=sum(t["num_deals"] for t in tasks), shards=len(results),
               evaluation_manifest_sha256=digest(manifest)))


if __name__ == "__main__":
    main()
