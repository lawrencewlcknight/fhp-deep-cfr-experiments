"""Resumable sampled evaluation; exact mixture queries, never exact exploitability."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from deep_cfr_poker.game import load_fhp_game
from deep_cfr_poker.sd_cfr_disk import (DiskArchiveReader, DiskSampledPolicy,
                                       DiskBehaviouralPolicy, sha256, write_json)
from fhp_evaluation.duplicate import evaluate_duplicate_match
from fhp_evaluation.lbr import LBRConfig, LocalBestResponsePolicy
from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES, published_rule_agents
from . import config as default_experiment
from .config import (SEEDS, HOURS, BASE_SEED, RULE_DEALS,
                     LBR_DEALS, LBR_ROLLOUTS, LBR_SHARD_DEALS, CROSSPLAY_DEALS)
from .train import write_csv


def checkpoint_index(root, *, smoke=False, experiment=default_experiment):
    records = []
    game = load_fhp_game()
    for manifest_path in sorted(Path(root).glob("workers/*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("experiment_name") != experiment.EXPERIMENT_NAME
                or manifest.get("algorithm_id") != experiment.ALGORITHM_ID):
            raise ValueError(f"Wrong SD-CFR experiment: {manifest_path}")
        if bool(manifest.get("smoke", False)) != smoke:
            raise ValueError("Smoke/production source mismatch")
        worker = manifest_path.parent
        if not (worker / "SUCCESS.json").is_file() or (worker / "FAILURE.json").exists():
            raise ValueError("Incomplete training worker")
        rows = json.loads((worker / "checkpoint_manifest.json").read_text())
        if sorted(float(row["checkpoint_target_hours"]) for row in rows) != list(HOURS):
            raise ValueError("Incomplete/duplicate checkpoint schedule")
        for row in rows:
            path = (worker / row["path"]).resolve()
            if not path.is_relative_to(worker.resolve()) or sha256(path) != row["sha256"]:
                raise ValueError("Checkpoint path/integrity mismatch")
            reader = DiskArchiveReader(path, game)
            if reader.contract.get("metadata", {}).get("feature_encoder") != experiment.FEATURE_ENCODER_METADATA:
                raise ValueError("Checkpoint encoder differs from the experiment contract")
            records.append(dict(experiment=experiment.REPORT_ID,
                                seed=int(manifest["seed"]), training_hours=int(row["checkpoint_target_hours"]),
                                active_seconds=float(row["actual_training_elapsed_seconds"]),
                                nodes_touched=int(row["nodes_touched"]), path=str(path),
                                sha256=row["sha256"], outer_iteration=int(row["outer_iteration"])))
    expected_seeds = (0,) if smoke else SEEDS
    if sorted((r["seed"], r["training_hours"]) for r in records) != list(itertools.product(expected_seeds, HOURS)):
        raise ValueError(f"Expected exactly seeds {expected_seeds} at all four checkpoints")
    return records


def make_tasks(records, *, smoke=False):
    tasks = []
    by_key = {(r["seed"], r["training_hours"]): r for r in records}
    seeds = (0,) if smoke else SEEDS
    hours = (6, 24) if smoke else HOURS
    for seed in seeds:
        for hour in hours:
            row = by_key[seed, hour]
            base = dict(training_seed=seed, training_hours=hour, path_a=row["path"],
                        sha_a=row["sha256"], nodes_touched=row["nodes_touched"])
            for index, opponent in enumerate(PUBLISHED_AGENT_NAMES):
                tasks.append(dict(**base, task_id=f"rule_s{seed}_{hour}h_{opponent}",
                                  kind="rule", opponent=opponent, num_deals=2 if smoke else RULE_DEALS,
                                  evaluation_seed=BASE_SEED + 100000 + index))
            shard_count = 1 if smoke else LBR_DEALS // LBR_SHARD_DEALS
            for shard in range(shard_count):
                tasks.append(dict(**base, task_id=f"lbr_s{seed}_{hour}h_{shard:04d}", kind="lbr",
                                  shard_index=shard, num_deals=1 if smoke else LBR_SHARD_DEALS,
                                  evaluation_seed=BASE_SEED + 1000000 + shard,
                                  lbr_seed=BASE_SEED + 1500000,
                                  lbr_rollouts=8 if smoke else LBR_ROLLOUTS))
        for earlier, later in itertools.combinations(hours, 2):
            a, b = by_key[seed, later], by_key[seed, earlier]
            tasks.append(dict(task_id=f"temporal_s{seed}_{later}h_{earlier}h", kind="temporal",
                              training_seed=seed, training_hours=later, earlier_hours=earlier,
                              path_a=a["path"], sha_a=a["sha256"], path_b=b["path"], sha_b=b["sha256"],
                              num_deals=2 if smoke else CROSSPLAY_DEALS,
                              evaluation_seed=BASE_SEED + 2000000 + earlier * 1000 + later))
    root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), root / "deep_cfr_poker/sd_cfr_disk.py",
               root / "deep_cfr_poker/fhp_features.py",
               root / "deep_cfr_poker/networks.py", root / "deep_cfr_poker/game.py",
               *sorted((root / "fhp_evaluation").glob("*.py"))]
    implementation = hashlib.sha256(json.dumps({str(p.relative_to(root)): sha256(p) for p in sources},
                                              sort_keys=True).encode()).hexdigest()
    for task in tasks:
        task["evaluation_protocol"] = "fhp_sdcfr_split_seeds_v1"
        task["implementation_sha256"] = implementation
    return tasks


_CACHE = OrderedDict()


def initialise_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def sd_policy(path, digest):
    key = (path, digest)
    if key not in _CACHE:
        # Full chunk hashes were validated once in the parent before dispatch.
        if sha256(path) != digest:
            raise ValueError("Checkpoint manifest changed during evaluation")
        game = load_fhp_game()
        reader = DiskArchiveReader(path, game, verify=False)
        _CACHE[key] = (DiskSampledPolicy(reader), DiskBehaviouralPolicy(reader, game))
    _CACHE.move_to_end(key)
    while len(_CACHE) > 2:
        _CACHE.popitem(last=False)
    return _CACHE[key]


def execute_task(task):
    started = time.perf_counter()
    game = load_fhp_game()
    target, mixture = sd_policy(task["path_a"], task["sha_a"])
    kind = task["kind"]
    if kind == "rule":
        opponent = published_rule_agents(game)[task["opponent"]]
        a, b, name_a, name_b = target, opponent, "sd_cfr", task["opponent"]
    elif kind == "lbr":
        responder = LocalBestResponsePolicy(game, mixture, config=LBRConfig(
            seed=task["lbr_seed"], preflop_rollout_samples=task["lbr_rollouts"]))
        # Query the exact behavioural mixture, but play the equivalent cheap
        # trajectory policy. The responder never sees the sampled model index.
        a, b, name_a, name_b = responder, target, "local_best_response", "sd_cfr"
    elif kind == "temporal":
        opponent, _ = sd_policy(task["path_b"], task["sha_b"])
        a, b, name_a, name_b = target, opponent, "sd_cfr", "earlier_sd_cfr"
    else:
        raise ValueError(kind)
    result = evaluate_duplicate_match(game, a, b, num_deals=task["num_deals"],
                                      seed=task["evaluation_seed"], policy_a_name=name_a,
                                      policy_b_name=name_b, seed_layout="split").to_dict()
    # A one-pair cost probe has a mean but no estimable sampling variance.
    # Keep strict JSON; missing uncertainty is null, never zero.
    for key, value in result.items():
        if isinstance(value, float) and not np.isfinite(value):
            if task["num_deals"] == 1 and key.startswith(("std_", "se_", "ci95_")):
                result[key] = None
            else:
                raise RuntimeError(f"Non-finite evaluation result: {key}")
    return dict(task=task, result=result, elapsed_seconds=time.perf_counter() - started)


def task_fingerprint(tasks):
    # Paths can change between VMs; immutable policy hashes identify sources.
    canonical = [{k: v for k, v in task.items() if not k.startswith("path_")} for task in tasks]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def run_tasks(tasks, output, *, workers):
    # Different code/protocol/policy sets get independent namespaces. Old
    # shards remain recoverable, but can never be silently mixed into a rerun.
    output = Path(output) / task_fingerprint(tasks)[:16]
    output.mkdir(parents=True, exist_ok=True)
    pending, results = [], []
    for task in tasks:
        path = output / (task["task_id"] + ".json")
        if path.exists():
            row = json.loads(path.read_text())
            if task_fingerprint([row["task"]]) != task_fingerprint([task]):
                raise ValueError("Existing evaluation task has a different protocol or policy")
            results.append(row)
        else:
            pending.append(task)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialise_worker) as pool:
        futures = {pool.submit(execute_task, task): task for task in pending}
        for future in as_completed(futures):
            row = future.result()
            write_json(output / (row["task"]["task_id"] + ".json"), row)
            results.append(row)
            print(f"Evaluation tasks complete: {len(results)}/{len(tasks)}", flush=True)
    return sorted(results, key=lambda r: r["task"]["task_id"])


def profile(tasks, output, *, workers, max_hours):
    # Worst observed full-archive checkpoint across every training seed.
    probes = []
    for kind in ("rule", "lbr", "temporal"):
        candidates = [t for t in tasks if t["kind"] == kind and t["training_hours"] == 24]
        for seed in sorted({t["training_seed"] for t in candidates}):
            selected = [t for t in candidates if t["training_seed"] == seed]
            # Different rule agents induce substantially different hand lengths.
            for task in selected if kind == "rule" else selected[:1]:
                probes.append(dict(task, task_id="profile_" + task["task_id"],
                                   num_deals=1 if kind == "lbr" else 32))
    measurements = run_tasks(probes, Path(output) / "profile_tasks", workers=workers)
    rate = {kind: max(r["elapsed_seconds"] / r["task"]["num_deals"]
                     for r in measurements if r["task"]["kind"] == kind)
            for kind in {t["kind"] for t in tasks}}
    estimated_hours = 2 * sum(rate[t["kind"]] * t["num_deals"] for t in tasks) / workers / 3600
    report = dict(task_fingerprint=task_fingerprint(tasks), seconds_per_pair=rate,
                  estimated_elapsed_hours_with_2x_margin=estimated_hours, workers=workers,
                  allowed_elapsed_hours=max_hours, passed=estimated_hours <= max_hours,
                  caveat="Pilot extrapolation, not a guaranteed completion time; no archive truncation")
    write_json(Path(output) / "evaluation_profile.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Estimated evaluation {estimated_hours:.1f}h exceeds {max_hours}h. "
                           "Training outputs are safe; review evaluation cost before increasing its budget.")
    return report


def main(*, experiment=default_experiment):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("profile", "run", "smoke"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=36)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 0 < args.max_hours <= 96:
        parser.error("Use 1..8 workers and an evaluation budget in (0, 96] hours")
    smoke = args.mode == "smoke"
    sd = checkpoint_index(args.source, smoke=smoke, experiment=experiment)
    tasks = make_tasks(sd, smoke=smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "checkpoint_index.csv", sd)
    if args.mode == "profile":
        profile(tasks, args.output, workers=args.workers, max_hours=args.max_hours)
        return
    if not smoke:
        checked = json.loads((args.output / "evaluation_profile.json").read_text())
        if (not checked["passed"] or checked["task_fingerprint"] != task_fingerprint(tasks)
                or checked["workers"] != args.workers
                or checked["estimated_elapsed_hours_with_2x_margin"] > args.max_hours):
            raise ValueError("A matching successful cost profile is required before full evaluation")
    manifest = dict(status="running", smoke=smoke, exact_exploitability=False,
                    experiment_name=experiment.EXPERIMENT_NAME,
                    algorithm_id=experiment.ALGORITHM_ID,
                    feature_encoder=experiment.FEATURE_ENCODER_METADATA,
                    seeds=sorted({r["seed"] for r in sd}), hours=list(HOURS),
                    evaluated_hours=sorted({task["training_hours"] for task in tasks}),
                    evaluation_scope="standalone_sd_cfr",
                    task_fingerprint=task_fingerprint(tasks), tasks=len(tasks),
                    source_checkpoints=sd, lbr_policy="exact_own_reach_historical_mixture",
                    play_policy="one_uniform_historical_network_per_player_per_hand",
                    uncertainty_unit="independent_training_seed; temporal matchups paired within seed",
                    evaluator_files={p.name: sha256(p) for p in (Path(__file__).parents[3] / "fhp_evaluation").glob("*.py")})
    write_json(args.output / "evaluation_manifest.json", manifest)
    results = run_tasks(tasks, args.output / "tasks", workers=args.workers)
    from .report import evaluation_report
    evaluation_report(results, sd, args.output, smoke=smoke)
    manifest["status"] = "complete"
    write_json(args.output / "evaluation_manifest.json", manifest)


if __name__ == "__main__":
    main()
