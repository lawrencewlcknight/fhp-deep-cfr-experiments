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
                                       sha256, write_json)
from deep_cfr_poker.sd_cfr_lbr import BatchedDiskBehaviouralPolicy, ExactSDCFRLocalBestResponsePolicy
from deep_cfr_poker.sd_cfr_lbr_audit import validate_queries
from fhp_evaluation.duplicate import evaluate_duplicate_match
from fhp_evaluation.lbr import LBRConfig
from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES, published_rule_agents
from . import config as default_experiment
from .config import (BASE_SEED, RULE_DEALS,
                     LBR_DEALS, LBR_ROLLOUTS, LBR_SHARD_DEALS, CROSSPLAY_DEALS)
from .train import write_csv


def checkpoint_index(root, *, smoke=False, experiment=default_experiment):
    records = []
    hours = None
    game = load_fhp_game()
    for manifest_path in sorted(Path(root).glob("workers/*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("experiment_name") != experiment.EXPERIMENT_NAME
                or manifest.get("algorithm_id") != experiment.ALGORITHM_ID):
            raise ValueError(f"Wrong SD-CFR experiment: {manifest_path}")
        if bool(manifest.get("smoke", False)) != smoke:
            raise ValueError("Smoke/production source mismatch")
        if hasattr(experiment, "execution_config"):
            if manifest.get("execution") != experiment.execution_config(int(manifest["seed"])):
                raise ValueError("Worker execution configuration differs from the experiment contract")
        worker = manifest_path.parent
        if not (worker / "SUCCESS.json").is_file() or (worker / "FAILURE.json").exists():
            raise ValueError("Incomplete training worker")
        rows = json.loads((worker / "checkpoint_manifest.json").read_text())
        worker_hours = (experiment.checkpoint_hours(manifest) if hasattr(experiment, "checkpoint_hours")
                        else experiment.HOURS)
        if hours is not None and hours != worker_hours:
            raise ValueError("Training workers have different checkpoint schedules")
        hours = worker_hours
        if sorted(float(row["checkpoint_target_hours"]) for row in rows) != list(hours):
            raise ValueError("Incomplete/duplicate checkpoint schedule")
        for row in rows:
            path = (worker / row["path"]).resolve()
            if not path.is_relative_to(worker.resolve()) or sha256(path) != row["sha256"]:
                raise ValueError("Checkpoint path/integrity mismatch")
            reader = DiskArchiveReader(path, game)
            if reader.contract.get("metadata", {}).get("feature_encoder") != experiment.FEATURE_ENCODER_METADATA:
                raise ValueError("Checkpoint encoder differs from the experiment contract")
            if hasattr(experiment, "execution_config"):
                execution = reader.contract["metadata"].get("parallel_execution", {})
                expected = experiment.execution_config(int(manifest["seed"]))
                if (execution.get("backend") != "ray_parallel_sd_cfr"
                        or execution.get("workers") != expected["parallel_num_workers"]
                        or execution.get("run_seed") != expected["parallel_run_seed"]
                        or execution != manifest.get("parallel_execution")):
                    raise ValueError("Checkpoint parallel execution metadata mismatch")
                if expected.get("distributed_fitting"):
                    fitting = execution.get("fitting", {})
                    if (fitting.get("backend") != "gloo_synchronous_allreduce"
                            or fitting.get("workers") != expected["parallel_num_workers"]
                            or fitting.get("global_batch_size") != manifest["config"]["batch_size_advantage"]
                            or fitting.get("updates_per_player") != manifest["config"]["advantage_network_train_steps"]
                            or fitting.get("target_normalization") != "global_minibatch"
                            or fitting.get("gradient_reduction") != "sum_example_weighted"):
                        raise ValueError("Checkpoint distributed fitting metadata mismatch")
            records.append(dict(experiment=experiment.REPORT_ID,
                                seed=int(manifest["seed"]), training_hours=int(row["checkpoint_target_hours"]),
                                active_seconds=float(row["actual_training_elapsed_seconds"]),
                                nodes_touched=int(row["nodes_touched"]), path=str(path),
                                sha256=row["sha256"], outer_iteration=int(row["outer_iteration"])))
    expected_seeds = (0,) if smoke else experiment.SEEDS
    if hours is None or sorted((r["seed"], r["training_hours"]) for r in records) != list(itertools.product(expected_seeds, hours)):
        raise ValueError(f"Expected exactly seeds {expected_seeds} at all checkpoints")
    return records


def make_tasks(records, *, smoke=False, lbr_device="cpu", include_lbr=True):
    if lbr_device not in ("cpu", "cuda"):
        raise ValueError("Unknown LBR device")
    tasks = []
    by_key = {(r["seed"], r["training_hours"]): r for r in records}
    seeds = (0,) if smoke else sorted({r["seed"] for r in records})
    all_hours = sorted({r["training_hours"] for r in records})
    hours = (all_hours[0], all_hours[-1]) if smoke else all_hours
    for seed in seeds:
        for hour in hours:
            row = by_key[seed, hour]
            base = dict(training_seed=seed, training_hours=hour, path_a=row["path"],
                        sha_a=row["sha256"], nodes_touched=row["nodes_touched"])
            for index, opponent in enumerate(PUBLISHED_AGENT_NAMES):
                tasks.append(dict(**base, task_id=f"rule_s{seed}_{hour}h_{opponent}",
                                  kind="rule", opponent=opponent, num_deals=2 if smoke else RULE_DEALS,
                                  evaluation_seed=BASE_SEED + 100000 + index))
            shard_count = (1 if smoke else LBR_DEALS // LBR_SHARD_DEALS) if include_lbr else 0
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
    sources = [Path(__file__), Path(__file__).with_name("report.py"), root / "deep_cfr_poker/sd_cfr_disk.py",
               root / "deep_cfr_poker/sd_cfr_lbr.py", root / "deep_cfr_poker/sd_cfr_lbr_audit.py",
               root / "deep_cfr_poker/fhp_features.py",
               root / "deep_cfr_poker/networks.py", root / "deep_cfr_poker/game.py",
               *sorted((root / "fhp_evaluation").glob("*.py"))]
    implementation = hashlib.sha256(json.dumps({str(p.relative_to(root)): sha256(p) for p in sources},
                                              sort_keys=True).encode()).hexdigest()
    for task in tasks:
        task["evaluation_protocol"] = "fhp_sdcfr_split_seeds_v1"
        task["lbr_enabled"] = include_lbr
        task["lbr_backend"] = "exact_batched_own_reach_v1" if include_lbr else None
        task["lbr_device"] = lbr_device if include_lbr else None
        task["implementation_sha256"] = implementation
    return tasks


_CACHE = OrderedDict()


def initialise_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def sd_policy(path, digest, *, lbr_device="cpu", need_mixture=False):
    key = (path, digest, lbr_device if need_mixture else None)
    if key not in _CACHE:
        # Full chunk hashes were validated once in the parent before dispatch.
        if sha256(path) != digest:
            raise ValueError("Checkpoint manifest changed during evaluation")
        game = load_fhp_game()
        reader = DiskArchiveReader(path, game, verify=False)
        mixture = BatchedDiskBehaviouralPolicy(reader, game, device=lbr_device) if need_mixture else None
        _CACHE[key] = (DiskSampledPolicy(reader), mixture)
    _CACHE.move_to_end(key)
    while len(_CACHE) > 2:
        _CACHE.popitem(last=False)
    return _CACHE[key]


def execute_task(task):
    started = time.perf_counter()
    task_started = started
    game = load_fhp_game()
    kind = task["kind"]
    target, mixture = sd_policy(task["path_a"], task["sha_a"],
                               lbr_device=task.get("lbr_device") or "cpu", need_mixture=kind == "lbr")
    if kind == "rule":
        opponent = published_rule_agents(game)[task["opponent"]]
        a, b, name_a, name_b = target, opponent, "sd_cfr", task["opponent"]
    elif kind == "lbr":
        responder = ExactSDCFRLocalBestResponsePolicy(game, mixture, config=LBRConfig(
            seed=task["lbr_seed"], preflop_rollout_samples=task["lbr_rollouts"]))
        # Query the exact behavioural mixture, but play the equivalent cheap
        # trajectory policy. The responder never sees the sampled model index.
        a, b, name_a, name_b = responder, target, "local_best_response", "sd_cfr"
    elif kind in {"temporal", "cross_experiment"}:
        opponent, _ = sd_policy(task["path_b"], task["sha_b"])
        a, b, name_a, name_b = target, opponent, "sd_cfr", (
            "earlier_sd_cfr" if kind == "temporal" else "exp5_central_fitting_sd_cfr")
    else:
        raise ValueError(kind)
    validation = None
    if kind == "lbr" and task.get("validate_lbr_backend"):
        validation = validate_queries(mixture.reader, game, batched=mixture)
        # Separate gate overhead from per-deal throughput, while retaining
        # total task duration and all full-archive production probes.
        started = time.perf_counter()
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
    output = dict(task=task, result=result, elapsed_seconds=time.perf_counter() - started)
    if validation is not None:
        output["lbr_backend_validation"] = validation
        output["total_elapsed_seconds_including_validation"] = time.perf_counter() - task_started
    return output


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
    for kind in sorted({t["kind"] for t in tasks}):
        candidates = [t for t in tasks if t["kind"] == kind
                      and t["training_hours"] == max(t["training_hours"] for t in tasks)]
        for seed in sorted({t["training_seed"] for t in candidates}):
            selected = [t for t in candidates if t["training_seed"] == seed]
            # Different rule agents induce substantially different hand lengths.
            for task in selected if kind in {"rule", "cross_experiment"} else selected[:1]:
                probes.append(dict(task, task_id="profile_" + task["task_id"],
                                   validate_lbr_backend=kind == "lbr",
                                   num_deals=1 if kind == "lbr" else 32))
    measurements = run_tasks(probes, Path(output) / "profile_tasks", workers=workers)
    validations = [r.get("lbr_backend_validation", {}) for r in measurements if r["task"]["kind"] == "lbr"]
    expected_validations = len({t["training_seed"] for t in tasks if t["kind"] == "lbr"})
    if len(validations) != expected_validations or not all(v.get("passed") for v in validations):
        raise RuntimeError("Missing/failed full-archive LBR numerical validation")
    rate = {kind: max(r["elapsed_seconds"] / r["task"]["num_deals"]
                     for r in measurements if r["task"]["kind"] == kind)
            for kind in {t["kind"] for t in tasks}}
    estimated_hours = 2 * sum(rate[t["kind"]] * t["num_deals"] for t in tasks) / workers / 3600
    report = dict(task_fingerprint=task_fingerprint(tasks), seconds_per_pair=rate,
                  estimated_elapsed_hours_with_2x_margin=estimated_hours, workers=workers,
                  allowed_elapsed_hours=max_hours, passed=estimated_hours <= max_hours,
                  lbr_enabled=bool(expected_validations),
                  lbr_status="included" if expected_validations else "omitted_by_configuration",
                  lbr_backend="exact_batched_own_reach_v1" if expected_validations else None,
                  lbr_validations=validations,
                  caveat="Pilot extrapolation, not a guaranteed completion time; no archive truncation")
    write_json(Path(output) / "evaluation_profile.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Estimated evaluation {estimated_hours:.1f}h exceeds {max_hours}h. "
                           "Training outputs are safe; review evaluation cost before increasing its budget.")
    return report


def main(*, experiment=default_experiment, comparison=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("profile", "run", "smoke"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=36)
    parser.add_argument("--lbr-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-lbr", action="store_true",
                        help="Retain rule and temporal matches only; leave all source policies intact")
    if comparison is not None:
        parser.add_argument("--reference-source", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 0 < args.max_hours <= 96:
        parser.error("Use 1..8 workers and an evaluation budget in (0, 96] hours")
    smoke = args.mode == "smoke"
    sd = checkpoint_index(args.source, smoke=smoke, experiment=experiment)
    tasks = make_tasks(sd, smoke=smoke, lbr_device=args.lbr_device, include_lbr=not args.skip_lbr)
    reference, source_provenance = [], {}
    if comparison is not None:
        source_provenance = dict(
            candidate=comparison.validate_metadata(args.source, smoke=smoke, experiment=experiment),
            reference=comparison.validate_metadata(args.reference_source, smoke=smoke))
        reference = checkpoint_index(args.reference_source, smoke=smoke, experiment=comparison.baseline)
        tasks = comparison.extend_tasks(tasks, sd, reference, smoke=smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "checkpoint_index.csv", sd)
    if reference:
        write_csv(args.output / "reference_checkpoint_index.csv", reference)
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
                    seeds=sorted({r["seed"] for r in sd}), hours=sorted({r["training_hours"] for r in sd}),
                    evaluated_hours=sorted({task["training_hours"] for task in tasks}),
                    evaluation_scope="standalone_sd_cfr",
                    task_fingerprint=task_fingerprint(tasks), tasks=len(tasks),
                    source_checkpoints=sd, lbr_enabled=not args.skip_lbr,
                    lbr_status="omitted_by_configuration" if args.skip_lbr else "included",
                    evaluated_metrics=sorted({task["kind"] for task in tasks}),
                    lbr_policy=None if args.skip_lbr else "exact_own_reach_historical_mixture",
                    lbr_backend=None if args.skip_lbr else "exact_batched_own_reach_v1",
                    lbr_device=None if args.skip_lbr else args.lbr_device,
                    play_policy="one_uniform_historical_network_per_player_per_hand",
                    uncertainty_unit="independent_training_seed; temporal matchups paired within seed",
                    evaluator_files={p.name: sha256(p) for p in (Path(__file__).parents[3] / "fhp_evaluation").glob("*.py")})
    if comparison is not None:
        manifest.update(evaluation_scope="exp7_vs_exp5_with_routine_evaluation",
                        reference_checkpoints=reference,
                        comparison_source_provenance=source_provenance,
                        comparison=comparison.PROTOCOL,
                        uncertainty_unit="training_seed_pairs; nine cross-seed cells are not nine independent replicates")
    write_json(args.output / "evaluation_manifest.json", manifest)
    results = run_tasks(tasks, args.output / "tasks", workers=args.workers)
    from .report import evaluation_report
    evaluation_report([r for r in results if r["task"]["kind"] != "cross_experiment"], sd, args.output,
                      smoke=smoke, include_lbr=not args.skip_lbr, has_comparison=comparison is not None)
    if comparison is not None:
        comparison.report(results, sd, reference, args.output, smoke=smoke)
    manifest["status"] = "complete"
    write_json(args.output / "evaluation_manifest.json", manifest)


if __name__ == "__main__":
    main()
