"""CLI runner for Experiment 1's FHP Deep CFR training run."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fhp_deep_cfr_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fhp_deep_cfr_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from tqdm import tqdm  # noqa: E402

from deep_cfr_poker.experiment_utils import (  # noqa: E402
    DEFAULT_FINAL_WINDOW,
    configure_run_logging,
    create_run_dir,
    export_results,
    json_safe,
    run_single_seed,
    write_dict_rows_csv,
)
from deep_cfr_poker.game import serialisable_game_definition  # noqa: E402

from .config import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    validate_config,
)


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.exp1")


def _str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_seeds(value: Optional[str]) -> list[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_config(args, *, base_config=None) -> dict:
    """Apply explicit CLI overrides to the approved defaults."""
    config = deepcopy(DEFAULT_CONFIG if base_config is None else base_config)
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "evaluation_interval": args.evaluation_interval,
        "checkpoint_schedule": parse_int_tuple(args.checkpoint_schedule),
        "policy_network_layers": parse_int_tuple(args.policy_network_layers),
        "advantage_network_layers": parse_int_tuple(args.advantage_network_layers),
        "learning_rate": args.learning_rate,
        "batch_size_advantage": args.batch_size_advantage,
        "batch_size_strategy": args.batch_size_strategy,
        "memory_capacity": args.memory_capacity,
        "policy_network_train_steps": args.policy_network_train_steps,
        "advantage_network_train_steps": args.advantage_network_train_steps,
        "policy_network_train_every": args.policy_network_train_every,
        "save_final_checkpoint": args.save_final_checkpoint,
        "final_checkpoint_include_buffers": args.final_checkpoint_include_buffers,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    validate_config(config)
    return config


def _write_run_manifest(
    path: Path,
    *,
    config: dict,
    seeds: Sequence[int],
    status: str,
    completed_seeds: Sequence[int] = (),
    failed_seeds: Sequence[dict] = (),
) -> None:
    payload = {
        "status": status,
        "game": serialisable_game_definition(),
        "experiment_config": json_safe(config),
        "seeds": [int(seed) for seed in seeds],
        "completed_seeds": [int(seed) for seed in completed_seeds],
        "failed_seeds": json_safe(list(failed_seeds)),
        "evaluation": {
            "exact_full_tree": False,
            "reason": "FHP full-tree enumeration is impractical",
            "recommended_follow_up": "sampled seat-averaged head-to-head play",
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_experiment(
    *,
    config: dict,
    seeds: Sequence[int],
    output_root: Path,
    final_window: int = DEFAULT_FINAL_WINDOW,
    verbose: bool = False,
) -> dict:
    """Run all configured seeds and export reproducible experiment artefacts."""
    validate_config(config)
    seeds = [int(seed) for seed in seeds]
    run_dir = create_run_dir(Path(output_root), str(config["experiment_name"]))
    configure_run_logging(run_dir, verbose=verbose)
    manifest_path = run_dir / "run_manifest.json"
    _write_run_manifest(
        manifest_path,
        config=config,
        seeds=seeds,
        status="running",
    )

    results = []
    failed = []
    snapshot_rows = []
    snapshot_manifest_csv = run_dir / "policy_snapshot_manifest.csv"

    for seed in tqdm(seeds, desc="FHP Deep CFR seeds"):
        _LOGGER.info("Starting seed %s", seed)
        try:
            result = run_single_seed(
                seed,
                deepcopy(config),
                export_dir=run_dir,
                save_final_checkpoint=bool(config["save_final_checkpoint"]),
                final_checkpoint_include_buffers=bool(
                    config["final_checkpoint_include_buffers"]
                ),
                policy_snapshot_iterations=config["checkpoint_schedule"],
                final_window=final_window,
            )
            results.append(result)
            snapshot_rows.extend(result["policy_snapshots"])
            write_dict_rows_csv(snapshot_rows, snapshot_manifest_csv)
        except Exception as exc:  # pragma: no cover - cloud/runtime failure path
            _LOGGER.exception("Seed %s failed: %s", seed, exc)
            failed.append(
                {
                    "seed": int(seed),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        _write_run_manifest(
            manifest_path,
            config=config,
            seeds=seeds,
            status="running",
            completed_seeds=[result["seed"] for result in results],
            failed_seeds=failed,
        )

    if not results:
        _write_run_manifest(
            manifest_path,
            config=config,
            seeds=seeds,
            status="failed",
            failed_seeds=failed,
        )
        return {"status": 1, "run_dir": run_dir, "failed_seeds": failed}

    export_info = export_results(
        results,
        run_dir,
        config,
        seeds,
        failed_seeds=failed or None,
    )
    final_status = "completed" if not failed else "completed_with_failures"
    _write_run_manifest(
        manifest_path,
        config=config,
        seeds=seeds,
        status=final_status,
        completed_seeds=[result["seed"] for result in results],
        failed_seeds=failed,
    )
    return {
        "status": 0,
        "run_dir": run_dir,
        "results": results,
        "failed_seeds": failed,
        "snapshot_manifest": snapshot_manifest_csv,
        "export_info": export_info,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Experiment 1: best validated Deep CFR configuration on FHP."
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--checkpoint-schedule", default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--advantage-network-layers", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--save-final-checkpoint", type=_str2bool, default=None)
    parser.add_argument(
        "--final-checkpoint-include-buffers", type=_str2bool, default=None
    )
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = build_config(args)
    seeds = parse_seeds(args.seeds)
    outcome = run_experiment(
        config=config,
        seeds=seeds,
        output_root=Path(args.output_root),
        final_window=args.final_window,
        verbose=args.verbose,
    )
    _LOGGER.info("Outputs: %s", Path(outcome["run_dir"]).resolve())
    return int(outcome["status"])


if __name__ == "__main__":
    sys.exit(main())
