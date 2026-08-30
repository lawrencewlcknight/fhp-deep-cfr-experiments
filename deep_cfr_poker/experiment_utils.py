"""Shared experiment execution and export utilities."""

from __future__ import annotations

import csv
import gc
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pyspiel
import torch
from scipy import stats

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability

from .constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_SOLVER_BATCH_SIZE,
    KNOWN_GAME_VALUES_PLAYER_0,
)
from .game import load_game
from .seeding import set_seed
from .snapshots import package_full_checkpoint_filename, package_snapshot_filename
from .solver import DeepCFRSolver, SolveResult


_LOGGER = logging.getLogger(__name__)

DEFAULT_FINAL_WINDOW = 5


def _positive_batch_size_or_none(value) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def resolve_solver_batch_sizes(
    config: Mapping[str, object],
    *,
    default_batch_size: int = DEFAULT_SOLVER_BATCH_SIZE,
) -> tuple[int, int]:
    """Returns positive advantage / strategy minibatch sizes for solver runs.

    The solver treats ``None`` and ``0`` as "train on the whole replay buffer",
    which is too memory-hungry for experiment runs. If the strategy batch size
    is not explicitly positive, keep it aligned with the advantage batch size.
    """
    advantage = _positive_batch_size_or_none(config.get("batch_size_advantage"))
    if advantage is None:
        advantage = int(default_batch_size)
    strategy = _positive_batch_size_or_none(config.get("batch_size_strategy"))
    if strategy is None:
        strategy = advantage
    return advantage, strategy


def cleanup_training_memory() -> None:
    """Prompt Python / PyTorch to release memory after a completed training run."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def json_safe(value):
    """Converts common NumPy / Python values into JSON-serialisable values."""
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def make_solver(game, config) -> DeepCFRSolver:
    """Constructs a :class:`DeepCFRSolver` from an experiment config dict."""
    policy_network_train_every = int(
        config.get("policy_network_train_every", config["evaluation_interval"])
    )
    batch_size_advantage, batch_size_strategy = resolve_solver_batch_sizes(config)
    solver_kwargs = {
        "policy_network_type": str(config.get("policy_network_type", "mlp")),
        "policy_network_layers": tuple(config["policy_network_layers"]),
        "advantage_network_type": str(config.get("advantage_network_type", "mlp")),
        "advantage_network_layers": tuple(config["advantage_network_layers"]),
        "num_iterations": int(config["num_iterations"]),
        "num_traversals": int(config["num_traversals"]),
        "learning_rate": float(config["learning_rate"]),
        "learning_rate_schedule": str(config.get("learning_rate_schedule", "constant")),
        "learning_rate_end": (
            float(config["learning_rate_end"])
            if config.get("learning_rate_end") is not None
            else None
        ),
        "learning_rate_decay_rate": float(config.get("learning_rate_decay_rate", 0.5)),
        "learning_rate_decay_steps": (
            int(config["learning_rate_decay_steps"])
            if config.get("learning_rate_decay_steps") is not None
            else None
        ),
        "learning_rate_warmup_iterations": int(
            config.get("learning_rate_warmup_iterations", 0)
        ),
        "batch_size_advantage": batch_size_advantage,
        "batch_size_strategy": batch_size_strategy,
        "memory_capacity": int(config["memory_capacity"]),
        "reinitialize_advantage_networks": bool(
            config["reinitialize_advantage_networks"]
        ),
        "policy_network_train_steps": int(config["policy_network_train_steps"]),
        "advantage_network_train_steps": int(config["advantage_network_train_steps"]),
        "compute_exploitability": bool(config["compute_exploitability"]),
        "target_processing": str(config.get("target_processing", "none")),
        "target_clip_value": float(config.get("target_clip_value", 1.0)),
        "target_standardize_epsilon": float(
            config.get("target_standardize_epsilon", 1e-6)
        ),
        "advantage_replay_sampling": str(
            config.get("advantage_replay_sampling", "uniform")
        ),
        "replay_buffer_type": str(config.get("replay_buffer_type", "python")),
        "average_strategy_weighting": str(
            config.get("average_strategy_weighting", "linear")
        ),
        "priority_alpha": float(config.get("priority_alpha", 1.0)),
        "priority_epsilon": float(config.get("priority_epsilon", 1e-6)),
        "policy_network_train_every": policy_network_train_every,
        "evaluation_interval": int(config["evaluation_interval"]),
        "policy_training_mode": str(config.get("policy_training_mode", "intermittent")),
        "final_policy_network_train_steps": (
            int(config["final_policy_network_train_steps"])
            if config.get("final_policy_network_train_steps") is not None
            else None
        ),
    }
    execution_backend = str(config.get("execution_backend", "sequential")).lower()
    if execution_backend == "sequential":
        return DeepCFRSolver(game, **solver_kwargs)
    if execution_backend == "ray_parallel":
        from .parallel_solver import ParallelDeepCFRSolver

        return ParallelDeepCFRSolver(
            game,
            game_name=str(config["game_name"]),
            parallel_num_workers=int(config.get("parallel_num_workers", 3)),
            parallel_run_seed=int(config.get("parallel_run_seed", 0)),
            parallel_ray_address=config.get("parallel_ray_address"),
            parallel_log_to_driver=bool(config.get("parallel_log_to_driver", False)),
            parallel_worker_memory_capacity=(
                int(config["parallel_worker_memory_capacity"])
                if config.get("parallel_worker_memory_capacity") is not None
                else None
            ),
            parallel_ray_object_store_memory=(
                int(config["parallel_ray_object_store_memory"])
                if config.get("parallel_ray_object_store_memory") is not None
                else None
            ),
            **solver_kwargs,
        )
    raise ValueError(f"Unsupported execution_backend: {execution_backend!r}")


def game_value_player_0(config: Mapping[str, object]) -> float:
    """Returns the configured player-0 value target for the OpenSpiel game."""
    if config.get("average_policy_value_target") is not None:
        return float(config["average_policy_value_target"])
    game_name = str(config.get("game_name", ""))
    return float(
        KNOWN_GAME_VALUES_PLAYER_0.get(game_name, DEFAULT_AVERAGE_POLICY_VALUE_TARGET)
    )


def first_nodes_to_threshold(nodes, metric, threshold) -> float:
    nodes = np.asarray(nodes)
    metric = np.asarray(metric, dtype=np.float64)
    idx = np.where(np.isfinite(metric) & (metric <= threshold))[0]
    return float("nan") if len(idx) == 0 else float(nodes[idx[0]])


def first_time_to_threshold(times, metric, threshold) -> float:
    times = np.asarray(times)
    metric = np.asarray(metric, dtype=np.float64)
    idx = np.where(np.isfinite(metric) & (metric <= threshold))[0]
    return float("nan") if len(idx) == 0 else float(times[idx[0]])


def final_window_mean(values, window: int = DEFAULT_FINAL_WINDOW) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    window = max(1, int(window))
    return float(np.mean(values[-min(window, values.size):]))


def final_window_std(values, window: int = DEFAULT_FINAL_WINDOW) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    window_values = values[-min(max(1, int(window)), values.size):]
    return float(np.std(window_values, ddof=1)) if window_values.size > 1 else 0.0


def finite_min(values) -> float:
    """Return the minimum finite value, or NaN when none are available."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else float("nan")


def column_finite_mean(matrix: np.ndarray) -> np.ndarray:
    """Column means that stay quiet when a metric is intentionally unavailable."""
    matrix = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(matrix)
    counts = np.sum(finite, axis=0)
    sums = np.sum(np.where(finite, matrix, 0.0), axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def column_finite_sem(matrix: np.ndarray) -> np.ndarray:
    """Column standard errors, returning NaN for fewer than two observations."""
    matrix = np.asarray(matrix, dtype=np.float64)
    output = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for index in range(matrix.shape[1]):
        finite = matrix[:, index][np.isfinite(matrix[:, index])]
        if finite.size > 1:
            output[index] = np.std(finite, ddof=1) / np.sqrt(finite.size)
    return output


def normalised_auc(x_values, y_values) -> float:
    """Computes area under ``y`` over ``x`` divided by the x-range."""
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size < 2:
        return float("nan")
    x_range = x_values[-1] - x_values[0]
    if x_range == 0:
        return float("nan")
    return float(np.trapz(y_values, x_values) / x_range)


def run_single_seed(
    seed: int,
    config: dict,
    export_dir: Optional[Path] = None,
    save_final_checkpoint: bool = False,
    final_checkpoint_include_buffers: bool = True,
    policy_snapshot_iterations: Optional[Sequence[int]] = None,
    final_window: int = DEFAULT_FINAL_WINDOW,
) -> dict:
    """Runs one fixed-budget Deep CFR training run for a single seed."""
    batch_size_advantage, batch_size_strategy = resolve_solver_batch_sizes(config)
    config["batch_size_advantage"] = batch_size_advantage
    config["batch_size_strategy"] = batch_size_strategy
    if config.get("parallel_run_seed") is None:
        config["parallel_run_seed"] = int(seed)
    set_seed(seed)
    run_start = time.perf_counter()
    game = load_game(str(config["game_name"]))
    solver = None
    try:
        solver_init_start = time.perf_counter()
        solver = make_solver(game, config)
        solver_initialization_seconds = time.perf_counter() - solver_init_start

        snapshot_schedule = {
            int(iteration) for iteration in (policy_snapshot_iterations or ())
        }
        snapshot_rows = []
        if snapshot_schedule and export_dir is None:
            raise ValueError("export_dir is required when policy snapshots are enabled")

        def save_requested_snapshot(active_solver, completed_iteration: int) -> None:
            if completed_iteration not in snapshot_schedule:
                return
            snapshot_dir = Path(export_dir) / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / package_snapshot_filename(
                seed, completed_iteration
            )
            active_solver.save_policy_snapshot(
                snapshot_path,
                seed=seed,
                target_iteration=completed_iteration,
                stage_label=f"FHP training checkpoint {completed_iteration}",
                experiment_name=str(config["experiment_name"]),
                game_name=str(config["game_name"]),
                solver_config=dict(config),
            )
            snapshot_rows.append(
                {
                    "seed": int(seed),
                    "checkpoint_iteration": int(completed_iteration),
                    "nodes_touched": int(active_solver._nodes_touched),
                    "policy_training_events": int(
                        active_solver._policy_training_events
                    ),
                    "policy_gradient_steps": int(
                        active_solver._policy_gradient_steps
                    ),
                    "strategy_buffer_size": int(
                        len(active_solver._strategy_memories)
                    ),
                    "advantage_buffer_size_player_0": int(
                        len(active_solver._advantage_memories[0])
                    ),
                    "advantage_buffer_size_player_1": int(
                        len(active_solver._advantage_memories[1])
                    ),
                    "policy_snapshot": str(snapshot_path),
                }
            )
            manifest_path = snapshot_dir / f"seed_{int(seed)}_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(json_safe(snapshot_rows), handle, indent=2)

        solve_start = time.perf_counter()
        solve_result: SolveResult = solver.solve(
            post_iteration_callback=(
                save_requested_snapshot if snapshot_schedule else None
            )
        )
        elapsed_seconds = time.perf_counter() - solve_start

        captured_snapshots = {
            int(row["checkpoint_iteration"]) for row in snapshot_rows
        }
        missing_snapshots = sorted(snapshot_schedule - captured_snapshots)
        if missing_snapshots:
            raise RuntimeError(
                f"Training completed without requested snapshots at {missing_snapshots}"
            )

        convs = solve_result.nash_conv
        nodes_touched = solve_result.nodes_touched
        avg_policy_values = solve_result.average_policy_value
        diagnostics = solve_result.diagnostics

        exploitability_curve = np.asarray(convs, dtype=np.float64) / 2.0
        nodes_touched = np.asarray(nodes_touched, dtype=np.float64)
        avg_policy_values = np.asarray(avg_policy_values, dtype=np.float64)
        value_target = game_value_player_0(config)
        value_signed_error = avg_policy_values - value_target
        value_error = np.abs(value_signed_error)

        diagnostics = {k: np.asarray(v) for k, v in diagnostics.items()}
        iterations = diagnostics["iteration"].astype(int)
        wall_clock = diagnostics["wall_clock_seconds"].astype(float)

        if bool(config["compute_exploitability"]):
            final_policy = policy.tabular_policy_from_callable(
                game, solver.action_probabilities
            )
            final_nash_conv = exploitability.nash_conv(game, final_policy)
            final_policy_value = expected_game_score.policy_value(
                game.new_initial_state(), [final_policy] * game.num_players()
            )[0]
        else:
            final_nash_conv = float("nan")
            final_policy_value = float("nan")
        end_to_end_seconds = time.perf_counter() - run_start

        summary = {
            "seed": int(seed),
            "execution_backend": str(getattr(solver, "execution_backend", "sequential")),
            "parallel_num_workers": int(getattr(solver, "parallel_num_workers", 1)),
            "solver_initialization_seconds": float(solver_initialization_seconds),
            "elapsed_seconds": float(elapsed_seconds),
            "end_to_end_seconds": float(end_to_end_seconds),
            "traversal_collection_seconds": float(
                diagnostics.get("cumulative_traversal_collection_seconds", [float("nan")])[-1]
            ),
            "final_exploitability": float(exploitability_curve[-1]),
            "best_exploitability": finite_min(exploitability_curve),
            "final_window_mean_exploitability": final_window_mean(
                exploitability_curve, window=final_window
            ),
            "final_policy_value": float(final_policy_value),
            "final_policy_value_signed_error": float(
                final_policy_value - value_target
            ),
            "final_policy_value_error": float(
                abs(final_policy_value - value_target)
            ),
            "best_policy_value_error": finite_min(value_error),
            "final_nodes_touched": float(nodes_touched[-1]),
            "final_wall_clock_seconds": float(wall_clock[-1]),
            "nodes_to_exploitability_threshold": first_nodes_to_threshold(
                nodes_touched, exploitability_curve, config["exploitability_threshold"]
            ),
            "seconds_to_exploitability_threshold": first_time_to_threshold(
                wall_clock, exploitability_curve, config["exploitability_threshold"]
            ),
            "final_legal_action_mass_mean": float(
                diagnostics["legal_action_mass_mean"][-1]
            ),
            "final_legal_action_mass_min": float(diagnostics["legal_action_mass_min"][-1]),
            "final_policy_normalized_entropy_mean": float(
                diagnostics["policy_normalized_entropy_mean"][-1]
            ),
            "final_advantage_target_variance": float(
                diagnostics["advantage_target_variance"][-1]
            ),
            "final_policy_loss": float(diagnostics["policy_loss"][-1]),
            "final_policy_grad_norm": float(diagnostics["policy_grad_norm"][-1]),
            "final_advantage_grad_norm_player_0": float(
                diagnostics["advantage_grad_norm_player_0"][-1]
            ),
            "final_advantage_grad_norm_player_1": float(
                diagnostics["advantage_grad_norm_player_1"][-1]
            ),
            "final_policy_training_events": int(
                diagnostics.get("policy_training_events", [0])[-1]
            ),
            "final_policy_gradient_steps": int(
                diagnostics.get("policy_gradient_steps", [0])[-1]
            ),
            "final_learning_rate": float(
                diagnostics.get("learning_rate", [config["learning_rate"]])[-1]
            ),
            "final_nash_conv_recomputed": float(final_nash_conv),
        }

        result = {
            "seed": int(seed),
            "execution_backend": summary["execution_backend"],
            "parallel_num_workers": summary["parallel_num_workers"],
            "solver_initialization_seconds": float(solver_initialization_seconds),
            "elapsed_seconds": float(elapsed_seconds),
            "end_to_end_seconds": float(end_to_end_seconds),
            "iterations": iterations,
            "nodes_touched": nodes_touched,
            "wall_clock_seconds": wall_clock,
            "exploitability": exploitability_curve,
            "average_policy_value": avg_policy_values,
            "policy_value_signed_error": value_signed_error,
            "policy_value_error": value_error,
            "diagnostics": diagnostics,
            "summary": summary,
            "policy_snapshots": snapshot_rows,
        }

        if save_final_checkpoint and export_dir is not None:
            checkpoint_dir = Path(export_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            solver.save_full_model(
                checkpoint_dir
                / package_full_checkpoint_filename(seed, int(config["num_iterations"])),
                include_buffers=bool(final_checkpoint_include_buffers),
            )

        return result
    finally:
        if solver is not None:
            close = getattr(solver, "close", None)
            if callable(close):
                close()
        cleanup_training_memory()


def create_run_dir(output_root, experiment_name: str) -> Path:
    """Creates a timestamped output directory for one experiment run."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{experiment_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_run_logging(run_dir: Path, *, verbose: bool = False) -> None:
    """Configures logging to stdout *and* to ``run_dir/experiment.log``.

    Idempotent: calling this twice for the same run_dir replaces the file
    handler. Used by every experiment's CLI so log location is uniform.
    """
    import logging
    import sys

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any pre-existing FileHandlers pointing at our run_dir to make this
    # idempotent if a caller invokes us repeatedly.
    target = run_dir / "experiment.log"
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == target:
            root.removeHandler(handler)

    # Add a stream handler to stdout if none exists.
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in root.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(log_level)
        stream.setFormatter(logging.Formatter(log_format))
        root.addHandler(stream)

    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    root.addHandler(file_handler)


def write_experiment_metadata(
    run_dir: Path,
    *,
    config: dict,
    seeds: Sequence[int],
    completed_seeds: Optional[Sequence[int]] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Writes ``experiment_metadata.json`` in the run directory.

    The metadata schema is shared across experiments so the thesis can refer
    to a single description of what each output directory contains.
    """
    payload = {
        "experiment_config": json_safe(config),
        "seeds": list(map(int, seeds)),
        "completed_seeds": (
            list(map(int, completed_seeds))
            if completed_seeds is not None
            else None
        ),
        "game_value_player_0": game_value_player_0(config),
        "average_policy_value_target": game_value_player_0(config),
        "torch_version": torch.__version__,
        "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
    }
    if extra:
        payload.update(json_safe(extra))
    out = Path(run_dir) / "experiment_metadata.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def write_failed_seeds(run_dir: Path, failed_seeds: Sequence[dict]) -> Optional[Path]:
    """Writes ``failed_seeds.json`` if any seed errored. Returns the path or None."""
    if not failed_seeds:
        return None
    out = Path(run_dir) / "failed_seeds.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(json_safe(list(failed_seeds)), f, indent=2)
    return out


def write_dict_rows_csv(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    """Writes a list of dicts to a CSV file. Uses the first row's keys as columns.

    All experiments use the same writer to keep CSV layouts (especially
    quoting and line endings) consistent.
    """
    path = Path(path)
    if not rows:
        # Write an empty file with no header so downstream code sees something.
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_matrix_csv(
    matrix: np.ndarray,
    row_labels: Sequence[object],
    col_labels: Sequence[object],
    path: Path,
    *,
    index_name: str = "row",
) -> Path:
    """Writes a 2-D matrix to CSV with row/column labels.

    Equivalent to ``pandas.DataFrame(matrix, index=row_labels, columns=col_labels).to_csv(...)``
    but does not require pandas as a runtime dependency.
    """
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([index_name, *[str(label) for label in col_labels]])
        for row_label, row in zip(row_labels, np.asarray(matrix)):
            writer.writerow([str(row_label), *[float(value) for value in row]])
    return path


def summarise_numeric_fields(summary_rows: Sequence[dict]) -> dict:
    """Computes mean, std, and standard error for each numeric summary field."""
    if not summary_rows:
        return {}
    summary_fields = list(summary_rows[0].keys())
    seen_summary_fields = set(summary_fields)
    for row in summary_rows[1:]:
        for key in row.keys():
            if key not in seen_summary_fields:
                summary_fields.append(key)
                seen_summary_fields.add(key)
    summary_numeric: dict = {}
    for field_name in summary_fields:
        if field_name == "seed":
            continue
        try:
            vals = np.asarray(
                [row.get(field_name, float("nan")) for row in summary_rows],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            continue
        finite = vals[np.isfinite(vals)]
        if finite.size:
            summary_numeric[field_name] = {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "se": float(stats.sem(finite)) if finite.size > 1 else 0.0,
                "n_finite": int(finite.size),
            }
    return summary_numeric


def _pad_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    """Pads ``arr`` along axis 0 with NaNs (or zeros for ints) to ``length``."""
    if arr.shape[0] == length:
        return arr
    if arr.shape[0] > length:
        return arr[:length]
    pad_count = length - arr.shape[0]
    if np.issubdtype(arr.dtype, np.integer):
        pad = np.zeros((pad_count,) + arr.shape[1:], dtype=arr.dtype)
    else:
        pad = np.full((pad_count,) + arr.shape[1:], np.nan, dtype=np.float64)
        arr = arr.astype(np.float64, copy=False)
    return np.concatenate([arr, pad], axis=0)


def _stack_padded(arrays: Iterable[np.ndarray]) -> np.ndarray:
    """vstack with right-padding so ragged per-seed curves still produce a matrix."""
    arrays = [np.asarray(a) for a in arrays]
    if not arrays:
        return np.empty((0, 0))
    max_len = max(a.shape[0] for a in arrays)
    padded = [_pad_to_length(a, max_len) for a in arrays]
    return np.vstack(padded)


def export_results(
    results: Sequence[dict],
    run_dir,
    config: dict,
    seeds: Sequence[int],
    failed_seeds: Optional[Sequence[dict]] = None,
    write_multiseed_npz: bool = True,
) -> dict:
    """Exports CSV / JSON / NPZ artefacts for a completed multi-seed run.

    Ragged per-seed curves are right-padded with NaNs so a single failed seed
    does not abort the entire export. Failed seeds, if any, are listed in
    ``failed_seeds.json`` for traceability.
    """
    run_dir = Path(run_dir)

    if not results:
        raise ValueError("No completed seeds to export.")

    summary_rows = [result["summary"] for result in results]
    summary_fields = list(summary_rows[0].keys())
    seen_summary_fields = set(summary_fields)
    for row in summary_rows[1:]:
        for key in row.keys():
            if key not in seen_summary_fields:
                summary_fields.append(key)
                seen_summary_fields.add(key)

    summary_csv = run_dir / "seed_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "experiment_config": json_safe(config),
        "seeds": list(map(int, seeds)),
        "completed_seeds": [int(r["seed"]) for r in results],
        "game_value_player_0": game_value_player_0(config),
        "average_policy_value_target": game_value_player_0(config),
        "torch_version": torch.__version__,
        "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
    }
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if failed_seeds:
        with open(run_dir / "failed_seeds.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(list(failed_seeds)), f, indent=2)

    summary_numeric = summarise_numeric_fields(summary_rows)
    with open(run_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_numeric, f, indent=2)

    curve_csv = run_dir / "checkpoint_curves.csv"
    curve_fields = [
        "stage",
        "config_label",
        "variant_id",
        "variant_label",
        "execution_backend",
        "parallel_num_workers",
        "schedule",
        "schedule_label",
        "learning_rate_schedule",
        "target_processing",
        "target_clip_value",
        "advantage_replay_sampling",
        "average_strategy_weighting",
        "priority_alpha",
        "priority_epsilon",
        "policy_network_type",
        "advantage_network_type",
        "policy_training_mode",
        "reinitialize_advantage_networks",
        "seed",
        "iteration",
        "nodes_touched",
        "wall_clock_seconds",
        "traversal_collection_seconds",
        "cumulative_traversal_collection_seconds",
        "exploitability",
        "average_policy_value",
        "policy_value_signed_error",
        "policy_value_error",
        "policy_loss",
        "learning_rate",
        "strategy_buffer_size",
        "advantage_buffer_size_player_0",
        "advantage_buffer_size_player_1",
        "legal_action_mass_mean",
        "legal_action_mass_min",
        "policy_normalized_entropy_mean",
        "advantage_target_variance",
        "processed_advantage_target_variance_player_0",
        "processed_advantage_target_variance_player_1",
        "target_standardization_scale_player_0",
        "target_standardization_scale_player_1",
        "target_clip_fraction_player_0",
        "target_clip_fraction_player_1",
        "advantage_priority_effective_sample_size",
        "advantage_priority_effective_sample_size_player_0",
        "advantage_priority_effective_sample_size_player_1",
        "policy_grad_norm",
        "policy_training_events",
        "policy_gradient_steps",
        "iterations_since_policy_train",
        "policy_network_has_been_trained",
        "trained_policy_this_iteration",
        "advantage_grad_norm_player_0",
        "advantage_grad_norm_player_1",
        "policy_network_train_every",
        "final_policy_network_train_steps",
    ]
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields)
        writer.writeheader()
        for result in results:
            diag = result["diagnostics"]
            for i, iteration in enumerate(result["iterations"]):
                writer.writerow(
                    {
                        "stage": result.get("stage", ""),
                        "config_label": result.get("config_label", ""),
                        "variant_id": result.get("variant_id", ""),
                        "variant_label": result.get("variant_label", ""),
                        "execution_backend": result.get("execution_backend", ""),
                        "parallel_num_workers": result.get("parallel_num_workers", ""),
                        "schedule": result.get("schedule", ""),
                        "schedule_label": result.get("schedule_label", ""),
                        "learning_rate_schedule": result.get(
                            "learning_rate_schedule", ""
                        ),
                        "target_processing": result.get("target_processing", ""),
                        "target_clip_value": result.get("target_clip_value", ""),
                        "advantage_replay_sampling": result.get(
                            "advantage_replay_sampling", ""
                        ),
                        "average_strategy_weighting": result.get(
                            "average_strategy_weighting", ""
                        ),
                        "priority_alpha": result.get("priority_alpha", ""),
                        "priority_epsilon": result.get("priority_epsilon", ""),
                        "policy_network_type": result.get("policy_network_type", ""),
                        "advantage_network_type": result.get(
                            "advantage_network_type", ""
                        ),
                        "policy_training_mode": result.get("policy_training_mode", ""),
                        "reinitialize_advantage_networks": (
                            bool(result["reinitialize_advantage_networks"])
                            if "reinitialize_advantage_networks" in result
                            else ""
                        ),
                        "seed": result["seed"],
                        "iteration": int(iteration),
                        "nodes_touched": float(result["nodes_touched"][i]),
                        "wall_clock_seconds": float(result["wall_clock_seconds"][i]),
                        "traversal_collection_seconds": float(
                            diag.get(
                                "traversal_collection_seconds",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "cumulative_traversal_collection_seconds": float(
                            diag.get(
                                "cumulative_traversal_collection_seconds",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "exploitability": float(result["exploitability"][i]),
                        "average_policy_value": float(result["average_policy_value"][i]),
                        "policy_value_signed_error": float(
                            result["policy_value_signed_error"][i]
                        ),
                        "policy_value_error": float(result["policy_value_error"][i]),
                        "policy_loss": float(diag["policy_loss"][i]),
                        "learning_rate": float(
                            diag.get(
                                "learning_rate",
                                [config.get("learning_rate", float("nan"))]
                                * len(result["iterations"]),
                            )[i]
                        ),
                        "strategy_buffer_size": int(diag["strategy_buffer_size"][i]),
                        "advantage_buffer_size_player_0": int(
                            diag["advantage_buffer_size_player_0"][i]
                        ),
                        "advantage_buffer_size_player_1": int(
                            diag["advantage_buffer_size_player_1"][i]
                        ),
                        "legal_action_mass_mean": float(
                            diag["legal_action_mass_mean"][i]
                        ),
                        "legal_action_mass_min": float(
                            diag["legal_action_mass_min"][i]
                        ),
                        "policy_normalized_entropy_mean": float(
                            diag["policy_normalized_entropy_mean"][i]
                        ),
                        "advantage_target_variance": float(
                            diag["advantage_target_variance"][i]
                        ),
                        "processed_advantage_target_variance_player_0": float(
                            diag.get(
                                "processed_advantage_target_variance_player_0",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "processed_advantage_target_variance_player_1": float(
                            diag.get(
                                "processed_advantage_target_variance_player_1",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "target_standardization_scale_player_0": float(
                            diag.get(
                                "target_standardization_scale_player_0",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "target_standardization_scale_player_1": float(
                            diag.get(
                                "target_standardization_scale_player_1",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "target_clip_fraction_player_0": float(
                            diag.get(
                                "target_clip_fraction_player_0",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "target_clip_fraction_player_1": float(
                            diag.get(
                                "target_clip_fraction_player_1",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "advantage_priority_effective_sample_size": float(
                            diag.get(
                                "advantage_priority_effective_sample_size",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "advantage_priority_effective_sample_size_player_0": float(
                            diag.get(
                                "advantage_priority_effective_sample_size_player_0",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "advantage_priority_effective_sample_size_player_1": float(
                            diag.get(
                                "advantage_priority_effective_sample_size_player_1",
                                [float("nan")] * len(result["iterations"]),
                            )[i]
                        ),
                        "policy_grad_norm": float(diag["policy_grad_norm"][i]),
                        "policy_training_events": int(
                            diag.get("policy_training_events", [0] * len(result["iterations"]))[i]
                        ),
                        "policy_gradient_steps": int(
                            diag.get("policy_gradient_steps", [0] * len(result["iterations"]))[i]
                        ),
                        "iterations_since_policy_train": int(
                            diag.get("iterations_since_policy_train", [0] * len(result["iterations"]))[i]
                        ),
                        "policy_network_has_been_trained": bool(
                            diag.get(
                                "policy_network_has_been_trained",
                                [True] * len(result["iterations"]),
                            )[i]
                        ),
                        "trained_policy_this_iteration": bool(
                            diag.get(
                                "trained_policy_this_iteration",
                                [False] * len(result["iterations"]),
                            )[i]
                        ),
                        "advantage_grad_norm_player_0": float(
                            diag["advantage_grad_norm_player_0"][i]
                        ),
                        "advantage_grad_norm_player_1": float(
                            diag["advantage_grad_norm_player_1"][i]
                        ),
                        "policy_network_train_every": (
                            int(result["policy_network_train_every"])
                            if "policy_network_train_every" in result
                            else ""
                        ),
                        "final_policy_network_train_steps": (
                            int(result["final_policy_network_train_steps"])
                            if "final_policy_network_train_steps" in result
                            else ""
                        ),
                    }
                )

    iterations = results[0]["iterations"]
    exploitability_mat = _stack_padded(r["exploitability"] for r in results)
    value_error_mat = _stack_padded(r["policy_value_error"] for r in results)
    nodes_mat = _stack_padded(r["nodes_touched"] for r in results)
    wall_clock_mat = _stack_padded(r["wall_clock_seconds"] for r in results)
    avg_policy_value_mat = _stack_padded(r["average_policy_value"] for r in results)
    traversal_collection_mat = _stack_padded(
        r["diagnostics"].get("cumulative_traversal_collection_seconds", [])
        for r in results
    )

    if write_multiseed_npz:
        with np.errstate(invalid="ignore"):
            np.savez_compressed(
                run_dir / "multiseed_curves.npz",
                seeds=np.asarray([r["seed"] for r in results]),
                iterations=np.asarray(iterations),
                exploitability=exploitability_mat,
                policy_value_error=value_error_mat,
                average_policy_value=avg_policy_value_mat,
                nodes_touched=nodes_mat,
                wall_clock_seconds=wall_clock_mat,
                cumulative_traversal_collection_seconds=traversal_collection_mat,
                mean_exploitability=column_finite_mean(exploitability_mat),
                se_exploitability=column_finite_sem(exploitability_mat),
                mean_policy_value_error=column_finite_mean(value_error_mat),
                se_policy_value_error=column_finite_sem(value_error_mat),
            )

    return {
        "summary_csv": summary_csv,
        "curve_csv": curve_csv,
        "aggregate_summary": run_dir / "aggregate_summary.json",
        "metadata": run_dir / "experiment_metadata.json",
        "summary_numeric": summary_numeric,
    }
