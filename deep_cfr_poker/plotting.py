"""Plotting utilities for thesis-ready Deep CFR experiment figures.

The plotting layer is NaN-aware: ragged or partially-failed seeds (padded with
NaNs by :mod:`experiment_utils`) are reduced with ``np.nanmean`` and
``scipy.stats.sem(..., nan_policy='omit')`` so a single failed seed never
produces a blank or all-NaN figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (needs the backend set first)
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from .constants import (
    DEFAULT_ALGORITHM_VARIANT,
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
    DEFAULT_GAME_NAME,
)


_POKER_VARIANT_LABELS = {
    "FHP": "FHP",
}


def poker_variant_label(*, game_name: str = DEFAULT_GAME_NAME, poker_variant: str = None) -> str:
    """Returns the human-readable poker variant label used in figure titles."""
    if poker_variant:
        return str(poker_variant).strip()
    game_name = str(game_name or DEFAULT_GAME_NAME).strip()
    if game_name in _POKER_VARIANT_LABELS:
        return _POKER_VARIANT_LABELS[game_name]
    label = game_name.removesuffix("_poker").replace("_", " ").strip()
    return label.title() if label else "Poker"


def format_chart_title(
    title: str,
    *,
    algorithm_variant: str = DEFAULT_ALGORITHM_VARIANT,
    game_name: str = DEFAULT_GAME_NAME,
    poker_variant: str = None,
) -> str:
    """Formats a chart title as ``Algorithm - Poker Variant - Description``."""
    algorithm = str(algorithm_variant or DEFAULT_ALGORITHM_VARIANT).strip()
    poker = poker_variant_label(game_name=game_name, poker_variant=poker_variant)
    prefix = f"{algorithm} - {poker} - "
    title = str(title).strip()
    if title.startswith(prefix):
        return title
    return f"{prefix}{title}"


def set_chart_title(
    ax,
    title: str,
    *,
    algorithm_variant: str = DEFAULT_ALGORITHM_VARIANT,
    game_name: str = DEFAULT_GAME_NAME,
    poker_variant: str = None,
) -> None:
    """Applies the repository-wide figure-title naming convention."""
    ax.set_title(
        format_chart_title(
            title,
            algorithm_variant=algorithm_variant,
            game_name=game_name,
            poker_variant=poker_variant,
        )
    )


def _pad_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    if arr.shape[0] == length:
        return arr.astype(np.float64, copy=False)
    if arr.shape[0] > length:
        return arr[:length].astype(np.float64, copy=False)
    pad = np.full((length - arr.shape[0],) + arr.shape[1:], np.nan, dtype=np.float64)
    return np.concatenate([arr.astype(np.float64, copy=False), pad], axis=0)


def _stack_curve(results: Sequence[dict], key: str) -> np.ndarray:
    arrays = [np.asarray(result[key], dtype=np.float64) for result in results]
    if not arrays:
        return np.empty((0, 0))
    max_len = max(a.shape[0] for a in arrays)
    return np.vstack([_pad_to_length(a, max_len) for a in arrays])


def _stack_diag(results: Sequence[dict], key: str) -> np.ndarray:
    arrays = [
        np.asarray(result["diagnostics"][key], dtype=np.float64) for result in results
    ]
    if not arrays:
        return np.empty((0, 0))
    max_len = max(a.shape[0] for a in arrays)
    return np.vstack([_pad_to_length(a, max_len) for a in arrays])


def _sem(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(matrix.shape[1] if matrix.ndim > 1 else 0)
    return stats.sem(matrix, axis=0, nan_policy="omit")


def plot_multiseed_results(
    results: Sequence[dict],
    run_dir,
    exploitability_threshold: float = DEFAULT_EXPLOITABILITY_THRESHOLD,
    average_policy_value_target: float = DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
) -> None:
    """Creates the standard plots for a multi-seed poker Deep CFR run."""
    if not results:
        raise ValueError("No results to plot.")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    iterations = np.asarray(results[0]["iterations"], dtype=np.float64)
    exploitability_mat = _stack_curve(results, "exploitability")
    average_policy_value_mat = _stack_curve(results, "average_policy_value")
    value_error_mat = _stack_curve(results, "policy_value_error")
    nodes_mat = _stack_curve(results, "nodes_touched")

    mean_exploitability = np.nanmean(exploitability_mat, axis=0)
    se_exploitability = _sem(exploitability_mat)
    mean_average_policy_value = np.nanmean(average_policy_value_mat, axis=0)
    se_average_policy_value = _sem(average_policy_value_mat)
    mean_value_error = np.nanmean(value_error_mat, axis=0)
    se_value_error = _sem(value_error_mat)
    mean_nodes = np.nanmean(nodes_mat, axis=0)

    # Pad iterations to the same length the matrices were padded to.
    if iterations.shape[0] < exploitability_mat.shape[1]:
        extra = exploitability_mat.shape[1] - iterations.shape[0]
        if iterations.size >= 2:
            step = iterations[-1] - iterations[-2]
            extension = iterations[-1] + step * np.arange(1, extra + 1)
            iterations = np.concatenate([iterations, extension])
        else:
            iterations = np.concatenate(
                [iterations, np.arange(iterations.size, exploitability_mat.shape[1])]
            )

    # Exploitability against training iteration.
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(
            result["iterations"], result["exploitability"], alpha=0.25, linewidth=1
        )
    ax.plot(iterations, mean_exploitability, linewidth=2, label="Mean across seeds")
    ax.fill_between(
        iterations,
        mean_exploitability - se_exploitability,
        mean_exploitability + se_exploitability,
        alpha=0.2,
        label="Mean ± s.e.",
    )
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Exploitability (NashConv/2)")
    set_chart_title(ax, "Exploitability Across Seeds")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "exploitability_by_iteration_multiseed.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Exploitability against nodes touched.
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(
            result["nodes_touched"],
            result["exploitability"],
            alpha=0.25,
            linewidth=1,
        )
    ax.plot(mean_nodes, mean_exploitability, linewidth=2, label="Mean across seeds")
    ax.fill_between(
        mean_nodes,
        mean_exploitability - se_exploitability,
        mean_exploitability + se_exploitability,
        alpha=0.2,
        label="Mean ± s.e.",
    )
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Exploitability (NashConv/2)")
    set_chart_title(ax, "Exploitability by Nodes Touched")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "exploitability_by_nodes_multiseed.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    # Average policy value against training iteration.
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(
            result["iterations"],
            result["average_policy_value"],
            alpha=0.25,
            linewidth=1,
        )
    ax.plot(iterations, mean_average_policy_value, linewidth=2, label="Mean across seeds")
    ax.fill_between(
        iterations,
        mean_average_policy_value - se_average_policy_value,
        mean_average_policy_value + se_average_policy_value,
        alpha=0.2,
        label="Mean ± s.e.",
    )
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label="Player 0 Nash value",
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Average policy value for player 0")
    set_chart_title(ax, "Average Policy Value Across Seeds")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "average_policy_value_by_iteration_multiseed.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Average policy value against nodes touched.
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(
            result["nodes_touched"],
            result["average_policy_value"],
            alpha=0.25,
            linewidth=1,
        )
    ax.plot(mean_nodes, mean_average_policy_value, linewidth=2, label="Mean across seeds")
    ax.fill_between(
        mean_nodes,
        mean_average_policy_value - se_average_policy_value,
        mean_average_policy_value + se_average_policy_value,
        alpha=0.2,
        label="Mean ± s.e.",
    )
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label="Player 0 Nash value",
    )
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Average policy value for player 0")
    set_chart_title(ax, "Average Policy Value by Nodes Touched")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "average_policy_value_by_nodes_multiseed.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Policy-value error from the configured known game value.
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(
            result["iterations"],
            result["policy_value_error"],
            alpha=0.25,
            linewidth=1,
        )
    ax.plot(iterations, mean_value_error, linewidth=2, label="Mean across seeds")
    ax.fill_between(
        iterations,
        mean_value_error - se_value_error,
        mean_value_error + se_value_error,
        alpha=0.2,
        label="Mean ± s.e.",
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$|v(\sigma) - v^*|$")
    set_chart_title(ax, "Policy-Value Error")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "policy_value_error_multiseed.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    plot_diagnostics(results, run_dir, iterations=iterations)


def plot_diagnostics(
    results: Sequence[dict], run_dir, iterations: np.ndarray = None
) -> None:
    """Creates neural-network and policy-distribution diagnostic plots."""
    run_dir = Path(run_dir)
    if iterations is None:
        iterations = np.asarray(results[0]["iterations"], dtype=np.float64)

    policy_loss_mat = _stack_diag(results, "policy_loss")
    adv_target_var_mat = _stack_diag(results, "advantage_target_variance")
    entropy_mat = _stack_diag(results, "policy_normalized_entropy_mean")
    legal_mass_mat = _stack_diag(results, "legal_action_mass_mean")

    if iterations.shape[0] < policy_loss_mat.shape[1]:
        extra = policy_loss_mat.shape[1] - iterations.shape[0]
        if iterations.size >= 2:
            step = iterations[-1] - iterations[-2]
            iterations = np.concatenate(
                [iterations, iterations[-1] + step * np.arange(1, extra + 1)]
            )
        else:
            iterations = np.arange(policy_loss_mat.shape[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        iterations,
        np.nanmean(policy_loss_mat, axis=0),
        linewidth=2,
        label="Policy loss",
    )
    ax.fill_between(
        iterations,
        np.nanmean(policy_loss_mat, axis=0) - _sem(policy_loss_mat),
        np.nanmean(policy_loss_mat, axis=0) + _sem(policy_loss_mat),
        alpha=0.2,
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("MSE loss")
    set_chart_title(ax, "Average-Policy Network Loss Diagnostic")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "policy_loss_diagnostic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, np.nanmean(adv_target_var_mat, axis=0), linewidth=2)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Advantage target variance")
    set_chart_title(ax, "Advantage-Target Variance Diagnostic")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        run_dir / "advantage_target_variance_diagnostic.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        iterations,
        np.nanmean(entropy_mat, axis=0),
        linewidth=2,
        label="Normalised entropy",
    )
    ax.plot(
        iterations,
        np.nanmean(legal_mass_mat, axis=0),
        linewidth=2,
        label="Raw legal-action mass before masking",
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Diagnostic value")
    set_chart_title(ax, "Policy Distribution Diagnostics")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "policy_distribution_diagnostics.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Generic helpers shared with downstream experiments.
# -----------------------------------------------------------------------------


def plot_pairwise_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[int],
    col_labels: Sequence[int],
    *,
    output_path,
    title: str,
    xlabel: str = "Checkpoint B iteration",
    ylabel: str = "Checkpoint A iteration",
    colorbar_label: str = "Seat-averaged EV for checkpoint A",
    cmap: str = "coolwarm",
    vmin: float = None,
    vmax: float = None,
    fmt: str = ".3f",
    annotate: bool = True,
    figsize: Sequence[float] = (9, 7),
) -> None:
    """Renders a labelled heatmap with optional cell annotations.

    Reusable across any experiment that needs a pairwise matrix figure.
    """
    data = np.asarray(matrix, dtype=float)
    if vmin is None or vmax is None:
        finite = data[np.isfinite(data)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
        if max_abs == 0:
            max_abs = 1.0
        if vmin is None:
            vmin = -max_abs
        if vmax is None:
            vmax = max_abs

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(list(col_labels))
    ax.set_yticklabels(list(row_labels))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    set_chart_title(ax, title)
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    fig.colorbar(im, ax=ax, label=colorbar_label)
    if annotate:
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                if np.isfinite(data[r, c]):
                    ax.text(
                        c,
                        r,
                        format(float(data[r, c]), fmt),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_strength_curve_with_errorbars(
    x: Sequence[float],
    means: Sequence[float],
    sems: Sequence[float],
    *,
    output_path,
    title: str,
    xlabel: str,
    ylabel: str,
    zero_line: bool = True,
    reference_line_value: float = 0.0,
    reference_line_label: Optional[str] = None,
    figsize: Sequence[float] = (9, 5),
) -> None:
    """Generic mean-with-error-bar plot used for cross-seed strength summaries."""
    x = np.asarray(list(x), dtype=float)
    means = np.asarray(list(means), dtype=float)
    sems = np.asarray(list(sems), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(x, means, yerr=sems, marker="o", capsize=3)
    if zero_line:
        ax.axhline(
            reference_line_value,
            linestyle="--",
            linewidth=1,
            label=reference_line_label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    set_chart_title(ax, title)
    ax.grid(True, alpha=0.3)
    if reference_line_label:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_annotated(
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    *,
    output_path,
    title: str,
    xlabel: str,
    ylabel: str,
    zero_line: bool = True,
    x_reference_line_value: Optional[float] = None,
    x_reference_line_label: Optional[str] = None,
    figsize: Sequence[float] = (7, 5),
) -> None:
    """Annotated scatter plot used for strength-vs-exploitability views."""
    xs = np.asarray(list(xs), dtype=float)
    ys = np.asarray(list(ys), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(str(label), (x, y), fontsize=7, alpha=0.7)
    if zero_line:
        ax.axhline(0.0, linestyle="--", linewidth=1)
    if x_reference_line_value is not None:
        ax.axvline(
            x_reference_line_value,
            linestyle="--",
            linewidth=1,
            label=x_reference_line_label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    set_chart_title(ax, title)
    ax.grid(True, alpha=0.3)
    if x_reference_line_label:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
