"""Evaluation utilities for trained policies.

This module concentrates the *evaluation* primitives that experiments share —
exact pairwise expected value, optional Monte Carlo head-to-head, and
monotonicity summaries.

Every function in here takes ``open_spiel`` :class:`policy.Policy`-compatible
objects and a game. Both :class:`deep_cfr_poker.snapshots.LoadedPolicy` and
``open_spiel.python.policy.tabular_policy_from_callable(...)`` outputs work.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from open_spiel.python import policy as osp_policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability


@dataclass
class CheckpointPolicyMetrics:
    seed: str
    checkpoint: int
    nash_conv: float
    exploitability: float
    average_policy_value: float
    policy_value_signed_error: float
    policy_value_error: float


def make_tabular_policy(game, callable_policy):
    """Convenience wrapper around ``tabular_policy_from_callable``."""
    return osp_policy.tabular_policy_from_callable(
        game, callable_policy.action_probabilities
    )


def evaluate_checkpoint_metrics(
    game,
    policies_by_seed: Mapping[str, Mapping[int, "osp_policy.Policy"]],
    *,
    known_game_value: float,
) -> List[CheckpointPolicyMetrics]:
    """Returns exact NashConv / value metrics per (seed, checkpoint)."""
    metrics: List[CheckpointPolicyMetrics] = []
    for seed, policies in policies_by_seed.items():
        for iteration in sorted(policies):
            pol = policies[iteration]
            tab_pol = make_tabular_policy(game, pol)
            nash_conv = float(exploitability.nash_conv(game, tab_pol))
            value = float(
                expected_game_score.policy_value(
                    game.new_initial_state(), [tab_pol] * game.num_players()
                )[0]
            )
            metrics.append(
                CheckpointPolicyMetrics(
                    seed=str(seed),
                    checkpoint=int(iteration),
                    nash_conv=nash_conv,
                    exploitability=nash_conv / 2.0,
                    average_policy_value=value,
                    policy_value_signed_error=value - known_game_value,
                    policy_value_error=abs(value - known_game_value),
                )
            )
    return metrics


def exact_seat_averaged_value(
    game,
    agent_a: "osp_policy.Policy",
    agent_b: "osp_policy.Policy",
) -> Dict[str, float]:
    """Returns A's exact EV against B, averaged over both seat assignments.

    Returned dict has keys ``A_EV_as_player_0``, ``A_EV_as_player_1``,
    ``A_EV_seat_averaged``.
    """
    a_p0, _b_p0 = expected_game_score.policy_value(
        game.new_initial_state(), [agent_a, agent_b]
    )
    _b_p1, a_p1 = expected_game_score.policy_value(
        game.new_initial_state(), [agent_b, agent_a]
    )
    return {
        "A_EV_as_player_0": float(a_p0),
        "A_EV_as_player_1": float(a_p1),
        "A_EV_seat_averaged": float((a_p0 + a_p1) / 2.0),
    }


@dataclass
class HeadToHeadRecord:
    seed: str
    checkpoint_a: int
    checkpoint_b: int
    A_EV_as_player_0: float
    A_EV_as_player_1: float
    A_EV_seat_averaged: float


def exact_pairwise_head_to_head(
    game,
    policies_by_seed: Mapping[str, Mapping[int, "osp_policy.Policy"]],
) -> List[HeadToHeadRecord]:
    """Computes the full pairwise head-to-head matrix per seed."""
    records: List[HeadToHeadRecord] = []
    for seed, policies in policies_by_seed.items():
        iterations = sorted(policies)
        for i in iterations:
            for j in iterations:
                if i == j:
                    payload = {
                        "A_EV_as_player_0": 0.0,
                        "A_EV_as_player_1": 0.0,
                        "A_EV_seat_averaged": 0.0,
                    }
                else:
                    payload = exact_seat_averaged_value(
                        game, policies[i], policies[j]
                    )
                records.append(
                    HeadToHeadRecord(
                        seed=str(seed),
                        checkpoint_a=int(i),
                        checkpoint_b=int(j),
                        **payload,
                    )
                )
    return records


def head_to_head_records_to_arrays(
    records: Sequence[HeadToHeadRecord],
) -> Dict[str, np.ndarray]:
    """Converts records to flat NumPy arrays keyed by column name."""
    n = len(records)
    seeds = np.empty(n, dtype=object)
    checkpoint_a = np.empty(n, dtype=np.int64)
    checkpoint_b = np.empty(n, dtype=np.int64)
    ev_p0 = np.empty(n, dtype=np.float64)
    ev_p1 = np.empty(n, dtype=np.float64)
    ev_seat = np.empty(n, dtype=np.float64)
    for idx, record in enumerate(records):
        seeds[idx] = record.seed
        checkpoint_a[idx] = record.checkpoint_a
        checkpoint_b[idx] = record.checkpoint_b
        ev_p0[idx] = record.A_EV_as_player_0
        ev_p1[idx] = record.A_EV_as_player_1
        ev_seat[idx] = record.A_EV_seat_averaged
    return {
        "seed": seeds,
        "checkpoint_A": checkpoint_a,
        "checkpoint_B": checkpoint_b,
        "A_EV_as_player_0": ev_p0,
        "A_EV_as_player_1": ev_p1,
        "A_EV_seat_averaged": ev_seat,
    }


def aggregate_head_to_head(
    records: Sequence[HeadToHeadRecord],
    *,
    equivalence_epsilon: float = 1e-3,
):
    """Returns (mean_matrix, win_fraction_matrix, sorted_iterations).

    ``mean_matrix[i, j]`` is the cross-seed mean of A's seat-averaged EV when
    A is checkpoint i and B is checkpoint j. ``win_fraction_matrix[i, j]`` is
    the fraction of seeds for which A's EV exceeds ``+equivalence_epsilon``.
    """
    iterations = sorted({r.checkpoint_a for r in records} | {r.checkpoint_b for r in records})
    index = {iteration: pos for pos, iteration in enumerate(iterations)}

    sums = np.zeros((len(iterations), len(iterations)), dtype=np.float64)
    counts = np.zeros((len(iterations), len(iterations)), dtype=np.int64)
    wins = np.zeros((len(iterations), len(iterations)), dtype=np.int64)

    for record in records:
        i = index[record.checkpoint_a]
        j = index[record.checkpoint_b]
        sums[i, j] += record.A_EV_seat_averaged
        counts[i, j] += 1
        if record.A_EV_seat_averaged > equivalence_epsilon:
            wins[i, j] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
        win_fraction = np.where(counts > 0, wins / counts, np.nan)
    return mean, win_fraction, iterations


@dataclass
class MonotonicitySummary:
    seed: str
    num_later_vs_earlier_pairs: int
    all_pairs_clear_improvement_rate: float
    all_pairs_tie_rate: float
    all_pairs_clear_regression_rate: float
    adjacent_clear_improvement_rate: float
    adjacent_tie_rate: float
    adjacent_clear_regression_rate: float
    mean_later_vs_earlier_ev: float
    worst_monotonicity_violation_ev: float
    best_later_vs_earlier_ev: float


def _classify_ev(ev: float, eps: float) -> str:
    if ev > eps:
        return "clear_win"
    if ev < -eps:
        return "clear_loss"
    return "tie"


def monotonicity_summary(
    seed_to_matrix: Mapping[str, np.ndarray],
    sorted_iterations: Sequence[int],
    *,
    equivalence_epsilon: float = 1e-3,
) -> List[MonotonicitySummary]:
    """Classifies later-vs-earlier and adjacent transitions per seed.

    ``seed_to_matrix[seed]`` should be the A-vs-B EV matrix indexed in the same
    order as ``sorted_iterations``.
    """
    summaries: List[MonotonicitySummary] = []
    iterations = list(sorted_iterations)
    for seed, matrix in seed_to_matrix.items():
        later_earlier_evs: List[float] = []
        adjacent_evs: List[float] = []
        for a_idx, _later in enumerate(iterations):
            for earlier_idx in range(a_idx):
                later_earlier_evs.append(float(matrix[a_idx, earlier_idx]))
            if a_idx > 0:
                adjacent_evs.append(float(matrix[a_idx, a_idx - 1]))

        if later_earlier_evs:
            classes = [_classify_ev(x, equivalence_epsilon) for x in later_earlier_evs]
            mean_later = float(np.mean(later_earlier_evs))
            worst = float(np.min(later_earlier_evs))
            best = float(np.max(later_earlier_evs))
            improvement = float(np.mean([c == "clear_win" for c in classes]))
            tie = float(np.mean([c == "tie" for c in classes]))
            regression = float(np.mean([c == "clear_loss" for c in classes]))
        else:
            mean_later = worst = best = float("nan")
            improvement = tie = regression = float("nan")

        if adjacent_evs:
            adj_classes = [_classify_ev(x, equivalence_epsilon) for x in adjacent_evs]
            adj_improvement = float(np.mean([c == "clear_win" for c in adj_classes]))
            adj_tie = float(np.mean([c == "tie" for c in adj_classes]))
            adj_regression = float(np.mean([c == "clear_loss" for c in adj_classes]))
        else:
            adj_improvement = adj_tie = adj_regression = float("nan")

        summaries.append(
            MonotonicitySummary(
                seed=str(seed),
                num_later_vs_earlier_pairs=len(later_earlier_evs),
                all_pairs_clear_improvement_rate=improvement,
                all_pairs_tie_rate=tie,
                all_pairs_clear_regression_rate=regression,
                adjacent_clear_improvement_rate=adj_improvement,
                adjacent_tie_rate=adj_tie,
                adjacent_clear_regression_rate=adj_regression,
                mean_later_vs_earlier_ev=mean_later,
                worst_monotonicity_violation_ev=worst,
                best_later_vs_earlier_ev=best,
            )
        )
    return summaries


@dataclass
class StrengthRow:
    seed: str
    checkpoint: int
    mean_EV_vs_all_other_checkpoints: float
    mean_EV_vs_earlier_checkpoints: float
    mean_EV_vs_later_checkpoints: float
    EV_vs_previous_checkpoint: float


def strength_per_checkpoint(
    seed_to_matrix: Mapping[str, np.ndarray],
    sorted_iterations: Sequence[int],
) -> List[StrengthRow]:
    """Computes per-(seed, checkpoint) head-to-head strength summaries."""
    rows: List[StrengthRow] = []
    iterations = list(sorted_iterations)
    for seed, matrix in seed_to_matrix.items():
        for idx, iteration in enumerate(iterations):
            row_vals = matrix[idx, :]
            others = np.delete(row_vals, idx)
            earlier_mask = np.zeros_like(row_vals, dtype=bool)
            later_mask = np.zeros_like(row_vals, dtype=bool)
            earlier_mask[:idx] = True
            later_mask[idx + 1 :] = True

            mean_all = float(np.mean(others)) if others.size else float("nan")
            mean_earlier = (
                float(np.mean(row_vals[earlier_mask])) if earlier_mask.any() else float("nan")
            )
            mean_later = (
                float(np.mean(row_vals[later_mask])) if later_mask.any() else float("nan")
            )
            ev_vs_prev = float(matrix[idx, idx - 1]) if idx > 0 else float("nan")

            rows.append(
                StrengthRow(
                    seed=str(seed),
                    checkpoint=int(iteration),
                    mean_EV_vs_all_other_checkpoints=mean_all,
                    mean_EV_vs_earlier_checkpoints=mean_earlier,
                    mean_EV_vs_later_checkpoints=mean_later,
                    EV_vs_previous_checkpoint=ev_vs_prev,
                )
            )
    return rows


def aggregate_strength(strength_rows: Sequence[StrengthRow]) -> List[Dict[str, float]]:
    """Across-seed mean / SE for each per-checkpoint strength column."""
    by_checkpoint: Dict[int, Dict[str, List[float]]] = {}
    for row in strength_rows:
        bucket = by_checkpoint.setdefault(int(row.checkpoint), {})
        for col in (
            "mean_EV_vs_all_other_checkpoints",
            "mean_EV_vs_earlier_checkpoints",
            "mean_EV_vs_later_checkpoints",
            "EV_vs_previous_checkpoint",
        ):
            bucket.setdefault(col, []).append(getattr(row, col))

    out: List[Dict[str, float]] = []
    for iteration in sorted(by_checkpoint):
        row: Dict[str, float] = {"checkpoint": iteration}
        for col, values in by_checkpoint[iteration].items():
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                row[f"{col}_mean"] = float(np.mean(arr))
                row[f"{col}_sem"] = (
                    float(np.std(arr, ddof=1) / math.sqrt(arr.size))
                    if arr.size > 1
                    else 0.0
                )
                row[f"{col}_n"] = int(arr.size)
            else:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_sem"] = float("nan")
                row[f"{col}_n"] = 0
        out.append(row)
    return out


# -----------------------------------------------------------------------------
# Optional Monte Carlo head-to-head play.
# -----------------------------------------------------------------------------


def _sample_from_probs(action_prob_dict: Dict[int, float], rng: random.Random) -> int:
    actions = list(action_prob_dict.keys())
    if not actions:
        raise ValueError("Cannot sample from empty action distribution.")
    probs = np.asarray([action_prob_dict[a] for a in actions], dtype=float)
    total = float(probs.sum())
    if total <= 0 or not np.isfinite(total):
        return rng.choice(actions)
    probs = probs / total
    r = rng.random()
    cum = 0.0
    for action, prob in zip(actions, probs):
        cum += prob
        if r <= cum:
            return action
    return actions[-1]


def _play_one_episode(game, policies, rng: random.Random):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            r = rng.random()
            cum = 0.0
            chosen = actions[-1]
            for action, prob in zip(actions, probs):
                cum += prob
                if r <= cum:
                    chosen = action
                    break
            state.apply_action(int(chosen))
        else:
            current = state.current_player()
            probs = policies[current].action_probabilities(state, current)
            state.apply_action(int(_sample_from_probs(probs, rng)))
    return state.returns()


@dataclass
class MonteCarloResult:
    A_avg_return: float
    A_sample_std: float
    A_avg_return_stderr: float
    A_ci95_lo: float
    A_ci95_hi: float
    num_episodes: int


def monte_carlo_head_to_head(
    game,
    agent_a: "osp_policy.Policy",
    agent_b: "osp_policy.Policy",
    *,
    num_episodes: int = 10_000,
    seed: int = 0,
    alternate_seats: bool = True,
) -> MonteCarloResult:
    """Plays ``num_episodes`` games between A and B and returns A's stats."""
    rng = random.Random(int(seed))
    a_returns: List[float] = []
    for episode in range(int(num_episodes)):
        if alternate_seats and (episode % 2 == 1):
            policies_for_episode = [agent_b, agent_a]
            a_seat = 1
        else:
            policies_for_episode = [agent_a, agent_b]
            a_seat = 0
        returns = _play_one_episode(game, policies_for_episode, rng)
        a_returns.append(float(returns[a_seat]))

    arr = np.asarray(a_returns, dtype=np.float64)
    if arr.size <= 1:
        return MonteCarloResult(
            A_avg_return=float(arr.mean()) if arr.size else 0.0,
            A_sample_std=0.0,
            A_avg_return_stderr=float("nan"),
            A_ci95_lo=float("nan"),
            A_ci95_hi=float("nan"),
            num_episodes=int(arr.size),
        )
    mean_return = float(arr.mean())
    sample_std = float(arr.std(ddof=1))
    stderr = sample_std / math.sqrt(arr.size)
    return MonteCarloResult(
        A_avg_return=mean_return,
        A_sample_std=sample_std,
        A_avg_return_stderr=stderr,
        A_ci95_lo=mean_return - 1.96 * stderr,
        A_ci95_hi=mean_return + 1.96 * stderr,
        num_episodes=int(arr.size),
    )
