"""Both-seat duplicate-deal evaluation with paired inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .game import MILLI_BIG_BLINDS_PER_CHIP
from .statistics import sample_summary


def _normalised_distribution(state, policy, player: int) -> tuple[list[int], np.ndarray]:
    legal = set(int(action) for action in state.legal_actions(player))
    raw: Mapping[int, float] = policy.action_probabilities(state, player)
    if set(int(action) for action in raw) - legal:
        raise ValueError("Policy assigned probability to an illegal action")
    actions = sorted(legal)
    probabilities = np.asarray([float(raw.get(action, 0.0)) for action in actions])
    total = float(probabilities.sum())
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or total <= 0.0:
        raise ValueError("Policy returned an invalid action distribution")
    return actions, probabilities / total


def _sample(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probabilities), p=probabilities))


def play_hand(game, policies, *, chance_seed: int, action_seed: int) -> tuple[float, float]:
    state = game.new_initial_state()
    chance_rng = np.random.default_rng(int(chance_seed))
    action_rng = np.random.default_rng(int(action_seed))
    # Independent policy-mixture randomness, never consuming the established
    # chance/action streams. Stateless policies remain byte-for-byte unchanged.
    for seat, agent in enumerate(policies):
        if hasattr(agent, "begin_episode"):
            agent.begin_episode(seed=np.random.SeedSequence([int(action_seed), seat, 934781]))
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            probabilities = np.asarray([float(probability) for _, probability in outcomes])
            probabilities /= probabilities.sum()
            state.apply_action(outcomes[_sample(probabilities, chance_rng)][0])
            continue
        player = int(state.current_player())
        actions, probabilities = _normalised_distribution(state, policies[player], player)
        state.apply_action(actions[_sample(probabilities, action_rng)])
    returns = state.returns()
    return float(returns[0]), float(returns[1])


@dataclass(frozen=True)
class DuplicateMatchResult:
    policy_a: str
    policy_b: str
    seed: int
    num_deal_pairs: int
    num_games: int
    mean_chips_per_hand: float
    std_chips_per_pair: float
    se_chips_per_hand: float
    ci95_low_chips_per_hand: float
    ci95_high_chips_per_hand: float
    mean_mbb_per_hand: float
    se_mbb_per_hand: float
    policy_a_player0_mean_chips: float
    policy_a_player1_mean_chips: float
    common_random_numbers: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_duplicate_match(
    game,
    policy_a,
    policy_b,
    *,
    num_deals: int,
    seed: int,
    policy_a_name: str = "policy_a",
    policy_b_name: str = "policy_b",
    seed_layout: str = "interleaved",
) -> DuplicateMatchResult:
    """Evaluate A against B on identical deals with both seat assignments."""
    if int(num_deals) <= 0:
        raise ValueError("num_deals must be positive")
    master = np.random.default_rng(int(seed))
    if seed_layout not in {"interleaved", "split"}:
        raise ValueError("Unknown duplicate-deal seed layout")
    # The completed UCV retrospective analysis generated all chance seeds,
    # then all action seeds. Preserve its exact streams when requested.
    if seed_layout == "split":
        chance_seeds = master.integers(0, 2**63 - 1, size=int(num_deals), dtype=np.int64)
        action_seeds = master.integers(0, 2**63 - 1, size=int(num_deals), dtype=np.int64)
    as_player_zero = np.empty(int(num_deals), dtype=float)
    as_player_one = np.empty(int(num_deals), dtype=float)
    for index in range(int(num_deals)):
        chance_seed = int(chance_seeds[index] if seed_layout == "split" else master.integers(0, 2**63 - 1))
        action_seed = int(action_seeds[index] if seed_layout == "split" else master.integers(0, 2**63 - 1))
        return_zero, _ = play_hand(
            game, (policy_a, policy_b), chance_seed=chance_seed, action_seed=action_seed
        )
        _, return_one = play_hand(
            game, (policy_b, policy_a), chance_seed=chance_seed, action_seed=action_seed
        )
        as_player_zero[index] = return_zero
        as_player_one[index] = return_one
    paired = 0.5 * (as_player_zero + as_player_one)
    summary = sample_summary(paired)
    return DuplicateMatchResult(
        policy_a=str(policy_a_name),
        policy_b=str(policy_b_name),
        seed=int(seed),
        num_deal_pairs=int(num_deals),
        num_games=2 * int(num_deals),
        mean_chips_per_hand=float(summary["mean"]),
        std_chips_per_pair=float(summary["std"]),
        se_chips_per_hand=float(summary["se"]),
        ci95_low_chips_per_hand=float(summary["ci95_low"]),
        ci95_high_chips_per_hand=float(summary["ci95_high"]),
        mean_mbb_per_hand=float(summary["mean"]) * MILLI_BIG_BLINDS_PER_CHIP,
        se_mbb_per_hand=float(summary["se"]) * MILLI_BIG_BLINDS_PER_CHIP,
        policy_a_player0_mean_chips=float(np.mean(as_player_zero)),
        policy_a_player1_mean_chips=float(np.mean(as_player_one)),
    )


def evaluate_cross_play(game, policies: Mapping[str, object], *, num_deals: int, seed: int):
    """Return one oriented duplicate result for every unordered policy pair."""
    names = list(policies)
    results = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            pair_seed = int(seed) + 1_000_003 * left + 10_007 * right
            results.append(
                evaluate_duplicate_match(
                    game,
                    policies[names[left]],
                    policies[names[right]],
                    num_deals=num_deals,
                    seed=pair_seed,
                    policy_a_name=names[left],
                    policy_b_name=names[right],
                )
            )
    return results
