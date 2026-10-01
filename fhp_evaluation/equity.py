"""Hand-strength calculations used by the published agents and LBR."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .cards import canonical_starting_hand, showdown_result
from .published_preflop import PUBLISHED_PREFLOP_EQUITY


def published_hand_equity(private_cards: Sequence[int], public_cards: Sequence[int]) -> float:
    """Reproduce the published agent's pre-flop lookup and exact flop equity."""
    private_cards = tuple(int(card) for card in private_cards)
    public_cards = tuple(int(card) for card in public_cards)
    if len(private_cards) != 2 or len(set(private_cards + public_cards)) != 2 + len(public_cards):
        raise ValueError("Cards must be distinct and the private hand must contain two cards")
    if not public_cards:
        return float(PUBLISHED_PREFLOP_EQUITY[canonical_starting_hand(private_cards)])
    if len(public_cards) != 3:
        raise ValueError("Canonical FHP has either zero or three public cards")
    return range_showdown_equity(private_cards, public_cards)


def published_hand_strength(private_cards: Sequence[int], public_cards: Sequence[int]) -> float:
    """Return the scalar used by Xu et al.'s five rule-based opponents."""
    return published_hand_equity(private_cards, public_cards) * 1326.0 - 663.0


def legal_opponent_hands(excluded_cards: Sequence[int]) -> tuple[tuple[int, int], ...]:
    excluded = {int(card) for card in excluded_cards}
    return tuple(combinations((card for card in range(52) if card not in excluded), 2))


def range_showdown_equity(
    hero: Sequence[int],
    board: Sequence[int],
    range_weights: Mapping[tuple[int, int], float] | None = None,
) -> float:
    """Compute exact flop equity against a uniform or explicitly weighted range."""
    hero = tuple(int(card) for card in hero)
    board = tuple(int(card) for card in board)
    if len(hero) != 2 or len(board) != 3 or len(set(hero + board)) != 5:
        raise ValueError("Expected two distinct hero cards and a distinct three-card flop")
    hands = legal_opponent_hands(hero + board)
    if range_weights is None:
        weights = np.full(len(hands), 1.0 / len(hands), dtype=float)
    else:
        weights = np.asarray([float(range_weights.get(hand, 0.0)) for hand in hands])
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Opponent range has no legal positive mass")
        weights /= total
    outcomes = np.fromiter(
        (showdown_result(hero, hand, board) for hand in hands),
        dtype=float,
        count=len(hands),
    )
    return float(np.clip(np.dot(weights, outcomes), 0.0, 1.0))


def sampled_rollout_equity(
    hero: Sequence[int],
    board: Sequence[int],
    hands: Sequence[tuple[int, int]],
    weights: Sequence[float],
    *,
    num_samples: int,
    rng: np.random.Generator,
) -> float:
    """Estimate FHP showdown equity while respecting an opponent hand range."""
    hero = tuple(int(card) for card in hero)
    board = tuple(int(card) for card in board)
    weights = np.asarray(weights, dtype=float)
    weights /= float(weights.sum())
    if len(board) == 3:
        return float(np.clip(
            sum(
                weight * showdown_result(hero, hand, board)
                for hand, weight in zip(hands, weights)
            ),
            0.0,
            1.0,
        ))
    if board or num_samples <= 0:
        raise ValueError("Pre-flop rollout requires no board and a positive sample count")
    indices = rng.choice(len(hands), size=int(num_samples), replace=True, p=weights)
    total = 0.0
    hero_set = set(hero)
    for index in indices:
        opponent = hands[int(index)]
        remaining = [card for card in range(52) if card not in hero_set and card not in opponent]
        flop = tuple(int(card) for card in rng.choice(remaining, size=3, replace=False))
        total += showdown_result(hero, opponent, flop)
    return total / float(num_samples)
