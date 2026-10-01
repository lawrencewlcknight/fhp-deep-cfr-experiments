"""Corrected implementations of the five agents used by Xu et al. (2026)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from open_spiel.python import policy

from .cards import cards_from_information_state
from .equity import published_hand_strength


FOLD, CALL, RAISE = 0, 1, 2
PUBLISHED_AGENT_NAMES = (
    "candid_statistician",
    "loose_aggressive",
    "loose_passive",
    "tight_passive",
    "tight_aggressive",
)


def _fallback(legal: set[int], preferred: tuple[int, ...]) -> int:
    for action in preferred:
        if action in legal:
            return action
    raise ValueError("No preferred action is legal")


@dataclass(frozen=True)
class StrengthBands:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError("Hand-strength bands must be strictly increasing")


class HandStrengthPolicy(policy.Policy):
    def __init__(
        self,
        game,
        bands: StrengthBands,
        *,
        name: str,
        strength_fn: Callable = published_hand_strength,
    ) -> None:
        super().__init__(game, list(range(game.num_players())))
        self.bands = bands
        self.name = str(name)
        self._strength_fn = strength_fn

    def action_for_strength(self, strength: float, legal_actions) -> int:
        legal = set(int(action) for action in legal_actions)
        if strength < self.bands.low:
            return _fallback(legal, (FOLD, CALL))
        if strength < self.bands.high:
            return _fallback(legal, (CALL,))
        return _fallback(legal, (RAISE, CALL))

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal = state.legal_actions(player)
        private, public = cards_from_information_state(state, player)
        action = self.action_for_strength(self._strength_fn(private, public), legal)
        return {action: 1.0}


class TightAggressivePolicy(HandStrengthPolicy):
    def __init__(self, game, *, bluff_probability: float = 0.2, strength_fn=published_hand_strength):
        super().__init__(
            game,
            StrengthBands(-100.0, 100.0),
            name="tight_aggressive",
            strength_fn=strength_fn,
        )
        if not 0.0 <= bluff_probability <= 1.0:
            raise ValueError("Bluff probability must lie in [0, 1]")
        self.bluff_probability = float(bluff_probability)

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal = set(int(action) for action in state.legal_actions(player))
        private, public = cards_from_information_state(state, player)
        strength = self._strength_fn(private, public)
        if strength < self.bands.low and RAISE in legal:
            passive = _fallback(legal, (FOLD, CALL))
            if passive == RAISE:
                return {RAISE: 1.0}
            return {RAISE: self.bluff_probability, passive: 1.0 - self.bluff_probability}
        action = self.action_for_strength(strength, legal)
        return {action: 1.0}


def published_rule_agents(game) -> dict[str, policy.Policy]:
    """Construct the five published agents with the loose-aggressive typo fixed.

    The released DeepPDCFR code specifies ``(-100, -300)`` for LooseAggressive,
    which makes the middle band unreachable.  The intended increasing ordering,
    ``(-300, -100)``, is used here and guarded by :class:`StrengthBands`.
    """
    return {
        "candid_statistician": HandStrengthPolicy(
            game, StrengthBands(-100.0, 100.0), name="candid_statistician"
        ),
        "loose_aggressive": HandStrengthPolicy(
            game, StrengthBands(-300.0, -100.0), name="loose_aggressive"
        ),
        "loose_passive": HandStrengthPolicy(
            game, StrengthBands(-300.0, 500.0), name="loose_passive"
        ),
        "tight_passive": HandStrengthPolicy(
            game, StrengthBands(100.0, 500.0), name="tight_passive"
        ),
        "tight_aggressive": TightAggressivePolicy(game),
    }

