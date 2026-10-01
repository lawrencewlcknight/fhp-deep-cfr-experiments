"""Poker-specific Local Best Response after Lisý and Bowling (2017)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import re
from typing import Sequence

import numpy as np
from open_spiel.python import policy

from .cards import cards_from_information_state
from .duplicate import evaluate_duplicate_match
from .equity import sampled_rollout_equity
from .game import FHP_GAME_PARAMETERS


FOLD, CALL, RAISE = 0, 1, 2


@dataclass(frozen=True)
class LBRConfig:
    """Parameters that define a reproducible LBR evaluation tier."""

    preflop_rollout_samples: int = 4096
    seed: int = 0
    probability_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.preflop_rollout_samples <= 0:
            raise ValueError("preflop_rollout_samples must be positive")
        if self.probability_tolerance < 0.0:
            raise ValueError("probability_tolerance must be non-negative")


@dataclass
class _Hypothesis:
    hand: tuple[int, int]
    weight: float
    state: object


def _policy_probability(target_policy, state, player: int, action: int) -> float:
    legal = set(int(value) for value in state.legal_actions(player))
    probabilities = target_policy.action_probabilities(state, player)
    if set(int(value) for value in probabilities) - legal:
        raise ValueError("Target policy assigned mass to an illegal action")
    values = np.asarray([float(probabilities.get(value, 0.0)) for value in sorted(legal)])
    if np.any(values < 0.0) or not np.isfinite(values).all() or values.sum() <= 0.0:
        raise ValueError("Target policy returned an invalid distribution")
    values /= values.sum()
    return float(values[sorted(legal).index(int(action))]) if int(action) in legal else 0.0


def _spent(state) -> tuple[float, float]:
    match = re.search(r"Spent: \[P0: ([0-9.]+)\s+P1: ([0-9.]+)\s*\]", str(state))
    if match is None:
        raise ValueError("Could not recover player contributions from Universal Poker state")
    return float(match.group(1)), float(match.group(2))


def _seed_for_information_set(text: str, seed: int, salt: str) -> int:
    payload = f"{seed}\0{salt}\0{text}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class LocalBestResponsePolicy(policy.Policy):
    """A legal one-step LBR strategy against a queryable target policy.

    Opponent ranges are reconstructed from the public action history by exact
    Bayes updates over all legal private hands.  The action scorer follows the
    published check/call-to-showdown heuristic.  Flop equity is exact; pre-flop
    board completion is a deterministic Monte Carlo approximation controlled
    by :class:`LBRConfig`.
    """

    def __init__(self, game, target_policy, *, config: LBRConfig | None = None):
        super().__init__(game, list(range(game.num_players())))
        self.game = game
        self.target_policy = target_policy
        self.config = config or LBRConfig()
        self._action_cache: dict[tuple[int, str], int] = {}

    def _replay_hypothesis(
        self,
        history: Sequence[int],
        *,
        responder: int,
        opponent_hand: tuple[int, int],
    ) -> tuple[object, float]:
        opponent = 1 - int(responder)
        # Universal Poker deals both private cards to player 0, then both to
        # player 1, before exposing a decision node.
        opponent_slots = (2, 3) if responder == 0 else (0, 1)
        replacements = dict(zip(opponent_slots, opponent_hand))
        state = self.game.new_initial_state()
        chance_index = 0
        likelihood = 1.0
        for observed_action in history:
            action = int(observed_action)
            if state.is_chance_node():
                action = int(replacements.get(chance_index, action))
                chance_index += 1
            else:
                actor = int(state.current_player())
                if actor == opponent:
                    likelihood *= _policy_probability(
                        self.target_policy, state, opponent, int(observed_action)
                    )
                action = int(observed_action)
            state.apply_action(action)
            if likelihood == 0.0:
                # Continue replaying to return the correctly reconstructed state.
                continue
        return state, likelihood

    def _opponent_range(self, state, responder: int) -> list[_Hypothesis]:
        private, board = cards_from_information_state(state, responder)
        excluded = set(private + board)
        hands = tuple(combinations((card for card in range(52) if card not in excluded), 2))
        history = tuple(int(action) for action in state.history())
        hypotheses: list[_Hypothesis] = []
        total = 0.0
        for hand in hands:
            hypothetical_state, likelihood = self._replay_hypothesis(
                history, responder=responder, opponent_hand=hand
            )
            if likelihood > 0.0:
                hypotheses.append(_Hypothesis(hand=hand, weight=likelihood, state=hypothetical_state))
                total += likelihood
        if not np.isfinite(total) or total <= self.config.probability_tolerance:
            raise RuntimeError("Bayesian opponent range collapsed to zero mass")
        for hypothesis in hypotheses:
            hypothesis.weight /= total
        return hypotheses

    def _equity(
        self,
        state,
        responder: int,
        hypotheses: Sequence[_Hypothesis],
        *,
        salt: str,
    ) -> float:
        private, board = cards_from_information_state(state, responder)
        info = state.information_state_string(responder)
        rng = np.random.default_rng(_seed_for_information_set(info, self.config.seed, salt))
        return sampled_rollout_equity(
            private,
            board,
            [hypothesis.hand for hypothesis in hypotheses],
            [hypothesis.weight for hypothesis in hypotheses],
            num_samples=self.config.preflop_rollout_samples,
            rng=rng,
        )

    def _raise_fold_range(
        self, hypotheses: Sequence[_Hypothesis], responder: int
    ) -> tuple[float, list[_Hypothesis]]:
        opponent = 1 - responder
        fold_probability = 0.0
        continuing: list[_Hypothesis] = []
        continuing_mass = 0.0
        for hypothesis in hypotheses:
            child = hypothesis.state.clone()
            child.apply_action(RAISE)
            probability = 0.0
            if not child.is_terminal() and not child.is_chance_node() and child.current_player() == opponent:
                probability = _policy_probability(self.target_policy, child, opponent, FOLD)
            fold_probability += hypothesis.weight * probability
            weight = hypothesis.weight * (1.0 - probability)
            if weight > 0.0:
                continuing.append(_Hypothesis(hypothesis.hand, weight, child))
                continuing_mass += weight
        if continuing_mass > self.config.probability_tolerance:
            for hypothesis in continuing:
                hypothesis.weight /= continuing_mass
        else:
            continuing = list(hypotheses)
        return float(fold_probability), continuing

    def _choose_action(self, state, responder: int) -> int:
        legal = set(int(action) for action in state.legal_actions(responder))
        if not legal:
            raise ValueError("LBR queried at a state with no legal actions")
        hypotheses = self._opponent_range(state, responder)
        spent = _spent(state)
        pot = spent[0] + spent[1]
        asked = max(0.0, spent[1 - responder] - spent[responder])

        values: dict[int, float] = {}
        if CALL in legal:
            equity = self._equity(state, responder, hypotheses, salt="call")
            values[CALL] = equity * pot - (1.0 - equity) * asked
        if RAISE in legal:
            fold_probability, continuing = self._raise_fold_range(hypotheses, responder)
            equity = self._equity(state, responder, continuing, salt="raise")
            actual_child = state.clone()
            actual_child.apply_action(RAISE)
            raised_spent = _spent(actual_child)
            raise_increment = raised_spent[responder] - max(spent)
            values[RAISE] = fold_probability * pot + (1.0 - fold_probability) * (
                equity * (pot + raise_increment)
                - (1.0 - equity) * (asked + raise_increment)
            )

        if values:
            best_action = max(values, key=lambda action: (values[action], -action))
            if values[best_action] > 0.0 or FOLD not in legal:
                return int(best_action)
        if FOLD in legal:
            return FOLD
        if CALL in legal:
            return CALL
        return min(legal)

    def action_probabilities(self, state, player_id=None):
        responder = state.current_player() if player_id is None else int(player_id)
        if responder != state.current_player():
            raise ValueError("LBR can only be queried for the acting player")
        key = (responder, state.information_state_string(responder))
        action = self._action_cache.get(key)
        if action is None:
            action = self._choose_action(state, responder)
            self._action_cache[key] = action
        return {action: 1.0}


def evaluate_lbr(
    game,
    target_policy,
    *,
    num_deals: int,
    seed: int,
    config: LBRConfig | None = None,
    target_name: str = "target",
) -> dict:
    """Estimate the both-seat LBR lower bound against one policy profile."""
    config = config or LBRConfig(seed=seed)
    lbr = LocalBestResponsePolicy(game, target_policy, config=config)
    result = evaluate_duplicate_match(
        game,
        lbr,
        target_policy,
        num_deals=num_deals,
        seed=seed,
        policy_a_name="local_best_response",
        policy_b_name=target_name,
    ).to_dict()
    result.update(
        {
            "schema_version": 1,
            "game": {"name": "FHP", "parameters": dict(FHP_GAME_PARAMETERS)},
            "metric": "lbr_lower_bound",
            "interpretation": "lower_bound_on_best_response_value_not_exact_exploitability",
            "lbr_config": asdict(config),
        }
    )
    return result
