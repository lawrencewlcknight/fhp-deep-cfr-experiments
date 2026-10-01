"""The thesis-facing evaluation bundle for one learned checkpoint."""

from __future__ import annotations

from .duplicate import evaluate_duplicate_match
from .game import FHP_GAME_PARAMETERS
from .rule_agents import published_rule_agents


def evaluate_against_published_agents(
    game,
    target_policy,
    *,
    num_deals: int,
    seed: int,
    target_name: str = "target",
) -> dict:
    results = []
    for index, (name, opponent) in enumerate(published_rule_agents(game).items()):
        result = evaluate_duplicate_match(
            game,
            target_policy,
            opponent,
            num_deals=num_deals,
            seed=int(seed) + 1_000_003 * index,
            policy_a_name=target_name,
            policy_b_name=name,
        )
        results.append(result.to_dict())
    return {
        "schema_version": 1,
        "suite": "five_published_rule_agents_corrected",
        "game": {"name": "FHP", "parameters": dict(FHP_GAME_PARAMETERS)},
        "loose_aggressive_bands": [-300.0, -100.0],
        "num_deal_pairs_per_opponent": int(num_deals),
        "base_seed": int(seed),
        "results": results,
    }
