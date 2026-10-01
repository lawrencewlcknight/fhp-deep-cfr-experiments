"""The immutable game contract used by every FHP training repository."""

from __future__ import annotations

from types import MappingProxyType

import pyspiel


FHP_GAME_PARAMETERS = MappingProxyType(
    {
        "betting": "limit",
        "blind": "50 100",
        "raiseSize": "100 100",
        "firstPlayer": "1 2",
        "maxRaises": "3 3",
        "numRounds": 2,
        "numSuits": 4,
        "numRanks": 13,
        "numHoleCards": 2,
        "numBoardCards": "0 3",
        "numPlayers": 2,
    }
)

BIG_BLIND_CHIPS = 100.0
MILLI_BIG_BLINDS_PER_CHIP = 1000.0 / BIG_BLIND_CHIPS


def load_fhp_game():
    """Load the canonical two-player fixed-limit FHP game."""
    return pyspiel.load_game("universal_poker", dict(FHP_GAME_PARAMETERS))


def assert_canonical_fhp(game) -> None:
    """Reject a game whose observable dimensions do not match the contract."""
    if game.num_players() != 2 or game.num_distinct_actions() != 3:
        raise ValueError("Expected two-player FHP with fold/call/raise actions")
    if game.information_state_tensor_size() != 190:
        raise ValueError("Unexpected FHP information-state tensor size")

