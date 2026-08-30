"""Canonical OpenSpiel definition of flop hold'em poker (FHP)."""

from __future__ import annotations

from types import MappingProxyType

import pyspiel


FHP_GAME_NAME = "FHP"
OPEN_SPIEL_GAME_NAME = "universal_poker"

# This is the exact FHP definition used by the existing FHP repository in this
# workspace and released by the VR-Deep authors. Keep round parameters as
# strings because OpenSpiel parses one value per round.
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

VR_DEEP_FHP_SOURCE = MappingProxyType(
    {
        "repository": "https://github.com/rpSebastian/DeepPDCFR",
        "commit": "9f156c9fcdac7f8c9bd0debf94c9432d222858d3",
        "class": "deeppdcfr.game.FHP",
    }
)


def load_fhp_game():
    """Load the canonical two-player FHP game through OpenSpiel."""
    return pyspiel.load_game(OPEN_SPIEL_GAME_NAME, dict(FHP_GAME_PARAMETERS))


def load_game(game_name: str = FHP_GAME_NAME):
    """Resolve the repository's serialisable game name to an OpenSpiel game."""
    if str(game_name).upper() != FHP_GAME_NAME:
        raise ValueError(f"Unsupported game {game_name!r}; expected {FHP_GAME_NAME!r}.")
    return load_fhp_game()


def serialisable_game_definition() -> dict:
    """Return the immutable game contract in checkpoint-friendly form."""
    return {
        "name": FHP_GAME_NAME,
        "open_spiel_game": OPEN_SPIEL_GAME_NAME,
        "parameters": dict(FHP_GAME_PARAMETERS),
        "source": dict(VR_DEEP_FHP_SOURCE),
    }
