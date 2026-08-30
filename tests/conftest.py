"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def fhp_game():
    pytest.importorskip("pyspiel")
    from deep_cfr_poker.game import load_fhp_game

    return load_fhp_game()
