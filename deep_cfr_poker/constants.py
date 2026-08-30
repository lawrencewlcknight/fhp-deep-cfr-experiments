"""Shared constants for FHP Deep CFR experiments."""

DEFAULT_ALGORITHM_VARIANT = "Deep CFR"
DEFAULT_GAME_NAME = "FHP"

# Exact equilibrium values are deliberately not claimed for FHP. The game is
# asymmetric by seat within a single hand, and exact full-tree evaluation is
# impractical at experiment scale.
KNOWN_GAME_VALUES_PLAYER_0 = {}
DEFAULT_AVERAGE_POLICY_VALUE_TARGET = float("nan")
DEFAULT_EXPLOITABILITY_THRESHOLD = 0.05
DEFAULT_SOLVER_BATCH_SIZE = 1024
