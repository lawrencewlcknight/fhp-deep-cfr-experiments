"""Experiment 3: change only the SD-CFR player-information representation."""
import numpy as np

from .fhp_features import DENOMINATORS, FEATURE_SIZE, FHPFeatureEncoder
from .game import load_fhp_game
from .sd_cfr_optimised import PackedAdvantageReservoirBuffer, OptimisedSingleDeepCFRSolver


class StructuredAdvantageReservoirBuffer(PackedAdvantageReservoirBuffer):
    """Lossless two-bit integer numerators; fractions reconstructed in float32.

    The 183 features occupy 46 bytes per row, not 732 float32 bytes. Replay
    insertion, replacement, sample order and RNG consumption are unchanged.
    This is exact encoding of the v1 finite alphabet, NOT lossy quantisation.
    """
    bits_per_feature = 2
    feature_encoding = "fhp_v1_two_bit_numerators_little"

    def __init__(self, capacity, *, info_state_size, num_actions):
        if info_state_size != FEATURE_SIZE:
            raise ValueError("Structured replay requires 183 features")
        super().__init__(capacity, info_state_size=info_state_size, num_actions=num_actions)

    def _pack(self, features):
        values = np.asarray(features, dtype=np.float32)
        if values.ndim not in (1, 2) or values.shape[-1] != FEATURE_SIZE:
            raise ValueError("Structured replay input shape mismatch")
        scaled = np.rint(values * DENOMINATORS)
        if (not np.isfinite(values).all() or np.any(scaled < 0) or np.any(scaled > DENOMINATORS)
                or not np.array_equal(scaled / DENOMINATORS, values)):
            raise ValueError("Features are outside the exact structured v1 alphabet")
        numerators = scaled.astype(np.uint8)
        bits = ((numerators[..., None] >> np.asarray([0, 1], dtype=np.uint8)) & 1)
        return np.packbits(bits.reshape(*values.shape[:-1], FEATURE_SIZE * 2), axis=-1, bitorder="little")

    def _unpack(self, features):
        bits = np.unpackbits(features, axis=-1, count=FEATURE_SIZE * 2, bitorder="little")
        bits = bits.reshape(*bits.shape[:-1], FEATURE_SIZE, 2)
        numerators = bits[..., 0] + 2 * bits[..., 1]
        return numerators.astype(np.float32) / DENOMINATORS


class StructuredSingleDeepCFRSolver(OptimisedSingleDeepCFRSolver):
    packed_buffer_class = StructuredAdvantageReservoirBuffer

    def __init__(self, game=None, *, metadata=None, **kwargs):
        game = load_fhp_game() if game is None else game
        if str(game) != str(load_fhp_game()):
            raise ValueError("The structured encoder is specific to the frozen FHP game")
        self.feature_encoder = FHPFeatureEncoder()
        super().__init__(game, metadata={**dict(metadata or {}),
                                       "feature_encoder": self.feature_encoder.metadata()}, **kwargs)
        self.archive.metadata["execution_optimisations"].update(
            packed_binary_replay=False,
            packed_structured_replay=isinstance(self._advantage_memories[0], StructuredAdvantageReservoirBuffer))

    def _information_state(self, state, player):
        return self.feature_encoder.information_state(state, player)
