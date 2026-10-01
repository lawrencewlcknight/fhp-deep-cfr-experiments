"""Opt-in, algorithm-preserving SD-CFR execution candidates.

Deliberately separate from the reference solver: Experiment 1 must establish
equivalence and speed before these become production defaults. No sampler,
optimiser, floating-point precision, archive schedule or diagnostics change.
"""

import numpy as np
import torch

from .replay import CompactAdvantageReservoirBuffer
from .single_solver import SingleDeepCFRSolver


class PackedAdvantageReservoirBuffer(CompactAdvantageReservoirBuffer):
    """Losslessly pack binary FHP inputs; targets remain float32.

All reservoir and minibatch decisions are delegated to the original buffer,
including its Python RNG consumption. Only sampled features are unpacked.
Non-binary inputs fail closed, including records the reservoir would discard.
This encoding must not be used as quantisation for other game features.
"""

    def __init__(self, capacity, *, info_state_size, num_actions):
        self.feature_count = int(info_state_size)
        if self.feature_count <= 0:
            raise ValueError("info_state_size must be positive")
        super().__init__(capacity, info_state_size=(self.feature_count + 7) // 8,
                         num_actions=num_actions)

    def _allocate_arrays(self, capacity):
        self._reservoir_buffer_capacity = int(capacity)
        self._info_states = np.empty((capacity, self._info_state_size), dtype=np.uint8)
        self._iterations = np.empty(capacity, dtype=np.int32)
        self._targets = np.empty((capacity, self._target_size), dtype=np.float32)

    def _pack(self, features):
        values = np.asarray(features)
        if values.ndim not in (1, 2) or values.shape[-1] != self.feature_count:
            raise ValueError("Packed replay input feature shape mismatch")
        if not np.all((values == 0) | (values == 1)):
            raise ValueError("Packed replay requires exactly binary finite features")
        return np.packbits(values.astype(np.uint8), axis=-1, bitorder="little")

    def _unpack(self, features):
        return np.unpackbits(features, axis=-1, count=self.feature_count,
                             bitorder="little").astype(np.float32)

    def add(self, element):
        super().add(element._replace(info_state=self._pack(element.info_state)))

    def add_batch(self, batch):
        super().add_batch({**batch, "info_states": self._pack(batch["info_states"])})

    def _record_at(self, index):
        record = super()._record_at(index)
        return record._replace(info_state=self._unpack(record.info_state))

    def as_batch(self, *, copy=False):
        batch = super().as_batch(copy=copy)
        batch["info_states"] = self._unpack(batch["info_states"])
        return batch

    def sample_batch(self, num_samples, *, probabilities=None):
        batch = super().sample_batch(num_samples, probabilities=probabilities)
        batch["info_states"] = self._unpack(batch["info_states"])
        return batch

    def state_dict(self):
        return {**super().state_dict(), "feature_encoding": "binary_packbits_little_v1",
                "original_info_state_size": self.feature_count}

    def load_state_dict(self, state):
        encoding = state.get("feature_encoding")
        if encoding == "binary_packbits_little_v1":
            if int(state["original_info_state_size"]) != self.feature_count:
                raise ValueError("Packed replay feature count mismatch")
            values = np.asarray(state["info_states"])
            if (values.ndim != 2 or values.shape[1] != self._info_state_size
                    or values.dtype != np.uint8):
                raise ValueError("Invalid packed replay payload")
            adapted = state
        elif encoding is not None:
            raise ValueError(f"Unknown replay feature encoding: {encoding}")
        elif state.get("compact"):
            adapted = {**state, "info_states": self._pack(state["info_states"])}
        else:
            adapted = {**state, "data": [row._replace(info_state=self._pack(row.info_state))
                                         for row in state.get("data", [])]}
        # The reference loader cannot reshape an empty (0, -1) array.
        size = len(adapted["iterations"]) if adapted.get("compact") else len(adapted["data"])
        if size == 0:
            capacity = int(adapted["capacity"])
            if capacity <= 0:
                raise ValueError("Replay capacity must be positive")
            self._allocate_arrays(capacity)
            self._size = 0
            self._add_calls = int(adapted.get("add_calls", 0))
            return
        super().load_state_dict(adapted)


class OptimisedSingleDeepCFRSolver(SingleDeepCFRSolver):
    """Live scripted traversal inference, optionally with packed replay.

Training and archiving use the original eager modules. Scripted modules share
their parameters, so player 1 traversals see player 0's just-completed update.
No freeze/trace is used. Unsupported configurations fail rather than silently
falling back and mislabelling a benchmark arm.
"""

    def __init__(self, game=None, *, pack_replay=True, **kwargs):
        super().__init__(game, **kwargs)
        if (self._uses_shared_advantage_trunk or self._reinitialize_advantage_networks
                or self._replay_buffer_type != "compact"):
            raise ValueError("Optimised SD-CFR requires independent warm-start networks and compact replay")
        if pack_replay:
            self._advantage_memories = [
                PackedAdvantageReservoirBuffer(buffer.capacity,
                                               info_state_size=self._embedding_size,
                                               num_actions=self._num_actions)
                for buffer in self._advantage_memories
            ]
        self._eager_advantages = self._advantage_networks
        self._scripted_advantages = [torch.jit.script(net) for net in self._eager_advantages]
        for eager, scripted in zip(self._eager_advantages, self._scripted_advantages):
            left, right = eager.state_dict(), scripted.state_dict()
            if left.keys() != right.keys() or any(
                left[key].data_ptr() != right[key].data_ptr() for key in left
            ):
                raise RuntimeError("Scripted traversal weights must share live learner storage")
        self.archive.metadata["execution_optimisations"] = {
            "scripted_live_inference": True, "packed_binary_replay": bool(pack_replay),
        }

    def _collect_traversals_for_player(self, player):
        if self._advantage_networks is not self._eager_advantages:
            raise RuntimeError("Advantage module replacement invalidated compiled inference")
        self._advantage_networks = self._scripted_advantages
        try:
            super()._collect_traversals_for_player(player)
        finally:
            self._advantage_networks = self._eager_advantages
