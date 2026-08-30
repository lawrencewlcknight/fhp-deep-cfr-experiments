"""Replay buffers and memory records for Deep CFR.

The reservoir buffer implements Algorithm R (Vitter, 1985): given a stream of
elements and a fixed capacity ``k``, every element seen so far is retained with
uniform probability ``k / n`` after ``n`` adds.

Both ``add`` and ``sample`` use Python's :mod:`random` module so that callers
seeding either ``random`` or ``numpy.random`` get a single, well-defined source
of randomness. This avoids the previous mix of ``np.random.randint`` and
``random.sample`` which made determinism harder to reason about.
"""

from __future__ import annotations

import collections
import random
from typing import Any, Iterator, List

import numpy as np


# ``info_state`` is the OpenSpiel information-state tensor for the player whose
# regret we are storing; ``iteration`` is the Deep CFR iteration index at which
# the sample was collected (used for linear-CFR weighting); ``advantage`` is the
# per-action sampled regret vector with zeros for illegal actions.
AdvantageMemory = collections.namedtuple(
    "AdvantageMemory", ["info_state", "iteration", "advantage"]
)

# ``strategy_action_probs`` is the regret-matched policy at the info state for
# the non-traversing player (the average policy target).
StrategyMemory = collections.namedtuple(
    "StrategyMemory", ["info_state", "iteration", "strategy_action_probs"]
)


class ReservoirBuffer:
    """Uniform reservoir sampling over a stream of replay data."""

    def __init__(self, reservoir_buffer_capacity: int) -> None:
        capacity = int(reservoir_buffer_capacity)
        if capacity <= 0:
            raise ValueError(
                f"ReservoirBuffer capacity must be positive, got {capacity}"
            )
        self._reservoir_buffer_capacity = capacity
        self._data: List[Any] = []
        self._add_calls = 0

    @property
    def capacity(self) -> int:
        return self._reservoir_buffer_capacity

    @property
    def add_calls(self) -> int:
        return self._add_calls

    def add(self, element: Any) -> None:
        """Adds ``element`` to the buffer with reservoir-sampling semantics."""
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            # Algorithm R: replace a uniformly chosen slot with probability
            # capacity / (add_calls + 1).
            idx = random.randint(0, self._add_calls)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples: int) -> List[Any]:
        """Returns ``num_samples`` distinct elements drawn uniformly at random."""
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if num_samples > len(self._data):
            raise ValueError(
                f"Cannot sample {num_samples} elements from a buffer of size "
                f"{len(self._data)}"
            )
        return random.sample(self._data, num_samples)

    def clear(self) -> None:
        self._data = []
        self._add_calls = 0

    def state_dict(self) -> dict:
        """Returns a checkpointable representation of the buffer."""
        return {
            "capacity": int(self._reservoir_buffer_capacity),
            "add_calls": int(self._add_calls),
            "data": list(self._data),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores buffer contents from :meth:`state_dict`."""
        capacity = int(state["capacity"])
        if capacity <= 0:
            raise ValueError(
                f"Loaded buffer capacity must be positive, got {capacity}"
            )
        self._reservoir_buffer_capacity = capacity
        self._add_calls = int(state["add_calls"])
        self._data = list(state["data"])
        if len(self._data) > self._reservoir_buffer_capacity:
            raise ValueError(
                "Loaded buffer has more elements than its capacity: "
                f"{len(self._data)} > {self._reservoir_buffer_capacity}"
            )

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)


class _CompactReservoirBuffer:
    """Array-backed reservoir for fixed-shape Deep CFR replay records.

    The original replay stores one Python namedtuple plus separate Python/
    NumPy objects per sample. At millions of samples this overhead dominates
    the actual tensor payload. This buffer keeps the same public interface but
    stores the payload in contiguous float32/int32 arrays and only materialises
    namedtuples for sampled minibatches.
    """

    def __init__(
        self,
        reservoir_buffer_capacity: int,
        *,
        info_state_size: int,
        target_size: int,
        record_type,
        target_attr: str,
    ) -> None:
        capacity = int(reservoir_buffer_capacity)
        if capacity <= 0:
            raise ValueError(
                f"Compact reservoir capacity must be positive, got {capacity}"
            )
        info_state_size = int(info_state_size)
        target_size = int(target_size)
        if info_state_size <= 0 or target_size <= 0:
            raise ValueError(
                "Compact reservoir dimensions must be positive: "
                f"info_state_size={info_state_size}, target_size={target_size}"
            )
        self._reservoir_buffer_capacity = capacity
        self._info_state_size = info_state_size
        self._target_size = target_size
        self._record_type = record_type
        self._target_attr = str(target_attr)
        self._allocate_arrays(capacity)
        self._size = 0
        self._add_calls = 0

    def _allocate_arrays(self, capacity: int) -> None:
        self._reservoir_buffer_capacity = int(capacity)
        self._info_states = np.empty(
            (self._reservoir_buffer_capacity, self._info_state_size),
            dtype=np.float32,
        )
        self._iterations = np.empty(self._reservoir_buffer_capacity, dtype=np.int32)
        self._targets = np.empty(
            (self._reservoir_buffer_capacity, self._target_size),
            dtype=np.float32,
        )

    @property
    def capacity(self) -> int:
        return self._reservoir_buffer_capacity

    @property
    def add_calls(self) -> int:
        return self._add_calls

    @property
    def compact(self) -> bool:
        return True

    def _record_at(self, index: int):
        return self._record_type(
            self._info_states[index].copy(),
            int(self._iterations[index]),
            self._targets[index].copy(),
        )

    def add(self, element: Any) -> None:
        """Adds ``element`` with Algorithm-R reservoir semantics."""
        write_index = None
        if self._size < self._reservoir_buffer_capacity:
            write_index = self._size
            self._size += 1
        else:
            idx = random.randint(0, self._add_calls)
            if idx < self._reservoir_buffer_capacity:
                write_index = idx

        if write_index is not None:
            info_state = np.asarray(element.info_state, dtype=np.float32).reshape(-1)
            target = np.asarray(
                getattr(element, self._target_attr), dtype=np.float32
            ).reshape(-1)
            if info_state.size != self._info_state_size:
                raise ValueError(
                    f"Expected info_state of size {self._info_state_size}, "
                    f"got {info_state.size}"
                )
            if target.size != self._target_size:
                raise ValueError(
                    f"Expected target of size {self._target_size}, got {target.size}"
                )
            self._info_states[write_index] = info_state
            self._iterations[write_index] = int(element.iteration)
            self._targets[write_index] = target
        self._add_calls += 1

    def sample(self, num_samples: int) -> List[Any]:
        """Returns ``num_samples`` distinct elements drawn uniformly at random."""
        num_samples = int(num_samples)
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if num_samples > self._size:
            raise ValueError(
                f"Cannot sample {num_samples} elements from a buffer of size "
                f"{self._size}"
            )
        indices = random.sample(range(self._size), num_samples)
        return [self._record_at(index) for index in indices]

    def sample_up_to(self, max_size: int) -> List[Any]:
        max_size = int(max_size)
        if max_size < 0:
            raise ValueError(f"max_size must be >= 0, got {max_size}")
        return self.sample(min(max_size, self._size))

    def target_abs_mean(self) -> np.ndarray:
        """Returns per-record mean absolute target magnitudes."""
        if self._size == 0:
            return np.empty(0, dtype=np.float32)
        return np.mean(np.abs(self._targets[: self._size]), axis=1)

    def sample_with_probabilities(
        self, probabilities: np.ndarray, batch_size: int
    ) -> List[Any]:
        if self._size == 0:
            return []
        indices = np.random.choice(
            self._size,
            size=int(batch_size),
            replace=False,
            p=np.asarray(probabilities, dtype=np.float64),
        )
        return [self._record_at(int(index)) for index in indices]

    def clear(self) -> None:
        self._size = 0
        self._add_calls = 0

    def state_dict(self) -> dict:
        """Returns a checkpointable representation without unused capacity."""
        return {
            "compact": True,
            "capacity": int(self._reservoir_buffer_capacity),
            "add_calls": int(self._add_calls),
            "size": int(self._size),
            "info_state_size": int(self._info_state_size),
            "target_size": int(self._target_size),
            "info_states": self._info_states[: self._size].copy(),
            "iterations": self._iterations[: self._size].copy(),
            "targets": self._targets[: self._size].copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        capacity = int(state["capacity"])
        if capacity <= 0:
            raise ValueError(
                f"Loaded buffer capacity must be positive, got {capacity}"
            )
        if state.get("compact"):
            info_states = np.asarray(state["info_states"], dtype=np.float32)
            targets = np.asarray(state["targets"], dtype=np.float32)
            iterations = np.asarray(state["iterations"], dtype=np.int32)
        else:
            data = list(state.get("data", []))
            info_states = np.asarray([row.info_state for row in data], dtype=np.float32)
            targets = np.asarray(
                [getattr(row, self._target_attr) for row in data], dtype=np.float32
            )
            iterations = np.asarray([row.iteration for row in data], dtype=np.int32)

        size = int(len(iterations))
        if size > capacity:
            raise ValueError(
                f"Loaded compact buffer has more elements than capacity: "
                f"{size} > {capacity}"
            )
        if info_states.reshape(size, -1).shape[1] != self._info_state_size:
            raise ValueError("Loaded info-state shape does not match this buffer.")
        if targets.reshape(size, -1).shape[1] != self._target_size:
            raise ValueError("Loaded target shape does not match this buffer.")

        if capacity != self._reservoir_buffer_capacity:
            self._allocate_arrays(capacity)
        self._size = size
        self._add_calls = int(state.get("add_calls", size))
        if size:
            self._info_states[:size] = info_states.reshape(size, self._info_state_size)
            self._targets[:size] = targets.reshape(size, self._target_size)
            self._iterations[:size] = iterations.reshape(size)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        for index in range(self._size):
            yield self._record_at(index)


class CompactAdvantageReservoirBuffer(_CompactReservoirBuffer):
    def __init__(
        self,
        reservoir_buffer_capacity: int,
        *,
        info_state_size: int,
        num_actions: int,
    ) -> None:
        super().__init__(
            reservoir_buffer_capacity,
            info_state_size=info_state_size,
            target_size=num_actions,
            record_type=AdvantageMemory,
            target_attr="advantage",
        )


class CompactStrategyReservoirBuffer(_CompactReservoirBuffer):
    def __init__(
        self,
        reservoir_buffer_capacity: int,
        *,
        info_state_size: int,
        num_actions: int,
    ) -> None:
        super().__init__(
            reservoir_buffer_capacity,
            info_state_size=info_state_size,
            target_size=num_actions,
            record_type=StrategyMemory,
            target_attr="strategy_action_probs",
        )


def _normalise_replay_buffer_type(buffer_type: str) -> str:
    normalised = str(buffer_type).lower()
    aliases = {
        "python": "python",
        "object": "python",
        "list": "python",
        "compact": "compact",
        "array": "compact",
        "numpy": "compact",
    }
    if normalised not in aliases:
        raise ValueError(
            f"Unknown replay_buffer_type={buffer_type!r}; expected 'python' "
            "or 'compact'."
        )
    return aliases[normalised]


def make_advantage_buffer(
    capacity: int,
    buffer_type: str,
    *,
    info_state_size: int,
    num_actions: int,
):
    if _normalise_replay_buffer_type(buffer_type) == "compact":
        return CompactAdvantageReservoirBuffer(
            capacity,
            info_state_size=info_state_size,
            num_actions=num_actions,
        )
    return ReservoirBuffer(capacity)


def make_strategy_buffer(
    capacity: int,
    buffer_type: str,
    *,
    info_state_size: int,
    num_actions: int,
):
    if _normalise_replay_buffer_type(buffer_type) == "compact":
        return CompactStrategyReservoirBuffer(
            capacity,
            info_state_size=info_state_size,
            num_actions=num_actions,
        )
    return ReservoirBuffer(capacity)
