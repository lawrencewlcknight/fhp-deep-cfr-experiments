from __future__ import annotations

import random

import numpy as np

from deep_cfr_poker.replay import (
    AdvantageMemory,
    CompactAdvantageReservoirBuffer,
    ReservoirBuffer,
    append_batch_to_buffer,
)


def _batch(count: int, info_state_size: int = 4, target_size: int = 3):
    return {
        "info_states": np.arange(
            count * info_state_size, dtype=np.float32
        ).reshape(count, info_state_size),
        "iterations": np.arange(1, count + 1, dtype=np.int32),
        "targets": np.arange(count * target_size, dtype=np.float32).reshape(
            count, target_size
        ),
    }


def test_compact_add_batch_matches_seeded_scalar_algorithm_r():
    payload = _batch(30)
    scalar = CompactAdvantageReservoirBuffer(
        7, info_state_size=4, num_actions=3
    )
    batched = CompactAdvantageReservoirBuffer(
        7, info_state_size=4, num_actions=3
    )

    random.seed(9182)
    for info_state, iteration, target in zip(
        payload["info_states"], payload["iterations"], payload["targets"]
    ):
        scalar.add(AdvantageMemory(info_state, int(iteration), target))

    random.seed(9182)
    batched.add_batch(payload)

    assert scalar.add_calls == batched.add_calls == 30
    assert len(scalar) == len(batched) == 7
    scalar_arrays = scalar.as_batch()
    batched_arrays = batched.as_batch()
    for key in ("info_states", "iterations", "targets"):
        np.testing.assert_array_equal(scalar_arrays[key], batched_arrays[key])


def test_compact_sample_batch_returns_typed_arrays_without_records():
    buffer = CompactAdvantageReservoirBuffer(
        20, info_state_size=4, num_actions=3
    )
    buffer.add_batch(_batch(12))

    random.seed(44)
    sampled = buffer.sample_batch(5)

    assert sampled["info_states"].shape == (5, 4)
    assert sampled["iterations"].shape == (5,)
    assert sampled["targets"].shape == (5, 3)
    assert sampled["info_states"].dtype == np.float32
    assert sampled["iterations"].dtype == np.int32
    assert sampled["targets"].dtype == np.float32


def test_typed_batch_can_fall_back_to_python_replay():
    payload = _batch(5)
    buffer = ReservoirBuffer(10)

    append_batch_to_buffer(buffer, payload, record_type=AdvantageMemory)

    assert len(buffer) == 5
    records = list(buffer)
    np.testing.assert_array_equal(
        records[3].info_state, payload["info_states"][3]
    )
    np.testing.assert_array_equal(records[3].advantage, payload["targets"][3])
    assert records[3].iteration == 4
