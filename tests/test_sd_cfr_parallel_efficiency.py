"""Regression tests for output-preserving parallel execution optimisations."""
import copy
import random

import numpy as np
import pytest
import torch

from deep_cfr_poker.fhp_features import (
    DENOMINATORS, FEATURE_SIZE, _canonicalise_suits, _STRAIGHT_PRESENT,
)
from deep_cfr_poker.sd_cfr_optimised import PackedAdvantageReservoirBuffer, OptimisedSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_structured import StructuredAdvantageReservoirBuffer, StructuredSingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_parallel import _FlatWeights, SDCFRTraversalWorker
from deep_cfr_poker.seeding import set_seed
from tests.test_sd_cfr_parallel import SMALL, CLASSES, training_signature
from experiments.fhp.exp1_sd_cfr_efficiency.run import compare_trees


def test_straight_lookup_equals_original_rule_for_every_rank_pattern():
    for pattern in range(8192):
        present = [(pattern >> rank) & 1 for rank in range(13)]
        old = any(all(present[start:start + 5]) for start in range(9))
        old = old or all(present[rank] for rank in (12, 0, 1, 2, 3))
        assert bool(_STRAIGHT_PRESENT[pattern]) == bool(old)


def test_vectorised_suit_order_is_exact_including_ties():
    rng = np.random.default_rng(18)
    for _ in range(200):
        channels = [rng.integers(0, 2, size=(4, 13)).astype(np.float32) for _ in range(2)]
        for channel in channels:
            channel[1] = channel[0]
        old_order = sorted(range(4), key=lambda suit: tuple(
            float(value) for channel in channels for value in channel[suit]), reverse=True)
        actual = _canonicalise_suits(*channels)
        for value, channel in zip(actual, channels):
            np.testing.assert_array_equal(value, channel[old_order])
            assert not np.shares_memory(value, channel)


@pytest.mark.parametrize("rows", [0, 1, 2048])
def test_two_bit_codec_matches_original_packbits_and_unpackbits(rows):
    rng = np.random.default_rng(8)
    values = rng.integers(0, DENOMINATORS.astype(int) + 1, size=(rows, FEATURE_SIZE)).astype(np.float32)
    values /= DENOMINATORS
    buffer = StructuredAdvantageReservoirBuffer(1, info_state_size=183, num_actions=3)
    numerators = np.rint(values * DENOMINATORS).astype(np.uint8)
    old_bits = ((numerators[..., None] >> np.array([0, 1], dtype=np.uint8)) & 1)
    old_encoded = np.packbits(old_bits.reshape(rows, 366), axis=-1, bitorder="little")
    actual = buffer._pack(values)
    np.testing.assert_array_equal(actual, old_encoded)
    np.testing.assert_array_equal(buffer._unpack(actual), values)
    if rows:
        np.testing.assert_array_equal(buffer._pack(values[0]), old_encoded[0])
        np.testing.assert_array_equal(buffer._unpack(actual[0]), values[0])


@pytest.mark.parametrize("cls,width", [(PackedAdvantageReservoirBuffer, 190),
                                     (StructuredAdvantageReservoirBuffer, 183)])
def test_packed_batch_validation_does_not_expand_or_copy_codes(cls, width):
    buffer = cls(8, info_state_size=width, num_actions=3)
    encoded = buffer._pack(np.zeros((4, width), dtype=np.float32))
    encoded.flags.writeable = False  # Ray supplies read-only object-store views.
    batch = dict(info_states=encoded, targets=np.zeros((4, 3), dtype=np.float32),
                 iterations=np.ones(4, dtype=np.int32))
    validated = buffer._validated_batch(batch)[0]
    assert validated.dtype == np.uint8 and np.shares_memory(validated, encoded)
    buffer.add_packed_batch(batch, feature_encoding=buffer.feature_encoding)
    assert buffer.add_calls == 4


@pytest.mark.parametrize("cls", [OptimisedSingleDeepCFRSolver, StructuredSingleDeepCFRSolver])
def test_solver_allocates_packed_replay_directly(cls, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Do not allocate a dense advantage replay first")
    monkeypatch.setattr("deep_cfr_poker.solver.make_advantage_buffer", forbidden)
    solver = cls(**SMALL)
    assert all(b._info_states.dtype == np.uint8 for b in solver.advantage_buffers)


@pytest.mark.parametrize("structured", [False, True])
def test_cache_and_flat_transport_match_legacy_workers_across_weight_changes(structured):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        workers = [SDCFRTraversalWorker(structured=structured, solver_kwargs=SMALL,
                    worker_index=1, run_seed=45, chunk_rows=11,
                    inference_cache_entries=capacity) for capacity in (0, 4096)]
        source = workers[0].solver
        transport = _FlatWeights(source._eager_advantages)
        hits = 0
        previous_payload = None
        for iteration, player in ((1, 0), (1, 1), (2, 0)):
            # Change the weights between phases so stale caching cannot pass.
            with torch.no_grad():
                for network in source._eager_advantages:
                    for param in network.parameters():
                        param.add_(.001)
            old_weights = [copy.deepcopy(n.state_dict()) for n in source._eager_advantages]
            new_weights = transport.snapshot()
            before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            old = workers[0].collect(80, player, old_weights, iteration)
            new = workers[1].collect(80, player, new_weights, iteration)
            assert compare_trees(before, (random.getstate(), np.random.get_state(), torch.get_rng_state()))["exact"]
            hits += new["inference_cache_hits"]
            for row in (old, new):
                for key in ("collection_seconds", "inference_cache_hits", "inference_cache_misses"):
                    row.pop(key)
            assert compare_trees(old, new)["exact"]
            assert all(w.solver._phase_cache is None for w in workers)
            assert np.shares_memory(workers[1].collectors[0].features, workers[1].collectors[1].features)
            if previous_payload:
                assert compare_trees(*previous_payload)["exact"]
            previous_payload = (new, copy.deepcopy(new))
        assert hits > 0
    finally:
        torch.set_num_threads(previous_threads)


def test_flat_snapshot_immutable_and_validated_before_mutation():
    source = OptimisedSingleDeepCFRSolver(**SMALL)
    target = OptimisedSingleDeepCFRSolver(**SMALL)
    encoder, decoder = _FlatWeights(source._eager_advantages), _FlatWeights(target._eager_advantages)
    snapshot = encoder.snapshot()
    expected = copy.deepcopy(snapshot)
    with torch.no_grad():
        for param in source._eager_advantages[0].parameters():
            param.add_(1)
    assert compare_trees(snapshot, expected)["exact"]
    decoder.load(snapshot)
    assert compare_trees(decoder.snapshot(), snapshot)["exact"]
    corrupt = copy.deepcopy(snapshot)
    corrupt["arrays"][1] = corrupt["arrays"][1][:-1]
    with pytest.raises(ValueError, match="shape/dtype"):
        decoder.load(corrupt)
    assert compare_trees(decoder.snapshot(), snapshot)["exact"]


@pytest.mark.parametrize("cls", CLASSES)
def test_cache_changes_no_training_outputs(cls):
    signatures = []
    for capacity in (0, 3, 4096):
        set_seed(71)
        with cls(parallel_backend="serial", parallel_inference_cache_entries=capacity, **SMALL) as solver:
            signatures.append(training_signature(solver, solver.solve()))
    assert compare_trees(signatures[0], signatures[1])["exact"]
    assert compare_trees(signatures[0], signatures[2])["exact"]
