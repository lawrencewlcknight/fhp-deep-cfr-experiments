import itertools
import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deep_cfr_poker.fhp_features import FHPFeatureEncoder, FEATURE_SIZE, _five_card_category
from deep_cfr_poker.game import load_fhp_game
from deep_cfr_poker.sd_cfr import HistoricalSDCFRPolicy, SDCFRArchive
from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive, DiskArchiveReader, DiskSampledPolicy, DiskBehaviouralPolicy
from deep_cfr_poker.sd_cfr_structured import StructuredSingleDeepCFRSolver, StructuredAdvantageReservoirBuffer
from deep_cfr_poker.seeding import set_seed
from experiments.fhp.exp2_sd_cfr_24h import config as baseline
from experiments.fhp.exp3_sd_cfr_structured_24h import config


FIXTURE = Path(__file__).parent / "fixtures/ucv_exp2_information_states.json"


def replay(history, suit_permutation=None):
    state = load_fhp_game().new_initial_state()
    for action in history:
        if state.is_chance_node() and suit_permutation is not None:
            action = (action // 4) * 4 + suit_permutation[action % 4]
        assert action in state.legal_actions()
        state.apply_action(action)
    return state


def golden_rows():
    fixture = json.loads(FIXTURE.read_text())
    for row in fixture["records"]:
        expected = np.zeros(FEATURE_SIZE, dtype=np.float32)
        expected[row["indices"]] = row["values"]
        yield row, expected


def small_config():
    return dict(config.solver_config(True), num_iterations=3, num_traversals=4)


def test_only_input_representation_changes_scientific_contract():
    assert config.solver_config() == baseline.solver_config()
    assert config.solver_config(True) == baseline.solver_config(True)
    assert config.SEEDS == baseline.SEEDS == (0, 1, 2)
    assert config.HOURS == baseline.HOURS == (6, 12, 18, 24)
    assert config.SECONDS[-1] == 86400
    assert config.REFERENCE_VM == baseline.REFERENCE_VM
    assert config.REFERENCE_VM["machine_type"] == "n2-standard-8"
    assert config.EXPERIMENT_NAME != baseline.EXPERIMENT_NAME
    assert config.task_name(1) == "task_001_structured_uniform_sd_cfr_seed_1"
    assert FHPFeatureEncoder().metadata() == json.loads(FIXTURE.read_text())["metadata"]
    assert not hasattr(FHPFeatureEncoder(), "full_state")


def test_exact_float32_parity_with_original_ucv_exp2_encoder():
    encoder = FHPFeatureEncoder()
    categories = set()
    for row, expected in golden_rows():
        state = replay(row["history"])
        actual = encoder.information_state(state, row["player"])
        assert actual.dtype == np.float32 and actual.shape == (183,)
        np.testing.assert_array_equal(actual, expected)
        if actual[-9:].sum():
            categories.add(int(actual[-9:].argmax()))
    assert categories == set(range(9))


def test_all_global_suit_permutations_preserve_features():
    encoder = FHPFeatureEncoder(cache_entries=2)
    for row, expected in golden_rows():
        for permutation in itertools.permutations(range(4)):
            state = replay(row["history"], permutation)
            np.testing.assert_array_equal(encoder.information_state(state, row["player"]), expected)
    assert len(encoder._information_cache) == 2


def test_private_opponent_cards_never_enter_player_features():
    encoder = FHPFeatureEncoder()
    # Same private cards for player 0, public board and actions, different opponent cards.
    histories = []
    for opponent in ([8, 12], [24, 28]):
        state = load_fhp_game().new_initial_state()
        cards = iter([0, 4] + opponent + [16, 20, 32])
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(next(cards))
            else:
                if np.asarray(state.information_state_tensor(0))[54:106].sum() == 3:
                    break
                state.apply_action(1)
        histories.append(state)
    assert histories[0].history() != histories[1].history()
    np.testing.assert_array_equal(encoder.information_state(histories[0], 0), encoder.information_state(histories[1], 0))
    # Rank changes must remain distinguishable, unlike suit-name changes.
    a, b = replay([0, 4, 8, 12]), replay([0, 16, 8, 12])
    assert not np.array_equal(encoder.information_state(a, 0), encoder.information_state(b, 0))


def test_structured_replay_is_lossless_and_bounded():
    buffer = StructuredAdvantageReservoirBuffer(20, info_state_size=183, num_actions=3)
    features = np.stack([value for _, value in golden_rows()])
    encoded = buffer._pack(features)
    assert encoded.dtype == np.uint8 and encoded.shape == (84, 46)
    np.testing.assert_array_equal(buffer._unpack(encoded), features)
    assert buffer._info_states.nbytes == 20 * 46
    batch = dict(info_states=features, iterations=np.arange(len(features)) + 1,
                 targets=np.zeros((len(features), 3), dtype=np.float32))
    buffer.add_batch(batch)
    payload = buffer.state_dict()
    restored = StructuredAdvantageReservoirBuffer(1, info_state_size=183, num_actions=3)
    restored.load_state_dict(payload)
    np.testing.assert_array_equal(buffer.as_batch()["info_states"], restored.as_batch()["info_states"])
    set_seed(44)
    sample = buffer.sample_batch(10)
    set_seed(44)
    same = restored.sample_batch(10)
    np.testing.assert_array_equal(sample["info_states"], same["info_states"])
    assert sample["info_states"].shape == (10, 183)
    invalid = features[0].copy()
    invalid[0] = .5
    with pytest.raises(ValueError, match="alphabet"):
        buffer._pack(invalid)
    payload["feature_encoding"] = "binary_packbits_little_v1"
    with pytest.raises(ValueError, match="Unknown"):
        restored.load_state_dict(payload)


def test_structured_packed_and_float_replay_produce_identical_training():
    outputs = []
    for packed in (False, True):
        set_seed(15)
        solver = StructuredSingleDeepCFRSolver(pack_replay=packed, **small_config())
        assert solver._embedding_size == 183
        assert solver._advantage_network_layers == (32,) * 8
        result = solver.solve()
        outputs.append((solver._nodes_touched, result.advantage_losses,
                        [network.state_dict() for network in solver._advantage_networks],
                        torch.get_rng_state().clone(), random.getstate(), np.random.get_state()))
    assert outputs[0][:2] == outputs[1][:2]
    for a, b in zip(outputs[0][2], outputs[1][2]):
        for key in a:
            assert torch.equal(a[key], b[key])
    assert torch.equal(outputs[0][3], outputs[1][3])
    assert outputs[0][4] == outputs[1][4]
    np.testing.assert_array_equal(outputs[0][5][1], outputs[1][5][1])


def test_archived_structured_policies_use_encoder_and_exact_own_reach(tmp_path):
    set_seed(91)
    solver = StructuredSingleDeepCFRSolver(**small_config())
    memory = solver.archive
    disk = DiskSDCFRArchive(solver, tmp_path / "archive", chunk_iterations=2)
    solver.solve(post_player_update_callback=disk.capture_from_solver)
    path = disk.checkpoint(tmp_path / "archive/time_24h.json")
    reader = DiskArchiveReader(path, solver._game)
    policy = DiskSampledPolicy(reader)
    policy.begin_episode(seed=18)
    fixed = HistoricalSDCFRPolicy(solver._game, memory, policy.selected_iterations)
    mixture = DiskBehaviouralPolicy(reader, solver._game, model_batch_size=2)
    # Includes multiple own actions before the queried decision and all card categories.
    for row, _ in list(golden_rows())[3::4]:
        state = replay(row["history"])
        player = state.current_player()
        np.testing.assert_allclose(list(policy.action_probabilities(state).values()),
                                   list(fixed.action_probabilities(state).values()), atol=1e-7)
        numerator = np.zeros(3)
        denominator = 0.
        for iteration in range(1, disk.count + 1):
            historic = HistoricalSDCFRPolicy(solver._game, memory, {0: iteration, 1: iteration})
            cursor, reach = solver._game.new_initial_state(), 1.
            for action in state.history():
                if not cursor.is_chance_node() and cursor.current_player() == player:
                    reach *= historic.action_probabilities(cursor)[action]
                cursor.apply_action(action)
            for action, probability in historic.action_probabilities(state).items():
                numerator[action] += reach * probability
            denominator += reach
        legal = state.legal_actions()
        expected = numerator[legal] / denominator if denominator > 0 else np.full(len(legal), 1 / len(legal))
        np.testing.assert_allclose(list(mixture.action_probabilities(state).values()), expected, atol=2e-6, rtol=2e-6)
    saved = tmp_path / "in_memory.pkl"
    memory.save(saved)
    reloaded = SDCFRArchive.load(saved)
    reloaded.validate_game(solver._game)
    assert reloaded.feature_encoder.metadata() == config.FEATURE_ENCODER_METADATA
    original = json.loads(path.read_text())
    original["metadata"]["feature_encoder"]["version"] = 99
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="encoder"):
        DiskArchiveReader(path, solver._game)
    original["metadata"].pop("feature_encoder")
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="representation"):
        DiskArchiveReader(path, solver._game)


def test_new_cloud_pipeline_is_standalone_and_keeps_baseline_resources(tmp_path):
    from gcp import exp2_sd_cfr_24h_batch as base
    from gcp import exp3_sd_cfr_structured_24h_batch as candidate
    arguments = dict(project="test", region="europe-west1", bucket="gs://test",
                     service_account="runner@test", repo_ref="a" * 40,
                     run_id="sdcfr3-test", start_stage="smoke", eval_max_hours=36)
    for stage in ("controller",) + candidate.STAGES:
        job = candidate.build_job(SimpleNamespace(**arguments), stage)
        previous = base.build_job(SimpleNamespace(**arguments), stage)
        assert job["allocationPolicy"] == previous["allocationPolicy"]
        group = job["taskGroups"][0]
        assert group["taskCount"] == group["parallelism"] == (3 if stage == "train" else 1)
        assert group["taskCountPerNode"] == 1
        assert group["taskSpec"]["computeResource"] == previous["taskGroups"][0]["taskSpec"]["computeResource"]
        assert group["taskSpec"]["maxRunDuration"] == previous["taskGroups"][0]["taskSpec"]["maxRunDuration"]
        script = group["taskSpec"]["runnables"][0]["script"]["text"]
        assert "ucv-source" not in script and "UCV_EXP" not in script
        if stage == "controller":
            assert "gcp/exp3_sd_cfr_structured_24h_batch.py orchestrate" in script
        else:
            assert "experiments.fhp.exp3_sd_cfr_structured_24h" in script
        if stage == "train":
            assert "structured_uniform_sd_cfr_seed_" in script
        if stage == "smoke":
            assert "tests/test_exp3_sd_cfr_structured.py" in script
        path = tmp_path / f"{stage}.sh"
        path.write_text(script)
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert job["labels"]["experiment"] == "fhp-sdcfr-exp3-24h"
        assert previous["labels"]["experiment"] == "fhp-sdcfr-exp2-24h"
