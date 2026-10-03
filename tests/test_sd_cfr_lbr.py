"""Numerical and protocol regression gates against the original shared LBR."""
import random

import numpy as np
import pytest
import torch

from deep_cfr_poker.game import load_fhp_game
from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive, DiskArchiveReader, DiskBehaviouralPolicy, DiskSampledPolicy
from deep_cfr_poker.sd_cfr_lbr import BatchedDiskBehaviouralPolicy, ExactSDCFRLocalBestResponsePolicy
from deep_cfr_poker.single_solver import SingleDeepCFRSolver
from deep_cfr_poker.sd_cfr_structured import StructuredSingleDeepCFRSolver
from deep_cfr_poker.seeding import set_seed
from fhp_evaluation.lbr import LocalBestResponsePolicy, LBRConfig, _policy_probability
from fhp_evaluation.duplicate import evaluate_duplicate_match


def fixture_archive(root, structured=False):
    torch.set_num_threads(1)
    set_seed(821)
    game = load_fhp_game()
    solver_type = StructuredSingleDeepCFRSolver if structured else SingleDeepCFRSolver
    solver = solver_type(game, advantage_network_type="residual_layer_norm_centered_advantage_mlp",
                         advantage_network_layers=(8, 8), memory_capacity=8, num_iterations=3)
    disk = DiskSDCFRArchive(solver, root, chunk_iterations=2)
    # Deliberately distinct models and nonzero legal mass on every action across
    # the archive; exercise own-reach weighting, not an identical-model fixture.
    for iteration in range(1, 4):
        for player, network in enumerate(solver._advantage_networks):
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.add_(torch.randn_like(parameter) * .15)
                list(network.parameters())[-1].add_(torch.tensor([iteration, 4-iteration, 2.]) * .1)
            disk.capture_from_solver(solver, player, iteration)
    path = disk.checkpoint(root / "policy.json")
    return game, DiskArchiveReader(path, game)


def state_from(game, actions):
    state = game.new_initial_state()
    for action in actions:
        assert action in state.legal_actions()
        state.apply_action(action)
    return state


HISTORIES = [
    [0, 4, 8, 12],
    [0, 4, 8, 12, 2],
    [0, 4, 8, 12, 2, 2, 2],  # raise cap: reach only, no post-raise query
    [0, 4, 8, 12, 1, 1, 16, 20, 32],
    [0, 4, 8, 12, 1, 1, 16, 20, 32, 2],
]


@pytest.mark.parametrize("structured", [False, True])
def test_batched_probabilities_and_telescoping_reach(tmp_path, structured):
    game, reader = fixture_archive(tmp_path / "archive", structured)
    scalar = DiskBehaviouralPolicy(reader, game, model_batch_size=2)
    batch = BatchedDiskBehaviouralPolicy(reader, game, model_batch_size=2, state_batch_size=2, cache_size=2)
    states = [state_from(game, history) for history in HISTORIES]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    for player in (0, 1):
        selected = [s for s in states if s.current_player() == player]
        reaches, distributions = batch.batch_reach_and_probabilities(selected, player)
        for state, reach, distribution in zip(selected, reaches, distributions):
            expected = scalar.action_probabilities(state)
            np.testing.assert_allclose(distribution[state.legal_actions()], list(expected.values()), atol=2e-6, rtol=2e-6)
            cursor, likelihood = game.new_initial_state(), 1.
            for action in state.history():
                if not cursor.is_chance_node() and cursor.current_player() == player:
                    likelihood *= _policy_probability(scalar, cursor, player, action)
                cursor.apply_action(action)
            assert reach == pytest.approx(likelihood, abs=2e-6, rel=2e-6)
        # Non-acting-player reach uses the same own sequence without asking for
        # an illegal action distribution at a responder decision.
        reach_only, _ = batch.batch_reach_and_probabilities(selected, player, include_action=False)
        np.testing.assert_allclose(reach_only, reaches, atol=2e-6, rtol=2e-6)
    assert random.getstate() == rng[0]
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    assert torch.equal(torch.get_rng_state(), rng[2])
    assert len(batch._reach_cache) <= 2


@pytest.mark.parametrize("structured", [False, True])
def test_full_ranges_folds_equity_and_actions_match_shared_lbr(tmp_path, structured):
    game, reader = fixture_archive(tmp_path / "archive", structured)
    config = LBRConfig(seed=21760922, preflop_rollout_samples=32)
    scalar = LocalBestResponsePolicy(game, DiskBehaviouralPolicy(reader, game), config=config)
    batched = ExactSDCFRLocalBestResponsePolicy(game, BatchedDiskBehaviouralPolicy(reader, game), config=config)
    for history in HISTORIES:
        state = state_from(game, history)
        player = state.current_player()
        old, new = scalar._opponent_range(state, player), batched._opponent_range(state, player)
        assert [h.hand for h in old] == [h.hand for h in new]
        np.testing.assert_allclose([h.weight for h in old], [h.weight for h in new], atol=2e-6, rtol=2e-5)
        for salt in ("call",):
            assert scalar._equity(state, player, old, salt=salt) == pytest.approx(batched._equity(state, player, new, salt=salt), abs=2e-6)
        if 2 in state.legal_actions():
            old_fold, old_cont = scalar._raise_fold_range(old, player)
            new_fold, new_cont = batched._raise_fold_range(new, player)
            assert old_fold == pytest.approx(new_fold, abs=2e-6)
            assert [h.hand for h in old_cont] == [h.hand for h in new_cont]
            np.testing.assert_allclose([h.weight for h in old_cont], [h.weight for h in new_cont], atol=2e-6, rtol=2e-5)
        assert scalar.action_probabilities(state) == batched.action_probabilities(state)
    # Replaying the same information sets incurs no further network inference.
    batches = batched.target_policy.stats["network_batches"]
    batched._opponent_range(state, player)
    assert batched.target_policy.stats["network_batches"] == batches
    assert batched._choose_action.__func__ is scalar._choose_action.__func__
    assert batched._equity.__func__ is scalar._equity.__func__


def test_no_hidden_opponent_or_sampled_model_information(tmp_path):
    game, reader = fixture_archive(tmp_path / "archive", True)
    policy = BatchedDiskBehaviouralPolicy(reader, game)
    lbr = ExactSDCFRLocalBestResponsePolicy(game, policy, config=LBRConfig(preflop_rollout_samples=8))
    first = state_from(game, [0, 4, 8, 12])
    second = state_from(game, [0, 4, 24, 28])
    assert first.current_player() == second.current_player() == 0
    a, b = lbr._opponent_range(first, 0), lbr._opponent_range(second, 0)
    assert [(h.hand, h.weight) for h in a] == [(h.hand, h.weight) for h in b]
    with pytest.raises(TypeError, match="all-model"):
        ExactSDCFRLocalBestResponsePolicy(game, DiskSampledPolicy(reader))


def test_duplicate_returns_unchanged(tmp_path):
    game, reader = fixture_archive(tmp_path / "archive", True)
    results = []
    for target_type, lbr_type in [(DiskBehaviouralPolicy, LocalBestResponsePolicy),
                                 (BatchedDiskBehaviouralPolicy, ExactSDCFRLocalBestResponsePolicy)]:
        responder = lbr_type(game, target_type(reader, game), config=LBRConfig(seed=37, preflop_rollout_samples=32))
        result = evaluate_duplicate_match(game, responder, DiskSampledPolicy(reader), num_deals=2,
                                          seed=41, seed_layout="split").to_dict()
        results.append(result)
    assert results[0] == results[1]


def test_zero_advantages_and_impossible_sequence_fallbacks(tmp_path):
    game = load_fhp_game()
    for name, biases in [("uniform", [0., 0., 0.]), ("fold_only", [1., 0., 0.])]:
        solver = SingleDeepCFRSolver(game, advantage_network_type="mlp", advantage_network_layers=(4,),
                                    memory_capacity=4, num_iterations=1)
        disk = DiskSDCFRArchive(solver, tmp_path / name, chunk_iterations=1)
        for player, network in enumerate(solver._advantage_networks):
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.zero_()
                list(network.parameters())[-1].copy_(torch.tensor(biases))
            disk.capture_from_solver(solver, player, 1)
        reader = DiskArchiveReader(disk.checkpoint(tmp_path / name / "policy.json"), game)
        scalar = DiskBehaviouralPolicy(reader, game)
        batch = BatchedDiskBehaviouralPolicy(reader, game)
        state = state_from(game, HISTORIES[-1])
        assert batch.action_probabilities(state) == scalar.action_probabilities(state)
        if name == "fold_only":
            for policy_type, lbr_type in [(DiskBehaviouralPolicy, LocalBestResponsePolicy),
                                         (BatchedDiskBehaviouralPolicy, ExactSDCFRLocalBestResponsePolicy)]:
                with pytest.raises(RuntimeError, match="collapsed"):
                    impossible = state_from(game, HISTORIES[2])
                    lbr_type(game, policy_type(reader, game))._opponent_range(impossible, impossible.current_player())


def test_evaluation_uses_optimised_lbr_and_full_archive_gate(tmp_path):
    from deep_cfr_poker.sd_cfr_disk import sha256
    from experiments.fhp.exp2_sd_cfr_24h.evaluate import execute_task
    game, reader = fixture_archive(tmp_path / "archive", True)
    row = execute_task(dict(kind="lbr", path_a=str(reader.path), sha_a=sha256(reader.path),
                            lbr_seed=37, lbr_rollouts=8, num_deals=1, evaluation_seed=41,
                            validate_lbr_backend=True))
    assert row["lbr_backend_validation"]["passed"]
    assert row["lbr_backend_validation"]["models_per_player"] == reader.count == 3
    assert row["total_elapsed_seconds_including_validation"] >= row["elapsed_seconds"]


def test_profile_cannot_pass_without_numerical_gate(tmp_path, monkeypatch):
    from experiments.fhp.exp2_sd_cfr_24h import evaluate
    tasks = [dict(kind="lbr", training_seed=0, training_hours=24, task_id="probe", num_deals=1000)]
    monkeypatch.setattr(evaluate, "run_tasks", lambda *a, **k: [dict(task=tasks[0], elapsed_seconds=.001)])
    with pytest.raises(RuntimeError, match="validation"):
        evaluate.profile(tasks, tmp_path, workers=8, max_hours=36)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution requires a GPU validation pilot")
@pytest.mark.parametrize("structured", [False, True])
def test_cuda_equivalence_gate_and_lbr_actions(tmp_path, structured):
    from deep_cfr_poker.sd_cfr_lbr_audit import validate_queries
    game, reader = fixture_archive(tmp_path / "archive", structured)
    batched = BatchedDiskBehaviouralPolicy(reader, game, device="cuda", device_cache_bytes=4096)
    assert validate_queries(reader, game, batched=batched)["passed"]
    assert not torch.backends.cuda.matmul.allow_tf32
    scalar = LocalBestResponsePolicy(game, DiskBehaviouralPolicy(reader, game), config=LBRConfig(preflop_rollout_samples=32))
    fast = ExactSDCFRLocalBestResponsePolicy(game, batched, config=scalar.config)
    for history in HISTORIES:
        state = state_from(game, history)
        assert scalar.action_probabilities(state) == fast.action_probabilities(state)
    assert batched._device_weight_bytes <= 4096


def test_execution_device_is_explicit_and_fingerprinted(tmp_path, monkeypatch):
    from experiments.fhp.exp2_sd_cfr_24h.evaluate import make_tasks, task_fingerprint
    records = [dict(seed=0, training_hours=h, path="policy", sha256="a", nodes_touched=10) for h in (6, 24)]
    cpu = make_tasks(records, smoke=True)
    gpu = make_tasks(records, smoke=True, lbr_device="cuda")
    assert task_fingerprint(cpu) != task_fingerprint(gpu)
    game, reader = fixture_archive(tmp_path / "archive")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        BatchedDiskBehaviouralPolicy(reader, game, device="cuda")


def test_gpu_batch_is_opt_in_and_never_changes_training(tmp_path):
    from types import SimpleNamespace
    import subprocess
    from gcp import exp2_sd_cfr_24h_batch as builder
    args = SimpleNamespace(project="test", region="europe-west1", bucket="gs://test", run_id="sdcfr2-test",
                           repo_ref="a" * 40, service_account="runner@test", start_stage="profile",
                           eval_max_hours=36, eval_lbr="1", eval_lbr_device="cuda", eval_workers=2, profile_only=True)
    for stage in ("train", "profile", "evaluate", "controller"):
        job = builder.build_job(args, stage)
        instance = job["allocationPolicy"]["instances"][0]
        script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
        if stage in {"profile", "evaluate"}:
            assert instance["policy"]["machineType"] == "g2-standard-8"
            assert instance["installGpuDrivers"]
            assert "--workers 2" in script and "--lbr-device cuda" in script
            assert "torch==2.7.0+cu126" in script
            assert "pip install -r requirements-dev.txt" not in script
        else:
            assert not instance.get("installGpuDrivers")
            assert "torch==2.7.0+cu126" not in script
        if stage == "train":
            assert instance["policy"]["machineType"] == "n2-standard-8"
            assert job["taskGroups"][0]["taskCount"] == 3
        path = tmp_path / f"{stage}.sh"
        path.write_text(script)
        subprocess.run(["bash", "-n", str(path)], check=True)
    args.eval_lbr_device, args.eval_workers = "cpu", 8
    assert builder.build_job(args, "profile")["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"


def test_profile_only_controller_cannot_launch_full_evaluation(monkeypatch):
    import sys
    from gcp import exp2_sd_cfr_24h_batch as builder
    monkeypatch.setattr(sys, "argv", ["batch.py", "orchestrate", "--profile-only", "--project", "test",
                                    "--region", "europe-west1", "--bucket", "results", "--service-account", "runner@test",
                                    "--repo-ref", "a" * 40, "--run-id", "sdcfr2-test", "--eval-lbr-device", "cuda"])
    stages = []
    monkeypatch.setattr(builder, "cloud", lambda *a, **k: None)
    monkeypatch.setattr(builder, "submit", lambda args, stage, **k: stages.append(stage) or "job")
    monkeypatch.setattr(builder, "wait", lambda *a, **k: None)
    builder.main()
    assert stages == ["profile"]
