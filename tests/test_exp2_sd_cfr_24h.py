import json
import random
from pathlib import Path

import numpy as np
import pyspiel
import pytest
import torch

from deep_cfr_poker.game import load_fhp_game
from deep_cfr_poker.sd_cfr import HistoricalSDCFRPolicy, exact_average_policy
from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive, DiskArchiveReader, DiskSampledPolicy, DiskBehaviouralPolicy
from deep_cfr_poker.sd_cfr_optimised import OptimisedSingleDeepCFRSolver
from deep_cfr_poker.single_solver import SingleDeepCFRSolver, SELECTED_SD_CFR_KWARGS
from deep_cfr_poker.seeding import set_seed
from experiments.fhp.exp2_sd_cfr_24h.config import SEEDS, HOURS, SECONDS, solver_config
from experiments.fhp.exp2_sd_cfr_24h.train import ActiveClock


def archive_fixture(tmp_path, game=None):
    game = game or pyspiel.load_game("kuhn_poker")
    set_seed(19)
    solver = SingleDeepCFRSolver(game, num_iterations=3, num_traversals=3,
                                memory_capacity=32, batch_size_advantage=4,
                                advantage_network_train_steps=1, evaluation_interval=1)
    disk = DiskSDCFRArchive(solver, tmp_path / "archive", chunk_iterations=2)
    memory = solver.archive
    solver.solve(post_player_update_callback=disk.capture_from_solver)
    path = disk.checkpoint(tmp_path / "archive" / "time_24h.json")
    return game, memory, disk, DiskArchiveReader(path, game)


def test_frozen_contract():
    assert SEEDS == (0, 1, 2)
    assert HOURS == (6, 12, 18, 24)
    assert SECONDS[-1] == 86400
    c = solver_config()
    assert all(c[key] == value for key, value in SELECTED_SD_CFR_KWARGS.items())
    assert c["num_iterations"] == 1_000_000


def test_archive_bit_exact_and_bounded(tmp_path):
    game, memory, disk, reader = archive_fixture(tmp_path)
    assert disk.pending == []
    assert len(disk.chunks) == 2
    for player in (0, 1):
        for entry in memory.entries_by_player[player]:
            for name, value in reader.weights(player, entry.iteration).items():
                assert torch.equal(value, entry.state_dict[name])
    prefix = disk.checkpoint(tmp_path / "archive" / "second_manifest.json")
    assert len(list(prefix.parent.glob("chunk_*.npy"))) == 2
    original = json.loads(prefix.read_text())
    original["chunks"][0]["sha256"] = "bad"
    prefix.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="integrity"):
        DiskArchiveReader(prefix, game)


def test_on_demand_behavioural_mixture_matches_exact_small_game(tmp_path):
    game, memory, _, reader = archive_fixture(tmp_path)
    exact = exact_average_policy(game, memory, weighting="uniform")
    online = DiskBehaviouralPolicy(reader, game, model_batch_size=2, cache_size=2)
    stack = [game.new_initial_state()]
    while stack:
        state = stack.pop()
        if state.is_terminal():
            continue
        if not state.is_chance_node():
            a, b = exact.action_probabilities(state), online.action_probabilities(state)
            assert a.keys() == b.keys()
            np.testing.assert_allclose(list(a.values()), list(b.values()), atol=2e-6, rtol=2e-6)
        for action in state.legal_actions():
            stack.append(state.child(action))
    assert len(online.cache) <= 2


def test_sampled_archive_policy_fixed_per_hand_and_rng_isolated(tmp_path):
    game, memory, _, reader = archive_fixture(tmp_path)
    before = torch.get_rng_state().clone()
    policy = DiskSampledPolicy(reader)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(1)
    with pytest.raises(RuntimeError, match="begin_episode"):
        policy.action_probabilities(state)
    policy.begin_episode(seed=91)
    iterations = dict(policy.selected_iterations)
    historic = HistoricalSDCFRPolicy(game, memory, iterations)
    assert policy.action_probabilities(state) == historic.action_probabilities(state)
    assert policy.action_probabilities(state) == historic.action_probabilities(state)
    assert policy.selected_iterations == iterations
    policy.begin_episode(seed=91)
    assert policy.selected_iterations == iterations


def test_clock_excludes_checkpoint_io_only():
    values = [10.0]
    clock = ActiveClock(lambda: values[0])
    values[0] += 5
    with clock.paused():
        values[0] += 100
    assert clock() == 5
    values[0] += 2
    assert clock() == 7 and clock.excluded == 100


def test_disk_archive_does_not_change_optimised_training(tmp_path):
    configurations = dict(num_iterations=3, num_traversals=4, memory_capacity=32,
                          batch_size_advantage=8, advantage_network_train_steps=1,
                          evaluation_interval=1)
    outputs = []
    for with_disk in (False, True):
        set_seed(94)
        solver = OptimisedSingleDeepCFRSolver(**configurations)
        if with_disk:
            solver.archive = DiskSDCFRArchive(solver, tmp_path / "disk", chunk_iterations=2)
        result = solver.solve()
        outputs.append((solver._nodes_touched, result.advantage_losses,
                        [n.state_dict() for n in solver._advantage_networks],
                        torch.get_rng_state().clone(), random.getstate(), np.random.get_state()))
    assert outputs[0][:2] == outputs[1][:2]
    for a, b in zip(outputs[0][2], outputs[1][2]):
        for key in a:
            assert torch.equal(a[key], b[key])
    assert torch.equal(outputs[0][3], outputs[1][3])
    assert outputs[0][4] == outputs[1][4]
    np.testing.assert_array_equal(outputs[0][5][1], outputs[1][5][1])


def test_duplicate_hand_calls_begin_episode_without_changing_stateless_play():
    from fhp_evaluation.duplicate import play_hand
    from open_spiel.python.policy import UniformRandomPolicy
    game = load_fhp_game()
    uniform = UniformRandomPolicy(game)
    class Tracking:
        def __init__(self):
            self.seeds = []
        def begin_episode(self, *, seed):
            self.seeds.append(seed)
        def action_probabilities(self, state, player_id=None):
            return uniform.action_probabilities(state, player_id)
    wrapped = Tracking()
    kwargs = dict(chance_seed=8, action_seed=12)
    assert play_hand(game, (wrapped, uniform), **kwargs) == play_hand(game, (uniform, uniform), **kwargs)
    assert len(wrapped.seeds) == 1


def test_time_budget_excludes_pauses_without_restarting_learner():
    values = [0.0]
    clock = ActiveClock(lambda: values[0])
    set_seed(10)
    solver = SingleDeepCFRSolver(pyspiel.load_game("kuhn_poker"), num_iterations=100,
                                num_traversals=1, advantage_network_train_steps=1,
                                batch_size_advantage=2, memory_capacity=16)
    visited = []
    def checkpoint(_solver, iteration):
        visited.append(iteration)
        values[0] += 1
        with clock.paused():
            values[0] += 1000
    solver.solve(post_iteration_callback=checkpoint, max_training_seconds=3, training_clock=clock)
    assert visited == [1, 2, 3]
    assert clock() == 3 and clock.excluded == 3000
    assert solver._iteration == 4


def test_standalone_production_evaluation_plan_preserves_protocol():
    from experiments.fhp.exp2_sd_cfr_24h.evaluate import make_tasks
    records = [dict(seed=s, training_hours=h, path=f"sd-{s}-{h}", sha256="abc", nodes_touched=1)
               for s in SEEDS for h in HOURS]
    tasks = make_tasks(records)
    assert len({t["task_id"] for t in tasks}) == len(tasks) == 1278
    assert sum(t["kind"] == "rule" for t in tasks) == 60
    assert sum(t["kind"] == "lbr" for t in tasks) == 1200
    assert sum(t["kind"] == "temporal" for t in tasks) == 18
    assert {t["kind"] for t in tasks} == {"rule", "lbr", "temporal"}
    assert {t["num_deals"] for t in tasks if t["kind"] == "temporal"} == {50000}
    assert {t["num_deals"] for t in tasks if t["kind"] == "rule"} == {10000}
    assert {t["num_deals"] for t in tasks if t["kind"] == "lbr"} == {10}
    assert {t["lbr_rollouts"] for t in tasks if t["kind"] == "lbr"} == {4096}
    # Identical rule/deal streams across checkpoints and all training seeds.
    assert {t["evaluation_seed"] for t in tasks if t.get("opponent") == "candid_statistician"} == {20360922}


def test_batch_contract_and_scripts_parse(tmp_path):
    import importlib.util
    import subprocess
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("builder", root / "gcp/exp2_sd_cfr_24h_batch.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    args = SimpleNamespace(project="test", region="europe-west1", bucket="gs://test",
                           service_account="runner@test.iam.gserviceaccount.com", repo_ref="a" * 40,
                           run_id="sdcfr2-test", start_stage="smoke", eval_max_hours=36)
    for stage in ("controller",) + builder.STAGES:
        job = builder.build_job(args, stage)
        group = job["taskGroups"][0]
        assert group["taskCount"] == (3 if stage == "train" else 1)
        assert group["parallelism"] == group["taskCount"]
        assert group["taskCountPerNode"] == 1
        assert group["taskSpec"]["maxRetryCount"] == 0  # no fake policy-only training resume
        path = tmp_path / f"{stage}.sh"
        script = group["taskSpec"]["runnables"][0]["script"]["text"]
        assert "ucv" not in script.lower()
        assert "reference-analysis" not in script
        path.write_text(script)
        subprocess.run(["bash", "-n", str(path)], check=True)
    train = builder.build_job(args, "train")
    assert train["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    assert train["taskGroups"][0]["taskSpec"]["maxRunDuration"] == "129600s"


@pytest.mark.parametrize("action,stage", [("run", "smoke"), ("evaluate-only", "profile")])
def test_launch_does_not_require_comparator_objects(monkeypatch, action, stage):
    import importlib.util
    import sys
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("standalone_builder", root / "gcp/exp2_sd_cfr_24h_batch.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    # Old terminal exports must not reintroduce comparator dependencies.
    monkeypatch.setenv("UCV_EXP1_RUN_ID", "nonexistent-ucv")
    monkeypatch.setenv("UCV_EVAL_RUN_ID", "nonexistent-evaluation")
    monkeypatch.setattr(sys, "argv", ["batch.py", action, "--project", "test", "--region", "europe-west1",
                                    "--bucket", "standalone-results", "--service-account", "runner@test",
                                    "--repo-ref", "a" * 40, "--run-id", "sdcfr2-test"])
    calls = []
    monkeypatch.setattr(builder, "cloud", lambda args, *command, **kwargs: calls.append(command))
    def submit(args, name, **kwargs):
        assert args.start_stage == stage
        assert args.bucket == "gs://standalone-results"
        assert not any(key.startswith("UCV_") for key in builder.environment(args))
        calls.append(("submit", name))
        return "test-controller"
    monkeypatch.setattr(builder, "submit", submit)
    builder.main()
    assert calls == [("iam", "service-accounts", "describe", "runner@test"), ("submit", "controller")]


def test_behavioural_mixture_is_not_pointwise_average_and_prefix_is_immutable(tmp_path):
    game = pyspiel.load_game("kuhn_poker")
    solver = SingleDeepCFRSolver(game, advantage_network_type="mlp", advantage_network_layers=(8,),
                                memory_capacity=8, num_iterations=2)
    disk = DiskSDCFRArchive(solver, tmp_path / "archive", chunk_iterations=1)
    memory = solver.archive
    for iteration, biases in enumerate(((1., 3.), (3., 1.)), start=1):
        for player, network in enumerate(solver._advantage_networks):
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.zero_()
                state = network.state_dict()
                bias_key = list(state)[-1]
                state[bias_key].copy_(torch.tensor(biases))
            memory.capture_from_solver(solver, player, iteration)
            disk.capture_from_solver(solver, player, iteration)
        disk.checkpoint(tmp_path / "archive" / f"prefix_{iteration}.json")
    early = DiskArchiveReader(tmp_path / "archive/prefix_1.json", game)
    late = DiskArchiveReader(tmp_path / "archive/prefix_2.json", game)
    assert early.count == 1 and late.count == 2
    assert len(early.chunks) == 1 and len(late.chunks) == 2
    state = game.new_initial_state()
    for action in (0, 1, 0, 1):
        state.apply_action(action)
    probability = DiskBehaviouralPolicy(late, game).action_probabilities(state)[0]
    assert probability == pytest.approx(.625, abs=1e-7)
    assert probability != pytest.approx(.5)  # naive average would be wrong
    assert exact_average_policy(game, memory, weighting="uniform").action_probabilities(state)[0] == pytest.approx(probability)
    assert DiskBehaviouralPolicy(early, game).action_probabilities(state)[0] == pytest.approx(.25)


def test_retrospective_deal_layout_matches_completed_ucv_analysis(monkeypatch):
    from fhp_evaluation import duplicate
    observed = []
    def fake_hand(game, policies, *, chance_seed, action_seed):
        observed.append((chance_seed, action_seed))
        return (1., -1.)
    monkeypatch.setattr(duplicate, "play_hand", fake_hand)
    duplicate.evaluate_duplicate_match(None, None, None, num_deals=3, seed=20360922, seed_layout="split")
    rng = np.random.default_rng(20360922)
    chance = rng.integers(0, 2**63 - 1, size=3, dtype=np.int64)
    actions = rng.integers(0, 2**63 - 1, size=3, dtype=np.int64)
    assert observed == [tuple(pair) for pair in zip(chance, actions) for _ in range(2)]


def test_standalone_tables_and_charts_without_comparator_data(tmp_path):
    import csv
    from experiments.fhp.exp2_sd_cfr_24h.report import evaluation_report
    from experiments.fhp.exp2_sd_cfr_24h.evaluate import make_tasks
    records = [dict(experiment="sd_cfr_exp2", seed=s, training_hours=h, path=f"sd-{s}-{h}",
                    sha256="abc", nodes_touched=100000 * h) for s in SEEDS for h in HOURS]
    results = [dict(task=task, result=dict(mean_mbb_per_hand=10.), elapsed_seconds=.1)
               for task in make_tasks(records)]
    output = tmp_path / "analysis"
    output.mkdir()
    evaluation_report(results, records, output)
    with (output / "quality_aggregate.csv").open() as stream:
        aggregate = list(csv.DictReader(stream))
    assert len(aggregate) == 8
    assert {r["experiment"] for r in aggregate} == {"sd_cfr_exp2"}
    assert {r["n"] for r in aggregate} == {"3"}
    with (output / "lbr_by_seed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
        assert len(rows) == 12
        assert {r["num_deals"] for r in rows} == {"1000"}
    with (output / "rule_agent_by_seed.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 60
    with (output / "temporal_crossplay_by_seed.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 18
    assert (output / "temporal_head_to_head.png").exists()
    assert (output / "policy_quality_by_training_hours.png").exists()
    assert (output / "policy_quality_by_mean_nodes.png").exists()
    assert not list(output.glob("*ucv*"))
    assert not list(output.glob("direct_*"))
    with pytest.raises(ValueError, match="standalone"):
        evaluation_report([dict(task=dict(kind="direct"))], records, output)


def test_production_profile_and_evaluation_cli_need_only_own_checkpoints(tmp_path, monkeypatch):
    import sys
    from experiments.fhp.exp2_sd_cfr_24h import evaluate
    records = [dict(experiment="sd_cfr_exp2", seed=s, training_hours=h, path=f"sd-{s}-{h}",
                    sha256="abc", nodes_touched=100000 * h) for s in SEEDS for h in HOURS]
    source = tmp_path / "own_checkpoints"
    output = tmp_path / "evaluation"
    def index(root, *, smoke=False, experiment=None):
        assert root == source and not smoke
        return records
    def run_tasks(tasks, output, *, workers):
        assert workers == 8
        assert {task["kind"] for task in tasks} == {"rule", "lbr", "temporal"}
        assert all(task["path_a"].startswith("sd-") for task in tasks)
        return [dict(task=task, result=dict(mean_mbb_per_hand=10.), elapsed_seconds=.001,
                     lbr_backend_validation=dict(passed=True))
                for task in tasks]
    monkeypatch.setattr(evaluate, "checkpoint_index", index)
    monkeypatch.setattr(evaluate, "run_tasks", run_tasks)
    for mode in ("profile", "run"):
        monkeypatch.setattr(sys, "argv", ["evaluate.py", mode, "--source", str(source), "--output", str(output)])
        evaluate.main()
    profile = json.loads((output / "evaluation_profile.json").read_text())
    assert profile["passed"]
    assert set(profile["seconds_per_pair"]) == {"rule", "lbr", "temporal"}
    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["evaluation_scope"] == "standalone_sd_cfr"
    assert manifest["tasks"] == 1278
    assert manifest["source_checkpoints"] == records
