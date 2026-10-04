"""Protocol, inference, integrity, cloud plan and real-loader integration tests."""
import argparse
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.retrospective_sd5_ucv9_evaluation import run as audit
from gcp import sd5_ucv9_head_to_head_batch as batch


def records():
    return [dict(algorithm=a, seed=s, training_hours=h, path=f"/{a}/{s}/{h}",
                 sha256=f"{a}-{s}-{h}", nodes_touched=h*10000, active_seconds=h*3600,
                 outer_iteration=h) for a, s, h in itertools.product(("sd", "ucv"), audit.SEEDS, audit.HOURS)]


def result(task, mean=1.0, std=2.0):
    n = task["num_deals"]
    row = dict(task=audit.portable(task), elapsed_seconds=1., result=dict(
        num_deal_pairs=n, num_games=n*2, mean_chips_per_hand=mean, std_chips_per_pair=std,
        se_chips_per_hand=std/np.sqrt(n), mean_mbb_per_hand=10*mean,
        policy_a_player0_mean_chips=mean+1, policy_a_player1_mean_chips=mean-1))
    row["result_sha256"] = audit.digest(row)
    return row


def test_schedule_all_cross_seed_pairs_and_prespecified_budgets():
    tasks = audit.build_tasks(records(), "code")
    assert len(tasks) == 630
    assert len({t["task_id"] for t in tasks}) == 630
    assert len({t["cell_id"] for t in tasks}) == 54
    assert sum(t["num_deals"] for t in tasks) == 3_150_000
    assert len({t["evaluation_seed"] for t in tasks}) == 630
    assert sum(t["num_deals"] for t in tasks if t["kind"] == "primary") == 900_000
    for kind, a, b, n in audit.COMPARISONS:
        rows = [t for t in tasks if (t["kind"], t["sd"]["training_hours"], t["ucv"]["training_hours"]) == (kind,a,b)]
        assert {(r["sd"]["seed"], r["ucv"]["seed"]) for r in rows} == set(itertools.product(range(3),repeat=2))
        assert sum(r["num_deals"] for r in rows) == 9*n
    assert {(a,b) for kind,a,b,n in audit.COMPARISONS if kind == "approximate_nodes"} == {(6,12),(12,24)}


def test_probes_have_disjoint_randomness_and_cover_every_match():
    production = audit.build_tasks(records(), "code")
    for stage, n in (("profile",128), ("smoke",2)):
        probes = audit.build_tasks(records(), "code", stage=stage)
        assert len(probes) == 54 and {p["num_deals"] for p in probes} == {n}
        assert {p["cell_id"] for p in probes} == {t["cell_id"] for t in production}
        assert not {p["evaluation_seed"] for p in probes} & {t["evaluation_seed"] for t in production}


def test_invalid_source_index_and_escape(tmp_path):
    with pytest.raises(ValueError, match="three distinct"):
        audit.build_tasks(records()[:-1], "code")
    bad = records(); bad[0] = bad[1]
    with pytest.raises(ValueError):
        audit.validate_index(bad)
    with pytest.raises(ValueError, match="escapes"):
        audit.contained(tmp_path, "../outside.pkl")


def test_corrupt_or_mismatched_shard_fails_closed():
    task = audit.build_tasks(records(), "code")[0]
    row = result(task)
    audit.validate_result(row, task)
    row["result"]["mean_mbb_per_hand"] += 1
    with pytest.raises(ValueError, match="Corrupt"):
        audit.validate_result(row, task)
    for mutate in (lambda t:t.update(implementation="new"), lambda t:t["sd"].update(sha256="new"),
                   lambda t:t.update(num_deals=1)):
        changed = deepcopy(task); mutate(changed)
        with pytest.raises(ValueError):
            audit.validate_result(result(task), changed)
    moved = deepcopy(task); moved["sd"]["path"] = "/new/root"
    audit.validate_result(result(task), moved)


def test_resume_never_launches_completed_shards(tmp_path, monkeypatch):
    from deep_cfr_poker.sd_cfr_disk import write_json
    tasks = audit.build_tasks(records(), "code")[:2]
    for t in tasks:
        write_json(tmp_path / (t["task_id"] + ".json"), result(t))
    monkeypatch.setattr(audit, "ProcessPoolExecutor", lambda **kw: pytest.fail("Unexpected worker launch"))
    assert len(audit.run_tasks(tasks,tmp_path,workers=8,ucv_repo="unused")) == 2
    with pytest.raises(TimeoutError):
        audit.run_tasks(audit.build_tasks(records(),"code")[:3],tmp_path,workers=8,ucv_repo="unused",deadline=0)


def test_pooled_variance_matches_unsharded_samples():
    samples = (np.array([1.,3.,7.]),np.array([-2.,8.,20.,22.]))
    task = audit.build_tasks(records(),"code")[0]
    rows = [result(dict(task,num_deals=len(x)),float(x.mean()),float(x.std(ddof=1))) for x in samples]
    pooled = audit.pool_shards(rows)
    full = np.concatenate(samples)
    assert pooled["mean_mbb_per_hand"] == pytest.approx(10*full.mean())
    assert pooled["mc_se_mbb_per_hand"] == pytest.approx(10*full.std(ddof=1)/np.sqrt(len(full)))
    assert pooled["hands"] == 14


def test_bootstrap_resamples_both_training_seed_axes_not_nine_cells():
    matrix = np.array([[0,0,0],[10,10,10],[20,20,20.]])
    lo,hi = audit.cluster_interval(matrix, draws=100_000)
    assert (lo,hi) == (0,20)  # Three clusters, not nine independent observations.
    assert audit.cluster_interval(np.ones((3,3))) == (1,1)
    assert audit.cluster_interval(matrix) == audit.cluster_interval(matrix)
    with pytest.raises(ValueError):
        audit.cluster_interval(np.ones((2,3)))


def test_timing_gate_estimates_remaining_work():
    tasks = audit.build_tasks(records(), "code")
    probes = [result(t) for t in audit.build_tasks(records(),"code",stage="profile")]
    for p in probes:
        p["elapsed_seconds"] = .025 * p["task"]["num_deals"]
    cost = audit.cost_estimate(probes,tasks,8)
    assert cost["predicted_hours_with_2x_margin"] == pytest.approx((2*3_150_000*.025/8+600)/3600)
    assert audit.cost_estimate(probes,tasks[:1],8)["remaining_duplicate_pairs"] == 5000
    with pytest.raises(ValueError, match="Missing"):
        audit.cost_estimate(probes[:-1],tasks,8)
    for p in probes:
        p["elapsed_seconds"] *= 20
    assert audit.cost_estimate(probes,tasks,8)["predicted_hours_with_2x_margin"] > 12


def test_main_cost_gate_never_launches_production_when_over_budget(tmp_path,monkeypatch):
    import sys
    output=tmp_path/"output"
    monkeypatch.setattr(sys,"argv",["run","--sd-root",str(tmp_path/"sd"),"--ucv-root",str(tmp_path/"ucv"),
        "--ucv-repo",str(tmp_path/"loader"),"--output",str(output)])
    monkeypatch.setattr(audit,"validate_sources",lambda *args:records())
    monkeypatch.setattr(audit,"implementation_digest",lambda *args:"implementation")
    def slow_profile(tasks,*args,**kwargs):
        assert {t["stage"] for t in tasks} == {"profile"}, "Must not launch production"
        rows=[result(t) for t in tasks]
        for row in rows: row["elapsed_seconds"]=1_000_000
        return rows
    monkeypatch.setattr(audit,"run_tasks",slow_profile)
    with pytest.raises(RuntimeError,match="Timing pilot exceeds"):
        audit.main()
    assert audit.read_json(output/"timing_pilot.json")["passed"] is False
    assert not (output/"SUCCESS.json").exists()


def test_reporting_and_incomplete_budget_rejection(tmp_path):
    rows = [result(t, mean=t["sd"]["seed"]-t["ucv"]["seed"]+.2) for t in audit.build_tasks(records(),"code")]
    summaries = audit.report(rows,tmp_path)
    assert len(summaries) == 6
    assert summaries[0]["mean_mbb_per_hand"] == pytest.approx(2)
    assert len(list(tmp_path.glob("*.png"))) == 3
    assert sum(r["duplicate_pairs"] for r in summaries) == 3_150_000
    with pytest.raises(ValueError,match="Incomplete"):
        audit.report(rows[:-1], tmp_path)
    with pytest.raises(ValueError,match="Incomplete production"):
        audit.report([result(t) for t in audit.build_tasks(records(),"code",stage="smoke")],tmp_path)


def cloud_args():
    return argparse.Namespace(project="project", region="europe-west1",run_id="sd5-ucv9-test",
        sd_run_id="sdcfr5-par8-20261002-102757",ucv_run_id="exp9-cache24-20261001-132550",
        bucket="gs://output",sd_bucket="gs://sd-source",ucv_bucket="gs://ucv-source",repo_ref="a"*40,
        ucv_ref=batch.DEFAULT_UCV_REF,max_hours=12,service_account="runner@test.iam.gserviceaccount.com",resume=False)


def test_cloud_plan_one_vm_and_only_playable_policy_inputs(tmp_path):
    config = batch.job_config(cloud_args())
    assert config["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    spec = config["taskGroups"][0]["taskSpec"]
    assert spec["maxRetryCount"] == 0 and spec["maxRunDuration"] == "50400s"
    text = spec["runnables"][0]["script"]["text"]
    assert "SD_SOURCE=gs://sd-source/" in text and "UCV_SOURCE=gs://ucv-source/" in text
    assert "training_states" not in text and "reservoir" not in text
    assert "--stage smoke" in text and "--stage run" in text
    assert "--workers 8" in text and "--max-hours 12" in text
    script = tmp_path / "job.sh"; script.write_text(text)
    subprocess.run(["bash","-n",str(script)],check=True)
    args = cloud_args(); args.run_id = args.sd_run_id
    with pytest.raises(ValueError): batch.job_config(args)
    args = cloud_args(); args.max_hours = 36
    with pytest.raises(ValueError): batch.job_config(args)


def test_cloud_access_errors_are_not_interpreted_as_absent_objects(monkeypatch):
    args = cloud_args()
    monkeypatch.setattr(batch,"cloud",lambda *a,**k:SimpleNamespace(returncode=1,stdout="",stderr="Permission denied"))
    with pytest.raises(RuntimeError,match="Permission"):
        batch.objects_exist(args,"gs://output/run/**")
    monkeypatch.setattr(batch,"cloud",lambda *a,**k:SimpleNamespace(returncode=1,stdout="",stderr="One or more URLs matched no objects."))
    assert not batch.objects_exist(args,"gs://output/run/**")


@pytest.fixture
def tiny_sources(tmp_path,monkeypatch):
    """Real archive and UCV loaders, tiny synthetic policies, no model training."""
    import torch
    from deep_cfr_poker.game import load_fhp_game,serialisable_game_definition
    from deep_cfr_poker.networks import build_network
    from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive,sha256,write_json
    from experiments.fhp.exp5_sd_cfr_parallel_24h import config
    ucv_repo = os.environ.get("UCV_TEST_REPO")
    if not ucv_repo:
        pytest.skip("Set UCV_TEST_REPO to exercise the separately pinned real UCV loader")
    loader = audit.setup_ucv(ucv_repo)
    from fhp_escher.features import encoder_from_metadata,StructuredFHPMLP
    torch.set_num_threads(1)
    encoder = encoder_from_metadata(config.FEATURE_ENCODER_METADATA)
    game = load_fhp_game()
    policy = StructuredFHPMLP(encoder.policy_layout,[8,8],3,branch_width=4)
    fake_solver = SimpleNamespace(ave_policy_trainer=SimpleNamespace(model=policy),feature_encoder=encoder,
        infostate_size=183,action_size=3,network_layers=[8,8])
    ucv_config = {"cache_frozen_critic_targets":True}
    monkeypatch.setattr(audit,"UCV_CONFIG_SHA256",audit.digest(ucv_config))
    sd_root, ucv_root = tmp_path / "sd", tmp_path / "ucv"
    for seed in audit.SEEDS:
        for algo, root, name, identifier, commit in (
            ("sd",sd_root,config.EXPERIMENT_NAME,config.ALGORITHM_ID,audit.SD_SOURCE_COMMIT),
            ("ucv",ucv_root,"exp9_fhp_cached_parallel_24h","cached_parallel_structured_ucv_escher",audit.UCV_SOURCE_COMMIT)):
            worker = root/"workers"/f"task_{seed:03d}"
            manifest = dict(seed=seed,smoke=False,experiment_name=name,algorithm_id=identifier,
                repository_commit=commit,game=serialisable_game_definition(),feature_encoder=config.FEATURE_ENCODER_METADATA,
                reference_vm={"machine_type":"n2-standard-16"})
            if algo == "sd":
                manifest.update(config=config.solver_config(),execution=config.execution_config(seed),parallel_execution={"seed":seed})
                model = build_network("mlp",183,[8],3)
                solver = SimpleNamespace(_num_players=2,_num_actions=3,_embedding_size=183,_game=game,
                    _advantage_network_type="mlp",_advantage_network_layers=[8],_advantage_networks=[model,model],
                    archive=SimpleNamespace(metadata=dict(feature_encoder=config.FEATURE_ENCODER_METADATA,
                        parallel_execution=manifest["parallel_execution"],solver_config=config.solver_config())))
                disk = DiskSDCFRArchive(solver,worker/"archive")
            else:
                manifest["training_config"] = ucv_config
                write_json(worker/"runtime_manifest.json",dict(frozen_critic_target_cache=True,traversal_execution="ray_parallel"))
            write_json(worker/"run_manifest.json",manifest)
            write_json(worker/"SUCCESS.json",dict(status="complete",checkpoints=4))
            rows = []
            for i,h in enumerate(audit.HOURS,1):
                row = dict(checkpoint_target_hours=h,checkpoint_target_seconds=float(h*3600),
                           actual_training_elapsed_seconds=h*3600+1,outer_iteration=i,nodes_touched=i*100,
                           algorithm_id=identifier,experiment_name=name,training_elapsed_seconds=h*3600+1)
                if algo == "sd":
                    for p in range(2): disk.capture_from_solver(solver,p,i)
                    file = disk.checkpoint(worker/"archive"/f"time_{h:02d}h.json")
                else:
                    fake_solver.num_iteration=i; fake_solver.episode=i; fake_solver.nodes_touched=i*100
                    file = loader.save_policy_checkpoint(fake_solver,worker/"checkpoints"/f"time_{h:02d}h.pkl",
                                seed=seed,config=ucv_config,checkpoint_row=row)
                row.update(path=str(file.relative_to(worker)),sha256=sha256(file))
                rows.append(row)
            write_json(worker/"checkpoint_manifest.json",rows)
    return sd_root,ucv_root,Path(ucv_repo)


def test_real_loader_integration_reproducible_shards_and_resume(tiny_sources,tmp_path):
    sd,ucv,repo=tiny_sources
    index=audit.validate_sources(sd,ucv,repo)
    assert len(index)==24
    tasks=audit.build_tasks(index,"fixture",stage="smoke")[:2]
    output=tmp_path/"matches"
    first=audit.run_tasks(tasks,output,workers=2,ucv_repo=repo)
    second=audit.run_tasks(tasks,output,workers=2,ucv_repo=repo)
    assert sorted(first,key=lambda r:r["task"]["task_id"])==sorted(second,key=lambda r:r["task"]["task_id"])
    # Fresh execution reproduces scientific outputs; timings need not match.
    audit._POLICIES.clear()
    assert audit.execute_task(tasks[0])["result"] == second[0]["result"]


def test_source_metadata_and_hash_rejections(tiny_sources):
    from deep_cfr_poker.sd_cfr_disk import write_json
    sd,ucv,repo=tiny_sources
    file=ucv/"workers/task_000/run_manifest.json"
    manifest=audit.read_json(file)
    changed=deepcopy(manifest);changed["game"]["parameters"]["numRanks"]=6
    write_json(file,changed)
    with pytest.raises(ValueError,match="game"):
        audit.validate_sources(sd,ucv,repo)
    write_json(file,manifest)
    checkpoint=ucv/"workers/task_000/checkpoint_manifest.json"
    rows=audit.read_json(checkpoint);rows[0]["sha256"]="bad";write_json(checkpoint,rows)
    with pytest.raises(ValueError,match="hash mismatch"):
        audit.validate_sources(sd,ucv,repo)
