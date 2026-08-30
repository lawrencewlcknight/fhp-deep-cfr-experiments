from __future__ import annotations

import csv
import json

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.fhp.exp1_deep_cfr_best_config_transfer.config import smoke_config
from experiments.fhp.exp1_deep_cfr_best_config_transfer.run import run_experiment


@pytest.mark.smoke
def test_exp1_smoke_writes_snapshots_checkpoint_and_manifests(tmp_path):
    outcome = run_experiment(
        config=smoke_config(),
        seeds=[1234],
        output_root=tmp_path,
    )

    assert outcome["status"] == 0
    run_dir = outcome["run_dir"]
    snapshots = sorted((run_dir / "snapshots").glob("*_snapshot.pt"))
    checkpoints = sorted((run_dir / "checkpoints").glob("*_full.pt"))
    assert len(snapshots) == 2
    assert len(checkpoints) == 1

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["game"]["name"] == "FHP"
    assert manifest["game"]["parameters"]["numBoardCards"] == "0 3"
    assert manifest["completed_seeds"] == [1234]
    assert manifest["evaluation"]["exact_full_tree"] is False

    with open(run_dir / "policy_snapshot_manifest.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["checkpoint_iteration"]) for row in rows] == [1, 2]

    for filename in (
        "seed_summary.csv",
        "checkpoint_curves.csv",
        "aggregate_summary.json",
        "experiment_metadata.json",
        "multiseed_curves.npz",
    ):
        assert (run_dir / filename).exists()
