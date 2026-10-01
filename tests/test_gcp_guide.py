from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "GCP_BATCH_EXPERIMENTS.md"


def test_archived_gcp_guide_preserves_the_old_workflow():
    text = (ROOT / "docs" / "archive_GCP_DEEP_CFR_EXPERIMENT1.md").read_text(encoding="utf-8")
    required = (
        "gcloud services enable",
        "roles/batch.agentReporter",
        "roles/batch.jobsEditor",
        "roles/iam.serviceAccountUser",
        "roles/logging.logWriter",
        "roles/storage.objectAdmin",
        "gcp/submit_batch_experiment.sh",
        "archive_exp1_deep_cfr_best_config_transfer.run",
        "gcloud batch jobs list",
        "gcloud batch jobs describe",
        "batch_task_logs",
        "gcloud storage cp --recursive",
        "n2-highmem-8",
        "345600",
        "pd-balanced",
    )
    missing = [entry for entry in required if entry not in text]
    assert missing == []


def test_active_gcp_guide_uses_renumbered_sd_cfr_audit():
    text = GUIDE.read_text(encoding="utf-8")
    assert "gcp/run_exp1_sd_cfr_efficiency.sh run" in text
    assert "fhp-sdcfr-exp1-" in text
    assert "n2-standard-8" in text
    assert "two hours" in text
    assert "gcloud storage rsync" in text


def test_readme_links_to_gcp_batch_guide():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/GCP_BATCH_EXPERIMENTS.md" in readme
