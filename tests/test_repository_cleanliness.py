from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_stale_variant_references():
    forbidden = "le" + "duc"
    offenders = []
    # These documents intentionally record the requested small-game port's
    # provenance; runtime game routing must still remain FHP-specific.
    provenance_documents = {ROOT / "README.md", ROOT / "docs" / "SD_CFR.md"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path in provenance_documents:
            continue
        if path.suffix.lower() in {".pyc", ".png", ".jpg", ".pt", ".npz"}:
            continue
        if forbidden in path.read_text(encoding="utf-8", errors="ignore").lower():
            offenders.append(path.relative_to(ROOT))
    assert offenders == []


def test_only_approved_experiment_and_no_outputs():
    experiment_children = sorted(
        path.name
        for path in (ROOT / "experiments" / "fhp").iterdir()
        if path.name not in {"__init__.py", "__pycache__"}
    )
    assert experiment_children == ["archive_exp1_deep_cfr_best_config_transfer", "exp1_sd_cfr_efficiency", "exp2_sd_cfr_24h", "exp3_sd_cfr_structured_24h", "exp4_sd_cfr_structured_n2_standard16"]

    output_children = sorted(path.name for path in (ROOT / "outputs").iterdir())
    assert output_children == [".gitkeep"]
    assert not (ROOT / "cloud_outputs").exists()
