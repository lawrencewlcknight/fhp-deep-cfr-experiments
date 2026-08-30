from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_stale_variant_references():
    forbidden = "le" + "duc"
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
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
    assert experiment_children == ["exp1_deep_cfr_best_config_transfer"]

    output_children = sorted(path.name for path in (ROOT / "outputs").iterdir())
    assert output_children == [".gitkeep"]
    assert not (ROOT / "cloud_outputs").exists()
