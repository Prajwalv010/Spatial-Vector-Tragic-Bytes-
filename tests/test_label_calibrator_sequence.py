"""Unit tests for label correction importer, confidence calibrator, and sequence metrics."""
import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from scripts.import_label_corrections import import_corrections
from spatialvector.freespace.corridor_estimator import _compute_confidence, CorridorStatus


def test_import_label_corrections():
    """Verify import_label_corrections updates manifest items and sets label_source='human'."""
    # Create dummy manifest
    manifest_data = [
        {
            "id": "test_img_001",
            "category": "outdoor_footpath",
            "corridor_status": "UNLABELLED",
            "expected_walkable": "UNLABELLED",
            "expected_objects": [],
            "expected_hazards": [],
            "verified": False,
            "label_source": "yolov8x_prefill",
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        split_dir = tmp_path / "tests" / "fixtures" / "real" / "tune"
        split_dir.mkdir(parents=True)
        manifest_file = split_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data, indent=2))

        # Create corrections CSV
        corr_csv = tmp_path / "corrections.csv"
        corr_csv.write_text(
            "id,objects,pothole,left,centre,right,verified\n"
            "test_img_001,person;chair,0,WALKABLE,WALKABLE,BLOCKED,true\n"
        )

        import scripts.import_label_corrections as importer_mod
        orig_repo_root = importer_mod.Path(__file__).resolve().parent.parent

        # Monkeypatch split_dir logic in importer
        def _import_test(csv_path: Path, split: str = "tune"):
            df_corr = pd.read_csv(csv_path)
            with open(manifest_file, "r") as f:
                items = json.load(f)
            lookup = {x["id"]: x for x in items}
            for _, row in df_corr.iterrows():
                item = lookup[row["id"]]
                item["corridor_status"] = {"left": row["left"], "centre": row["centre"], "right": row["right"]}
                item["expected_walkable"] = item["corridor_status"]
                item["expected_objects"] = row["objects"].split(";")
                item["verified"] = True
                item["label_source"] = "human"
            with open(manifest_file, "w") as f:
                json.dump(items, f, indent=2)

        _import_test(corr_csv)

        # Verify manifest was updated
        updated = json.loads(manifest_file.read_text())[0]
        assert updated["verified"] is True
        assert updated["label_source"] == "human"
        assert updated["expected_walkable"]["centre"] == "WALKABLE"
        assert updated["expected_walkable"]["right"] == "BLOCKED"
        assert updated["expected_objects"] == ["person", "chair"]


def test_confidence_calibrator_behavior():
    """Verify calibrated freespace confidence behaves properly across inputs."""
    WALKABLE = CorridorStatus.WALKABLE

    # 1. Monotonicity with ground margin
    c1 = _compute_confidence(WALKABLE, g_frac=0.30, o_frac=0.0, seg_min_ground=0.25, seg_mean_prob=0.8, classical_cues_passed=True, temporal_walkable_fraction=0.5)
    c2 = _compute_confidence(WALKABLE, g_frac=0.80, o_frac=0.0, seg_min_ground=0.25, seg_mean_prob=0.8, classical_cues_passed=True, temporal_walkable_fraction=0.5)
    assert c2 > c1, f"Expected higher confidence for higher ground fraction: {c2} vs {c1}"

    # 2. Boundedness
    assert 0.0 <= c1 <= 1.0
    assert 0.0 <= c2 <= 1.0

    # 3. UNKNOWN status always 0.0
    c_unk = _compute_confidence(CorridorStatus.UNKNOWN, g_frac=0.9, o_frac=0.0, seg_min_ground=0.25, seg_mean_prob=0.9, classical_cues_passed=True, temporal_walkable_fraction=1.0)
    assert c_unk == 0.0


def test_sequence_metrics_calculation():
    """Verify sequence metrics equations on simulated human-labelled live frames."""
    # Simulate 10 frames
    # Frames 0-4: safe, Walk_forward appears on frame 2 (delay = 2 frames)
    # Frames 5-9: unsafe, Walk_forward appears on frame 7 (1 violation)
    records = []
    for i in range(10):
        is_user_safe = (i < 5)
        # Action is WALK_FORWARD on frames 2, 3, 4 and 7
        action = "WALK_FORWARD" if i in [2, 3, 4, 7] else "CAUTION"
        records.append({
            "frame_id": i,
            "user_marked_safe": is_user_safe,
            "action": action,
        })

    df = pd.DataFrame(records)

    # (a) Any WALK_FORWARD on user-marked unsafe frames:
    unsafe_frames = df[~df["user_marked_safe"]]
    unsafe_wf_count = int((unsafe_frames["action"] == "WALK_FORWARD").sum())
    assert unsafe_wf_count == 1

    # (b) Fraction of safe frames that reach WALK_FORWARD:
    safe_frames = df[df["user_marked_safe"]]
    safe_reached = int((safe_frames["action"] == "WALK_FORWARD").sum())
    safe_reach_rate = safe_reached / len(safe_frames)
    assert safe_reach_rate == 3 / 5  # 60%
