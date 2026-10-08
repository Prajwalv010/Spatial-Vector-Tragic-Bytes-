"""Unit tests for scripts/compute_metrics.py using synthetic test fixtures.

Verifies:
1. Strict schema validation (missing columns raise ValueError, unexpected values raise TypeError).
2. Missing categories cause immediate ValueError failure.
3. Accurate metric computation against hand-calculated expected values.
"""
import io
import pandas as pd
import pytest

from scripts.compute_metrics import (
    validate_dataframe,
    compute_all_metrics,
    wilson_score_upper_bound,
    EXPECTED_HAZARD_CATEGORIES,
    EXPECTED_CLEAR_CATEGORIES,
)


def _make_dummy_csv_df() -> pd.DataFrame:
    """Build a minimal valid DataFrame covering all required categories."""
    rows = []
    
    # 7 hazard categories (1 fixture each = 7 fixtures = 21 corridor rows)
    # Let blank_wall_01 have LEFT=WALKABLE, CENTRE=UNKNOWN, RIGHT=UNKNOWN (1 per-image false walkable, 0 centre false walkable)
    for cat in EXPECTED_HAZARD_CATEGORIES:
        for c in ["left", "centre", "right"]:
            got = "WALKABLE" if (cat == "blank_wall" and c == "left") else "UNKNOWN"
            rows.append({
                "fixture": f"{cat}_01",
                "category": cat,
                "corridor": c,
                "expected_walkable": "UNKNOWN",
                "got_walkable": got,
                "conf": 0.70 if got == "WALKABLE" else 0.00,
                "pothole_expected": 0,
                "pothole_got": 0,
                "objects_expected": "",
                "objects_got": "",
            })

    # 4 clear categories (1 fixture each = 4 fixtures = 12 corridor rows)
    # Let indoor_corridor_01 be WALKABLE in all 3 corridors
    # Other 3 clear fixtures be UNKNOWN
    for cat in EXPECTED_CLEAR_CATEGORIES:
        for c in ["left", "centre", "right"]:
            got = "WALKABLE" if cat == "indoor_corridor" else "UNKNOWN"
            rows.append({
                "fixture": f"{cat}_01",
                "category": cat,
                "corridor": c,
                "expected_walkable": "WALKABLE",
                "got_walkable": got,
                "conf": 0.85 if got == "WALKABLE" else 0.00,
                "pothole_expected": 0,
                "pothole_got": 0,
                "objects_expected": "person;car" if cat == "indoor_corridor" else "",
                "objects_got": "person;car;car" if cat == "indoor_corridor" else "",
            })

    # Pothole categories (1 positive fixture, 1 negative fixture)
    # Positive pothole: detected (TP=1)
    for c in ["left", "centre", "right"]:
        rows.append({
            "fixture": "road_pothole_01",
            "category": "road_pothole",
            "corridor": c,
            "expected_walkable": "BLOCKED",
            "got_walkable": "BLOCKED",
            "conf": 0.00,
            "pothole_expected": 1,
            "pothole_got": 1,
            "objects_expected": "",
            "objects_got": "",
        })

    # Negative pothole (road_manhole): false positive (FP=1 on negative)
    for c in ["left", "centre", "right"]:
        rows.append({
            "fixture": "road_manhole_01",
            "category": "road_manhole",
            "corridor": c,
            "expected_walkable": "BLOCKED",
            "got_walkable": "BLOCKED",
            "conf": 0.00,
            "pothole_expected": 0,
            "pothole_got": 1,
            "objects_expected": "",
            "objects_got": "",
        })

    return pd.DataFrame(rows)


def test_schema_validation_passes():
    df = _make_dummy_csv_df()
    validate_dataframe(df)


def test_schema_validation_fails_on_missing_column():
    df = _make_dummy_csv_df().drop(columns=["got_walkable"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe(df)


def test_schema_validation_fails_on_missing_hazard_category():
    df = _make_dummy_csv_df()
    df = df[df["category"] != "blank_wall"]
    with pytest.raises(ValueError, match="missing expected hazard categories"):
        validate_dataframe(df)


def test_schema_validation_fails_on_unexpected_value_type():
    df = _make_dummy_csv_df()
    df.loc[0, "got_walkable"] = "INVALID_STATUS"
    with pytest.raises(TypeError, match="contains unexpected values"):
        validate_dataframe(df)


def test_metrics_computation_hand_calculated():
    df = _make_dummy_csv_df()
    m = compute_all_metrics(df)

    # 7 hazard images: blank_wall has left=WALKABLE, other 6 have all UNKNOWN
    # Per-image false walkable: 1 / 7
    hz_img = m["hazard_per_image"]
    assert hz_img["count"] == 1
    assert hz_img["total"] == 7
    assert pytest.approx(hz_img["rate"], 1e-4) == 1.0 / 7.0

    # Centre-only false walkable: 0 / 7
    hz_ctr = m["hazard_centre_only"]
    assert hz_ctr["count"] == 0
    assert hz_ctr["total"] == 7
    assert hz_ctr["rate"] == 0.0

    # Clear recall: 4 clear images. Only indoor_corridor is walkable in centre
    # Centre recall: 1 / 4 = 25%
    cl_ctr = m["clear_recall_centre"]
    assert cl_ctr["count"] == 1
    assert cl_ctr["total"] == 4
    assert pytest.approx(cl_ctr["rate"], 1e-4) == 0.25

    # Any corridor recall: 1 / 4 = 25%
    cl_any = m["clear_recall_any"]
    assert cl_any["count"] == 1
    assert cl_any["total"] == 4
    assert pytest.approx(cl_any["rate"], 1e-4) == 0.25

    # Pothole: 1 positive (road_pothole), 12 negatives (7 hazard + 4 clear + 1 manhole)
    ph = m["pothole"]
    assert ph["tp"] == 1
    assert ph["pos_total"] == 1
    assert ph["recall"] == 1.0
    assert ph["fp"] == 1  # road_manhole
    assert ph["neg_total"] == 12

    # Objects in indoor_corridor_01:
    # exp = ['person', 'car'], got = ['person', 'car', 'car']
    # Bag-of-words: TP=2, FP=1, FN=0
    # Set-based: TP=2, FP=0, FN=0
    obj_bow = m["object_agreement"]["bag_of_words"]
    assert obj_bow["tp"] == 2
    assert obj_bow["fp"] == 1
    assert obj_bow["fn"] == 0

    obj_set = m["object_agreement"]["unique_class_set"]
    assert obj_set["tp"] == 2
    assert obj_set["fp"] == 0
    assert obj_set["fn"] == 0


def test_wilson_score_bounds():
    # 0 successes out of 100: Wilson UB should be around 3.6%
    ub0 = wilson_score_upper_bound(0, 100)
    assert 0.03 < ub0 < 0.04
    # 4 successes out of 175: around 5.73%
    ub4 = wilson_score_upper_bound(4, 175)
    assert pytest.approx(ub4, abs=0.001) == 0.0573


def test_unlabelled_corridor_exclusion():
    df = _make_dummy_csv_df()
    # Mark road_pothole_01 expected_walkable as UNLABELLED
    df.loc[df["category"] == "road_pothole", "expected_walkable"] = "UNLABELLED"
    validate_dataframe(df)
    m = compute_all_metrics(df)
    assert m["unlabelled_excluded"]["images"] == 1
    assert m["unlabelled_excluded"]["corridor_decisions"] == 3
    # ECE should be a valid float
    assert 0.0 <= m["expected_calibration_error"] <= 1.0

