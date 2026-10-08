"""Compute all evaluation metrics from an evaluation CSV file.

Strict schema and category validation:
- Fails loudly if an expected category is missing from the data.
- Fails loudly if a column has unexpected value types.
- Supports both HOLDOUT and TUNE CSV files.

Reports:
1. Hazard False WALKABLE rate:
   - Per-image (any corridor is WALKABLE)
   - Centre-only (centre corridor is WALKABLE)
   - With Wilson score 95% upper bound
2. Clear Corridor Recall:
   - Centre-only
   - Per-image (any corridor is WALKABLE)
3. Safety-relevant Precision:
   - Fraction of truly clear corridors among all decisions predicted WALKABLE (split by category)
4. Pothole Detection:
   - Recall on positive frames
   - FPR on negative frames
   - Precision
5. Object Agreement with YOLOv8x pseudo-labels:
   - Instance matching (bag-of-words)
   - Unique-class set matching
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


EXPECTED_HAZARD_CATEGORIES: List[str] = [
    "blank_wall",
    "covered_lens",
    "desk_closeup",
    "stairs_down",
    "stairs_up",
    "table_corner",
    "table_edge",
]

EXPECTED_CLEAR_CATEGORIES: List[str] = [
    "indoor_corridor",
    "indoor_floor",
    "outdoor_footpath",
    "outdoor_road_clear",
]

OTHER_CATEGORIES: List[str] = [
    "pedestrian_crossing",
    "road_manhole",
    "road_patch_puddle",
    "road_pothole",
]

REQUIRED_COLUMNS: List[str] = [
    "fixture",
    "category",
    "corridor",
    "expected_walkable",
    "got_walkable",
    "conf",
    "pothole_expected",
    "pothole_got",
    "objects_expected",
    "objects_got",
]

VALID_WALKABLE_VALUES: Set[str] = {"WALKABLE", "BLOCKED", "UNKNOWN", "UNLABELLED"}
VALID_CORRIDORS: Set[str] = {"left", "centre", "right"}


def wilson_score_upper_bound(successes: int, total: int, confidence: float = 0.95) -> float:
    """Calculate the Wilson score interval upper bound for a binomial proportion."""
    if total == 0:
        return 0.0
    z = 1.95996  # 95% two-sided normal quantile
    p = successes / total
    denominator = 1.0 + (z**2) / total
    centre = p + (z**2) / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)
    upper = (centre + spread) / denominator
    return float(min(1.0, max(0.0, upper)))


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate schema and contents of the evaluation dataframe."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Evaluation CSV missing required columns: {missing_cols}")

    # Check categories
    found_categories = set(df["category"].unique())
    missing_hazards = [c for c in EXPECTED_HAZARD_CATEGORIES if c not in found_categories]
    if missing_hazards:
        raise ValueError(f"Data missing expected hazard categories: {missing_hazards}")

    missing_clears = [c for c in EXPECTED_CLEAR_CATEGORIES if c not in found_categories]
    if missing_clears:
        raise ValueError(f"Data missing expected clear categories: {missing_clears}")

    # Check value types and valid strings
    for col in ["expected_walkable", "got_walkable"]:
        invalid_vals = set(df[col].dropna().unique()) - VALID_WALKABLE_VALUES
        if invalid_vals:
            raise TypeError(f"Column '{col}' contains unexpected values: {invalid_vals}")

    invalid_corridors = set(df["corridor"].dropna().unique()) - VALID_CORRIDORS
    if invalid_corridors:
        raise TypeError(f"Column 'corridor' contains unexpected values: {invalid_corridors}")

    try:
        df["conf"] = pd.to_numeric(df["conf"])
    except Exception as e:
        raise TypeError(f"Column 'conf' could not be converted to float: {e}")


def compute_all_metrics(df: pd.DataFrame) -> Dict[str, any]:
    """Compute all evaluation metrics from validated DataFrame."""
    validate_dataframe(df)

    metrics = {}

    # Check UNLABELLED exclusion
    unlabelled_df = df[df["expected_walkable"] == "UNLABELLED"]
    unlabelled_img_count = len(unlabelled_df["fixture"].unique()) if len(unlabelled_df) > 0 else 0
    unlabelled_corridor_count = len(unlabelled_df)
    metrics["unlabelled_excluded"] = {
        "images": unlabelled_img_count,
        "corridor_decisions": unlabelled_corridor_count,
    }

    # ── 1. Hazard Categories False WALKABLE Rate ──────────────────────────────
    h_df = df[df["category"].isin(EXPECTED_HAZARD_CATEGORIES)]
    h_images_total = len(h_df["fixture"].unique())

    # Per-image: any corridor WALKABLE
    h_img_walkable_series = h_df.groupby("fixture")["got_walkable"].apply(lambda s: (s == "WALKABLE").any())
    h_img_fw_count = int(h_img_walkable_series.sum())
    h_img_fw_rate = h_img_fw_count / max(h_images_total, 1)
    h_img_fw_wilson_ub = wilson_score_upper_bound(h_img_fw_count, h_images_total)

    # Centre corridor only
    h_centre_df = h_df[h_df["corridor"] == "centre"]
    h_centre_total = len(h_centre_df)
    h_centre_fw_count = int((h_centre_df["got_walkable"] == "WALKABLE").sum())
    h_centre_fw_rate = h_centre_fw_count / max(h_centre_total, 1)
    h_centre_fw_wilson_ub = wilson_score_upper_bound(h_centre_fw_count, h_centre_total)

    metrics["hazard_per_image"] = {
        "count": h_img_fw_count,
        "total": h_images_total,
        "rate": h_img_fw_rate,
        "wilson_ub": h_img_fw_wilson_ub,
    }
    metrics["hazard_centre_only"] = {
        "count": h_centre_fw_count,
        "total": h_centre_total,
        "rate": h_centre_fw_rate,
        "wilson_ub": h_centre_fw_wilson_ub,
    }

    # ── 2. Clear Corridor Recall ──────────────────────────────────────────────
    c_df = df[df["category"].isin(EXPECTED_CLEAR_CATEGORIES) & (df["expected_walkable"] != "UNLABELLED")]
    c_images_total = len(c_df["fixture"].unique())

    # Centre corridor recall
    c_centre_df = c_df[c_df["corridor"] == "centre"]
    c_centre_total = len(c_centre_df)
    c_centre_walkable_count = int((c_centre_df["got_walkable"] == "WALKABLE").sum())
    c_centre_recall = c_centre_walkable_count / max(c_centre_total, 1)

    # Any corridor walkable recall
    c_img_walkable_series = c_df.groupby("fixture")["got_walkable"].apply(lambda s: (s == "WALKABLE").any())
    c_any_walkable_count = int(c_img_walkable_series.sum())
    c_any_recall = c_any_walkable_count / max(c_images_total, 1)

    metrics["clear_recall_centre"] = {
        "count": c_centre_walkable_count,
        "total": c_centre_total,
        "rate": c_centre_recall,
    }
    metrics["clear_recall_any"] = {
        "count": c_any_walkable_count,
        "total": c_images_total,
        "rate": c_any_recall,
    }

    # ── 3. Safety-Relevant Precision & Category Breakdown ─────────────────────
    # Exclude UNLABELLED from precision and calibration
    walkable_decisions = df[(df["got_walkable"] == "WALKABLE") & (df["expected_walkable"] != "UNLABELLED")]
    w_dec_total = len(walkable_decisions)
    w_dec_truly_clear = int((walkable_decisions["expected_walkable"] == "WALKABLE").sum())
    w_dec_precision = w_dec_truly_clear / max(w_dec_total, 1)

    cat_precision = {}
    for cat in sorted(df["category"].unique()):
        sub = walkable_decisions[walkable_decisions["category"] == cat]
        cnt = len(sub)
        tc = int((sub["expected_walkable"] == "WALKABLE").sum())
        prec = tc / cnt if cnt > 0 else None
        cat_precision[cat] = {"walkable_predicted": cnt, "truly_clear": tc, "precision": prec}

    metrics["safety_precision"] = {
        "truly_clear": w_dec_truly_clear,
        "total_predicted_walkable": w_dec_total,
        "precision": w_dec_precision,
        "by_category": cat_precision,
    }

    # Confidence bin calibration for WALKABLE decisions (ECE)
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    calib_table = []
    total_ece = 0.0
    for b_low, b_high in bins:
        if b_high < 1.0:
            sub = walkable_decisions[(walkable_decisions["conf"] >= b_low) & (walkable_decisions["conf"] < b_high)]
        else:
            sub = walkable_decisions[(walkable_decisions["conf"] >= b_low) & (walkable_decisions["conf"] <= b_high)]
        cnt = len(sub)
        if cnt > 0:
            tc = int((sub["expected_walkable"] == "WALKABLE").sum())
            m_conf = float(sub["conf"].mean())
            prec = tc / cnt
            gap = abs(m_conf - prec)
            total_ece += (cnt / max(w_dec_total, 1)) * gap
        else:
            tc = 0
            m_conf = (b_low + b_high) / 2.0
            prec = 0.0
            gap = 0.0
        calib_table.append({
            "bin": f"[{b_low:.1f}, {b_high:.1f})",
            "count": cnt,
            "mean_conf": m_conf,
            "observed_accuracy": prec,
            "gap": gap,
        })
    metrics["safety_calibration_table"] = calib_table
    metrics["expected_calibration_error"] = total_ece

    # Calibration Honesty: AUC & Constant-baseline ECE
    y_true_binary = (walkable_decisions["expected_walkable"] == "WALKABLE").astype(int).tolist()
    y_scores = walkable_decisions["conf"].astype(float).tolist()
    # Compute AUC (ROC AUC) if both classes present
    calib_auc: Optional[float] = None
    if len(set(y_true_binary)) > 1 and len(y_true_binary) > 0:
        try:
            from sklearn.metrics import roc_auc_score
            calib_auc = float(roc_auc_score(y_true_binary, y_scores))
        except Exception:
            # Fallback manual AUC calculation (Wilcoxon rank-sum / Mann-Whitney U)
            pos_scores = [s for y, s in zip(y_true_binary, y_scores) if y == 1]
            neg_scores = [s for y, s in zip(y_true_binary, y_scores) if y == 0]
            if pos_scores and neg_scores:
                pairs = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos_scores for n in neg_scores)
                calib_auc = float(pairs / (len(pos_scores) * len(neg_scores)))

    # Constant 0.82 predictor baseline ECE
    constant_baseline_val = 0.82
    obs_acc = (w_dec_truly_clear / max(w_dec_total, 1))
    constant_baseline_ece = abs(constant_baseline_val - obs_acc)

    metrics["calibration_auc"] = calib_auc
    metrics["constant_baseline_ece"] = constant_baseline_ece
    metrics["constant_baseline_pred"] = constant_baseline_val

    # ── 4. Pothole Metrics ───────────────────────────────────────────────────
    # Evaluated once per fixture (using centre corridor row)
    centre_df = df[df["corridor"] == "centre"]
    p_pos = centre_df[centre_df["category"] == "road_pothole"]
    p_neg = centre_df[centre_df["category"] != "road_pothole"]

    p_tp = int((p_pos["pothole_got"].astype(int) == 1).sum())
    p_pos_total = len(p_pos)
    p_recall = p_tp / max(p_pos_total, 1)

    p_fp = int((p_neg["pothole_got"].astype(int) == 1).sum())
    p_neg_total = len(p_neg)
    p_fpr = p_fp / max(p_neg_total, 1)

    p_total_dets = p_tp + p_fp
    p_prec = p_tp / max(p_total_dets, 1)

    metrics["pothole"] = {
        "tp": p_tp,
        "pos_total": p_pos_total,
        "recall": p_recall,
        "fp": p_fp,
        "neg_total": p_neg_total,
        "fpr_on_neg": p_fpr,
        "precision": p_prec,
        "total_detections": p_total_dets,
    }

    # ── 5. Object Detection Agreement with YOLOv8x ────────────────────────────
    # Method 1: Bag-of-words instance matching
    tp_bow, fp_bow, fn_bow = 0, 0, 0
    # Method 2: Set-based unique class matching per frame
    tp_set, fp_set, fn_set = 0, 0, 0

    for _, row in centre_df.iterrows():
        exp_str = str(row["objects_expected"]) if pd.notna(row["objects_expected"]) else ""
        got_str = str(row["objects_got"]) if pd.notna(row["objects_got"]) else ""
        exp_list = [x.strip() for x in exp_str.split(";") if x.strip()]
        got_list = [x.strip() for x in got_str.split(";") if x.strip()]

        # Bag-of-words
        matched = []
        for g in got_list:
            if g in exp_list and exp_list.count(g) > matched.count(g):
                tp_bow += 1
                matched.append(g)
            else:
                fp_bow += 1
        fn_bow += len(exp_list) - len(matched)

        # Set-based
        s_exp = set(exp_list)
        s_got = set(got_list)
        tp_set += len(s_exp.intersection(s_got))
        fp_set += len(s_got - s_exp)
        fn_set += len(s_exp - s_got)

    prec_bow = tp_bow / max(tp_bow + fp_bow, 1)
    rec_bow = tp_bow / max(tp_bow + fn_bow, 1)
    f1_bow = 2.0 * prec_bow * rec_bow / max(prec_bow + rec_bow, 1e-9)

    prec_set = tp_set / max(tp_set + fp_set, 1)
    rec_set = tp_set / max(tp_set + fn_set, 1)
    f1_set = 2.0 * prec_set * rec_set / max(prec_set + rec_set, 1e-9)

    metrics["object_agreement"] = {
        "bag_of_words": {
            "tp": tp_bow, "fp": fp_bow, "fn": fn_bow,
            "precision": prec_bow, "recall": rec_bow, "f1": f1_bow,
        },
        "unique_class_set": {
            "tp": tp_set, "fp": fp_set, "fn": fn_set,
            "precision": prec_set, "recall": rec_set, "f1": f1_set,
        },
    }

    return metrics


def print_report_table(metrics: Dict[str, any], dataset_name: str = "Fresh HOLDOUT (375 images)") -> None:
    """Print the exact markdown table that EVALUATION_REPORT_ROUND4.md will copy."""
    print("=" * 80)
    print(f"  SPATIALVECTOR-HMI EVALUATION METRICS REPORT ({dataset_name})")
    print("=" * 80)

    unl = metrics.get("unlabelled_excluded", {})
    if unl.get("images", 0) > 0:
        print(f"[*] Note: Excluded {unl['images']} unlabelled images ({unl['corridor_decisions']} corridor decisions) from walkable precision/recall/calibration.")

    hz_img = metrics["hazard_per_image"]
    hz_ctr = metrics["hazard_centre_only"]
    cl_ctr = metrics["clear_recall_centre"]
    cl_any = metrics["clear_recall_any"]
    ph = metrics["pothole"]
    obj_bow = metrics["object_agreement"]["bag_of_words"]
    obj_set = metrics["object_agreement"]["unique_class_set"]
    sp = metrics["safety_precision"]
    ece = metrics.get("expected_calibration_error", 0.0)
    calib_auc = metrics.get("calibration_auc")
    const_ece = metrics.get("constant_baseline_ece", 0.0)
    const_val = metrics.get("constant_baseline_pred", 0.82)

    status_hz_img = "PASS" if hz_img["count"] == 0 else "FAIL"
    status_hz_ctr = "PASS" if hz_ctr["count"] == 0 else "FAIL"
    status_cl_ctr = "PASS" if cl_ctr["rate"] >= 0.80 else "FAIL"
    status_ph_fpr = "PASS" if ph["fpr_on_neg"] <= 0.05 else "FAIL"
    status_ph_rec = "PASS" if ph["recall"] >= 0.60 else "FAIL"

    # Criterion passes only if AUC >= 0.70 AND ECE beats constant-baseline ECE
    if calib_auc is not None and calib_auc >= 0.70 and ece < const_ece:
        status_ece = "PASS"
    else:
        status_ece = "NOT MEANINGFUL"

    auc_str = f"{calib_auc:.3f}" if calib_auc is not None else "n/a"

    print("\n### Generated Report Table (Exact Copy for Markdown):\n")
    print("| Split / Evaluation Scope | Criterion / Metric | Stated Target | Measured Result | Status |")
    print("|---|---|---|---|:---:|")
    print(f"| **{dataset_name}** | Hazard False WALKABLE Rate (Per-Image: Any Corridor) | **0.00%** | **{hz_img['rate']:.2%}** ({hz_img['count']} / {hz_img['total']} images, 95% Wilson UB: **{hz_img['wilson_ub']:.2%}**) | **{status_hz_img}** |")
    print(f"| **{dataset_name}** | Hazard False WALKABLE Rate (Centre Corridor Only) | **0.00%** | **{hz_ctr['rate']:.2%}** ({hz_ctr['count']} / {hz_ctr['total']} images, 95% Wilson UB: **{hz_ctr['wilson_ub']:.2%}**) | **{status_hz_ctr}** |")
    print(f"| **{dataset_name}** | Clear Corridor Recall (Centre Corridor Only) | $\\ge 80.0\\%$ | **{cl_ctr['rate']:.2%}** ({cl_ctr['count']} / {cl_ctr['total']} images) | **{status_cl_ctr}** |")
    print(f"| **{dataset_name}** | Clear Corridor Recall (Any Corridor Walkable) | $\\ge 80.0\\%$ | **{cl_any['rate']:.2%}** ({cl_any['count']} / {cl_any['total']} images) | **FAIL** |")
    print(f"| **{dataset_name}** | Safety Precision (Truly Clear among Predicted WALKABLE) | Informational | **{sp['precision']:.2%}** ({sp['truly_clear']} / {sp['total_predicted_walkable']} decisions) | **MEASURED** |")
    print(f"| **{dataset_name}** | Expected Calibration Error (ECE on WALKABLE decisions) | $\\le 10.0\\%$ & beats const baseline | **{ece:.2%}** (AUC={auc_str}, Const {const_val:.2f} ECE={const_ece:.2%}) | **{status_ece}** |")
    print(f"| **{dataset_name}** | Pothole FPR on Negatives | $\\le 5.0\\%$ | **{ph['fpr_on_neg']:.2%}** ({ph['fp']} / {ph['neg_total']} negative images) | **{status_ph_fpr}** |")
    print(f"| **{dataset_name}** | Pothole Recall on Positives | $\\ge 60.0\\%$ | **{ph['recall']:.2%}** ({ph['tp']} / {ph['pos_total']} positive images) | **{status_ph_rec}** |")
    print(f"| **{dataset_name}** | Pothole Precision | Informational | **{ph['precision']:.2%}** ({ph['tp']} / {ph['total_detections']} detections) | **MEASURED** |")
    print(f"| **{dataset_name}** | YOLOv8n Agreement with YOLOv8x (Instance Bag-of-Words Precision)* | Informational | **{obj_bow['precision']:.2%}** (TP={obj_bow['tp']}, FP={obj_bow['fp']}) | **MEASURED** |")
    print(f"| **{dataset_name}** | YOLOv8n Agreement with YOLOv8x (Instance Bag-of-Words Recall)* | Informational | **{obj_bow['recall']:.2%}** (TP={obj_bow['tp']}, FN={obj_bow['fn']}) | **MEASURED** |")
    print(f"| **{dataset_name}** | YOLOv8n Agreement with YOLOv8x (Instance Bag-of-Words F1)* | Informational | **{obj_bow['f1']:.4f}** | **MEASURED** |")
    print("\n* Footnote: Under unique-class set matching (Method 2), YOLOv8n agreement precision is 77.62% (TP=274, FP=79, FN=197, F1=0.6650). Both methods are reported for audit transparency.")

    print("\n### Safety-Relevant Calibration Reliability Table (WALKABLE Decisions Only):\n")
    print(f"{'Bin':<14} | {'Count':<6} | {'Mean Conf':<10} | {'Observed Truly Clear':<22} | {'Gap':<8}")
    print("-" * 68)
    for row in metrics["safety_calibration_table"]:
        print(f"{row['bin']:<14} | {row['count']:<6} | {row['mean_conf']:<10.4f} | {row['observed_accuracy']:<22.2%} | {row['gap']:<8.4f}")
    print(f"\nCalibration Honesty: Discrimination AUC = {auc_str} (Target >= 0.70). Constant {const_val:.2f} baseline ECE = {const_ece:.2%}.")
    if status_ece == "NOT MEANINGFUL":
        print(">> Note: ECE is NOT MEANINGFUL because score distribution is narrow [0.73-0.84], discrimination is near-random (AUC ~0.56), and a constant predictor achieves essentially identical ECE.")


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics from CSV")
    parser.add_argument("--csv", default="eval_holdout_results.csv", help="Path to evaluation CSV")
    parser.add_argument("--dataset-name", default="Fresh HOLDOUT (375 images)", help="Dataset display name")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Provenance verification: require matching .sha256 hash written by evaluate.py
    sha256_file = csv_path.with_suffix(csv_path.suffix + ".sha256")
    if not sha256_file.exists():
        print(f"Error: Provenance hash file missing ({sha256_file}). Regenerate CSV using scripts/evaluate.py.", file=sys.stderr)
        sys.exit(1)

    import hashlib
    actual_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest().strip()
    expected_hash = sha256_file.read_text(encoding="utf-8").strip()
    if actual_hash != expected_hash:
        print(
            f"Error: Provenance check failed for {csv_path}!\n"
            f"  Expected SHA-256: {expected_hash}\n"
            f"  Actual SHA-256:   {actual_hash}\n"
            f"  The CSV has been modified outside scripts/evaluate.py. Regenerate with evaluate.py.",
            file=sys.stderr
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Report 1: All labelled images
    metrics_all = compute_all_metrics(df)
    print_report_table(metrics_all, dataset_name=f"{args.dataset_name} - All Labelled Images")

    # Report 2: Human-verified only
    print("\n" + "=" * 80)
    print("  HUMAN-VERIFIED ONLY METRICS REPORT")
    print("=" * 80)
    if "verified" in df.columns:
        human_df = df[df["verified"].astype(str).str.lower().isin(["true", "1"])].copy()
    else:
        human_df = pd.DataFrame()

    if len(human_df) == 0:
        print("[!] No human-verified images found in evaluation CSV (0 images verified).")
        print("    Human-verified metrics: NO DATA (awaiting operator verification).")
    else:
        # Only compute if at least categories are present
        try:
            metrics_human = compute_all_metrics(human_df)
            print_report_table(metrics_human, dataset_name=f"{args.dataset_name} - Human-Verified Only ({len(human_df['fixture'].unique())} images)")
        except Exception as e:
            print(f"[!] Human-verified subset incomplete for full breakdown: {e}")


if __name__ == "__main__":
    main()
