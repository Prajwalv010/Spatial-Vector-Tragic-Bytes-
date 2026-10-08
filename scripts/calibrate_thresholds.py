"""Instant grid search on cached TUNE features to find optimal thresholds for M14.

Strict objective:
1. False WALKABLE rate on hazard categories MUST BE 0.00% (0 / 84).
2. Maximize WALKABLE recall on clear categories.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "tune_features.json"

records = json.load(open(FEATURES_PATH))

hazard_cats = {"table_edge", "table_corner", "desk_closeup", "stairs_down", "stairs_up", "blank_wall", "covered_lens"}
clear_cats = {"indoor_corridor", "indoor_floor", "outdoor_footpath", "outdoor_road_clear"}

best_rec = 0
best_cfg = None
candidates = []

# Sweep grid
for g_th in [0.20, 0.25, 0.30, 0.35, 0.40]:
    for o_th in [0.08, 0.10, 0.12, 0.15]:
        for full_o_th in [0.20, 0.25, 0.30, 0.35]:
            for j_th in [20.0, 24.0, 28.0, 32.0]:
                for s_th in [25.0, 30.0, 35.0, 40.0]:
                    haz_fp = 0
                    haz_tot = 0
                    clr_tp = 0
                    clr_tot = 0
                    
                    for r in records:
                        cat = r["category"]
                        if cat not in hazard_cats and cat not in clear_cats:
                            continue
                            
                        # Centre corridor walkability decision:
                        # 1. Validity
                        if r["lap"] < 15.0 or r["lum"] < 15.0 or r["lum"] > 240.0:
                            is_walk = False
                        # 2. YOLO
                        elif r["has_blocking"]:
                            is_walk = False
                        # 3. Obstacle veto (centre corridor or full ground ROI)
                        elif r["o_fracs"]["centre"] > o_th or r["full_o_frac"] > full_o_th:
                            is_walk = False
                        # 4. Classical gradient jump veto
                        elif (r["max_jump"] > j_th or r["span_diff"] > s_th) and r["g_fracs"]["centre"] < 0.70:
                            is_walk = False
                        # 5. Ground confirmation
                        elif r["g_fracs"]["centre"] >= g_th:
                            is_walk = True
                        else:
                            is_walk = False
                            
                        if cat in hazard_cats:
                            haz_tot += 1
                            if is_walk:
                                haz_fp += 1
                        elif cat in clear_cats:
                            clr_tot += 1
                            if is_walk:
                                clr_tp += 1
                                
                    if haz_fp == 0:
                        rec = clr_tp / clr_tot
                        candidates.append((rec, g_th, o_th, full_o_th, j_th, s_th, clr_tp, clr_tot, haz_tot))
                        if rec > best_rec:
                            best_rec = rec
                            best_cfg = (rec, g_th, o_th, full_o_th, j_th, s_th, clr_tp, clr_tot, haz_tot)

print(f"Total 0-FP configurations found: {len(candidates)}")
if best_cfg:
    print("\n" + "=" * 65)
    print(" OPTIMAL CALIBRATED THRESHOLDS (TUNE SET ONLY)")
    print("=" * 65)
    print(f"  WALKABLE Recall on Clear Cats : {best_cfg[0]:.2%} ({best_cfg[6]}/{best_cfg[7]})")
    print(f"  False WALKABLE on Hazard Cats : 0.00% (0/{best_cfg[8]})")
    print(f"  Ground Threshold (centre)     : {best_cfg[1]:.2f}")
    print(f"  Obstacle Threshold (centre)   : {best_cfg[2]:.2f}")
    print(f"  Full ROI Obstacle Threshold   : {best_cfg[3]:.2f}")
    print(f"  Gradient Jump Veto Threshold  : {best_cfg[4]:.1f}")
    print(f"  Span Diff Veto Threshold      : {best_cfg[5]:.1f}")
    print("=" * 65)
else:
    print("No 0-FP config found in this grid. Checking min FP...")
    # Find min FP
    min_fp = 999
    min_fp_cand = None
    # Let's inspect which image is triggering
