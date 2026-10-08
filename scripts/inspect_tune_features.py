import sys
import json
from pathlib import Path
import cv2
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from scripts.tune_m14 import extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "tests" / "fixtures" / "real" / "manifest.json"

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model.eval()

manifest = json.load(open(MANIFEST_PATH))
tune_set = [x for x in manifest if x["split"] == "TUNE"]

hazard_cats = {"table_edge", "table_corner", "desk_closeup", "stairs_down", "stairs_up", "blank_wall", "covered_lens"}
clear_cats = {"indoor_corridor", "indoor_floor", "outdoor_footpath", "outdoor_road_clear"}

print("--- Hazard images with g_frac > 0.15 ---")
for item in tune_set:
    cat = item["category"]
    if cat not in hazard_cats:
        continue
    img_p = BASE_DIR / "tests" / "fixtures" / "real" / "tune" / item["image_file"]
    img = cv2.imread(str(img_p))
    feats = extract_features(img, model, processor)
    if feats["g_frac"] > 0.15:
        print(f"{item['id']:<20} ({cat:<14}): g={feats['g_frac']:.2f}, h={feats['h_frac']:.2f}, dy={feats['max_row_dy']:.1f}, lum={feats['mean_lum']:.1f}, lap={feats['lap_var']:.1f}")

print("\n--- Clear images ---")
for item in tune_set:
    cat = item["category"]
    if cat not in clear_cats:
        continue
    img_p = BASE_DIR / "tests" / "fixtures" / "real" / "tune" / item["image_file"]
    img = cv2.imread(str(img_p))
    feats = extract_features(img, model, processor)
    print(f"{item['id']:<20} ({cat:<20}): g={feats['g_frac']:.2f}, h={feats['h_frac']:.2f}, dy={feats['max_row_dy']:.1f}, lum={feats['mean_lum']:.1f}, lap={feats['lap_var']:.1f}")
