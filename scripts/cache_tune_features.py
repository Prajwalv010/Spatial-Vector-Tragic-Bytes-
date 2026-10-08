"""Extract all features for TUNE fixtures and save to data/tune_features.json."""
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from transformers import SegformerForSemanticSegmentation
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "tests" / "fixtures" / "real" / "manifest.json"

device = "cpu"
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)
model.eval()
yolo = YOLO(str(BASE_DIR / "yolov8n.pt"))

GROUND_CLASSES = {3, 6, 11, 13, 28, 52}
OBSTACLE_CLASSES = {0, 1, 2, 7, 8, 10, 12, 14, 15, 31, 32, 33, 53, 56, 59, 64, 80, 102, 121}
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

manifest = json.load(open(MANIFEST_PATH))
tune_set = [x for x in manifest if x["split"] == "TUNE"]

records = []
print(f"[*] Extracting and caching features for {len(tune_set)} TUNE images...")

for idx, item in enumerate(tune_set):
    cat = item["category"]
    img = cv2.imread(str(BASE_DIR / "tests" / "fixtures" / "real" / "tune" / item["image_file"]))
    if img is None:
        continue
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lum = float(np.mean(gray))

    # YOLO objects
    res = yolo(img, conf=0.20, verbose=False)[0]
    blocking_classes = []
    has_blocking = False
    for box, cls_id in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
        cname = yolo.names[int(cls_id)]
        if cname in ["dining table", "desk", "chair", "bench", "couch", "bed", "laptop", "tv", "person", "car", "truck"]:
            bx1, by1, bx2, by2 = box
            if by2 > h * 0.50 and bx1 < w * 0.65 and bx2 > w * 0.35:
                has_blocking = True
                blocking_classes.append(cname)

    # SegFormer
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    tensor = torch.from_numpy(((resized - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    pred = torch.argmax(out.logits, dim=1)[0].numpy()
    ph, pw = pred.shape
    
    # Per-corridor ground and obstacle fractions
    corrs = {
        "left":   (0, int(pw * 0.38)),
        "centre": (int(pw * 0.38), int(pw * 0.62)),
        "right":  (int(pw * 0.62), pw),
    }
    g_fracs = {}
    o_fracs = {}
    for cname, (cx0, cx1) in corrs.items():
        c_pred = pred[int(ph * 0.55):, cx0:cx1]
        g_fracs[cname] = float(np.mean(np.isin(c_pred, list(GROUND_CLASSES))))
        o_fracs[cname] = float(np.mean(np.isin(c_pred, list(OBSTACLE_CLASSES))))

    full_roi = pred[int(ph * 0.55):, :]
    full_o_frac = float(np.mean(np.isin(full_roi, list(OBSTACLE_CLASSES))))

    # Classical gradients
    roi_gray = gray[int(h * 0.55):int(h * 0.95), int(w * 0.38):int(w * 0.62)]
    rh = roi_gray.shape[0]
    row_means = np.mean(roi_gray, axis=1) if rh > 0 else np.array([0])
    diffs = np.abs(np.diff(row_means)) if len(row_means) > 1 else np.array([0])
    max_jump = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    span_diff = abs(float(np.mean(roi_gray[:rh // 3, :])) - float(np.mean(roi_gray[-rh // 3:, :]))) if rh >= 6 else 0.0
    std_val = float(np.std(roi_gray)) if rh > 0 else 0.0

    records.append({
        "id": item["id"],
        "category": cat,
        "lap": lap,
        "lum": lum,
        "has_blocking": has_blocking,
        "blocking_classes": blocking_classes,
        "g_fracs": g_fracs,
        "o_fracs": o_fracs,
        "full_o_frac": full_o_frac,
        "max_jump": max_jump,
        "span_diff": span_diff,
        "std_val": std_val,
    })
    if (idx + 1) % 45 == 0:
        print(f"  Processed {idx + 1}/{len(tune_set)}...")

out_p = BASE_DIR / "data" / "tune_features.json"
out_p.parent.mkdir(parents=True, exist_ok=True)
with open(out_p, "w") as f:
    json.dump(records, f, indent=2)

print(f"[+] Successfully saved {len(records)} records to {out_p}")
