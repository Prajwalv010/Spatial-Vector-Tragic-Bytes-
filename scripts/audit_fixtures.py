"""Audit fixtures for mismatched categories, incorrect labels, and COCO object detections."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation
from ultralytics import YOLO

def main():
    seg = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512').eval()
    yolo = YOLO('yolov8x.pt')

    base_dir = Path('tests/fixtures/real')
    mismatches = []
    category_summary = {}

    for cat_dir in sorted([d for d in base_dir.iterdir() if d.is_dir()]):
        for img_path in sorted(cat_dir.glob('*.jpg')):
            json_path = img_path.with_suffix('.json')
            if not json_path.exists():
                continue
            data = json.loads(json_path.read_text())
            cat = data.get('category', '')

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[ERROR] Cannot read {img_path}")
                continue

            # Run SegFormer on ground ROI
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
            norm = ((res - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32)
            t = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                out = seg(t)
            pred = torch.argmax(out.logits, dim=1)[0].numpy()
            roi = pred[int(pred.shape[0]*0.55):, :]
            u, counts = np.unique(roi, return_counts=True)
            top_classes = sorted([(int(k), float(v)/roi.size) for k, v in zip(u, counts)], key=lambda x: -x[1])
            top_label = seg.config.id2label.get(top_classes[0][0]) if top_classes else "none"
            top_frac = top_classes[0][1] if top_classes else 0.0

            # Run YOLOv8x for all visible COCO objects
            yres = yolo(img, conf=0.30, verbose=False)[0]
            detected_objs = []
            if yres.boxes is not None and len(yres.boxes) > 0:
                for b in yres.boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    name = yres.names[cls_id]
                    detected_objs.append((name, round(conf, 2)))

            # Check for glaring category mismatches
            # e.g., category == 'indoor_floor' but SegFormer says 95%+ wall or building
            is_mismatch = False
            mismatch_reason = ""
            if cat == 'indoor_floor' and top_label in ('wall', 'building') and top_frac > 0.85:
                is_mismatch = True
                mismatch_reason = f"Labeled indoor_floor but image is {top_frac:.0%} {top_label}"
            elif cat == 'blank_wall' and top_label in ('floor', 'road', 'sidewalk') and top_frac > 0.70:
                is_mismatch = True
                mismatch_reason = f"Labeled blank_wall but ground ROI is {top_frac:.0%} {top_label}"
            elif cat == 'table_edge' and top_label in ('road', 'sidewalk', 'earth'):
                is_mismatch = True
                mismatch_reason = f"Labeled table_edge but ground ROI is {top_frac:.0%} {top_label}"

            if is_mismatch:
                mismatches.append({
                    "path": str(img_path),
                    "category": cat,
                    "reason": mismatch_reason,
                    "top_label": top_label,
                    "top_frac": round(top_frac, 2),
                    "url": data.get("source_url", "")
                })

    print(f"Total fixtures audited across real/. Found {len(mismatches)} obvious category mismatches:")
    for m in mismatches:
        print(f"  {Path(m['path']).name} ({m['category']}): {m['reason']} [URL: {m['url'][:60]}]")

    Path('data').mkdir(exist_ok=True)
    with open('data/audit_mismatches.json', 'w') as f:
        json.dump(mismatches, f, indent=2)

if __name__ == '__main__':
    main()
