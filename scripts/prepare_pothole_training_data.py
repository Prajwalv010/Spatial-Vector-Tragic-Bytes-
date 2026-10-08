"""Prepare fine-tuning dataset for pothole detector with >=30% negative samples.

Combines positive pothole samples from Ryukijano/Pothole-detection-Yolov8 (CC BY 4.0)
with negative real images from the TUNE set (plain road, shadows, manholes,
patches, puddles, tiles, carpet, walls).
"""

import os
import shutil
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "pothole_finetune"
TUNE_DIR = BASE_DIR / "tests" / "fixtures" / "real" / "tune"

def download_pair(pair, is_train):
    s, lbl = pair
    target_img_dir = DATA_DIR / ("train" if is_train else "val") / "images"
    target_lbl_dir = DATA_DIR / ("train" if is_train else "val") / "labels"
    try:
        img_p = hf_hub_download("Ryukijano/Pothole-detection-Yolov8", s, repo_type="dataset")
        lbl_p = hf_hub_download("Ryukijano/Pothole-detection-Yolov8", lbl, repo_type="dataset")
        shutil.copyfile(img_p, target_img_dir / Path(s).name)
        shutil.copyfile(lbl_p, target_lbl_dir / Path(lbl).name)
        return True
    except Exception as e:
        print(f"Error downloading {s}: {e}")
        return False

def setup_finetune_dataset():
    print("[*] Preparing pothole fine-tuning dataset with >=30% negatives...")
    
    train_img_dir = DATA_DIR / "train" / "images"
    train_lbl_dir = DATA_DIR / "train" / "labels"
    val_img_dir = DATA_DIR / "val" / "images"
    val_lbl_dir = DATA_DIR / "val" / "labels"
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    api = HfApi()
    info = api.dataset_info("Ryukijano/Pothole-detection-Yolov8")
    
    train_pos_pairs = []
    val_pos_pairs = []
    
    siblings = [f.rfilename for f in info.siblings]
    for s in siblings:
        if s.startswith("train/images/") and s.endswith(".jpg"):
            lbl = s.replace("train/images/", "train/labels/").rsplit(".", 1)[0] + ".txt"
            if lbl in siblings:
                train_pos_pairs.append((s, lbl))
        elif s.startswith("valid/images/") and s.endswith(".jpg"):
            lbl = s.replace("valid/images/", "valid/labels/").rsplit(".", 1)[0] + ".txt"
            if lbl in siblings:
                val_pos_pairs.append((s, lbl))

    num_train_pos = min(60, len(train_pos_pairs))
    num_val_pos = min(15, len(val_pos_pairs))
    
    selected_train = train_pos_pairs[:num_train_pos]
    selected_val = val_pos_pairs[:num_val_pos]
    
    print(f"  Downloading {num_train_pos} train positives and {num_val_pos} val positives with 12 threads...")
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(download_pair, p, True) for p in selected_train]
        futs += [ex.submit(download_pair, p, False) for p in selected_val]
        for f in as_completed(futs):
            f.result()
            
    print(f"  Downloaded positive pairs.")
    
    # Negative samples from TUNE (plain road, manhole, patch/puddle, floor, corridor, wall)
    neg_categories = [
        "outdoor_road_clear",
        "road_manhole",
        "road_patch_puddle",
        "indoor_floor",
        "indoor_corridor",
        "blank_wall",
        "table_edge",
    ]
    
    tune_files = list(TUNE_DIR.glob("*.jpg"))
    neg_train_count = 0
    neg_val_count = 0
    
    for f in tune_files:
        cat = next((c for c in neg_categories if f.name.startswith(c + "_")), None)
        if not cat:
            continue
        
        if neg_val_count < 10 and (neg_train_count + neg_val_count) % 4 == 0:
            dest_img = val_img_dir / f.name
            dest_lbl = val_lbl_dir / (f.stem + ".txt")
            neg_val_count += 1
        else:
            dest_img = train_img_dir / f.name
            dest_lbl = train_lbl_dir / (f.stem + ".txt")
            neg_train_count += 1
            
        shutil.copyfile(f, dest_img)
        dest_lbl.write_text("")

    total_train = len(list(train_img_dir.glob("*.jpg")))
    total_val = len(list(val_img_dir.glob("*.jpg")))
    
    print(f"  Total train: {total_train} images ({neg_train_count} negatives, {neg_train_count/total_train:.1%})")
    print(f"  Total val  : {total_val} images ({neg_val_count} negatives, {neg_val_count/total_val:.1%})")
    
    data_yaml = {
        "train": str(train_img_dir.resolve()).replace("\\", "/"),
        "val": str(val_img_dir.resolve()).replace("\\", "/"),
        "nc": 1,
        "names": ["pothole"],
    }
    
    data_yaml_path = DATA_DIR / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
        
    print(f"[+] Dataset configured at: {data_yaml_path}")
    return str(data_yaml_path)

if __name__ == "__main__":
    setup_finetune_dataset()
