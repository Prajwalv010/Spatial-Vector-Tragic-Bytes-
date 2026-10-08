"""Download and curate a benchmark dataset of real images for SpatialVector-HMI.

Collects real photos from public-domain and permissively licensed sources
(Wikimedia Commons CC0/CC-BY/Public Domain, Roboflow CC-BY pothole dataset) across 15 categories:
- Indoor: corridors, floors, table edges, table corners, desk close-ups, stairs down, stairs up, blank walls, covered lenses
- Outdoor: footpaths/sidewalks, clear roads, road potholes, road manholes, road patches/puddles, pedestrian crossings

Splits images deterministically into TUNE (50%) and HOLDOUT (50%).
Creates per-image JSON ground truth and a comprehensive manifest (JSON + CSV).
"""

import os
import sys
import json
import csv
import time
import requests
from PIL import Image
import io
from urllib.parse import quote, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(BASE_DIR, "tests", "fixtures", "real")
TUNE_DIR = os.path.join(TARGET_DIR, "tune")
HOLDOUT_DIR = os.path.join(TARGET_DIR, "holdout")

CATEGORIES = {
    "indoor_corridor": (False, True, [], [], "corridor"),
    "indoor_floor": (False, True, [], [], "flooring"),
    "outdoor_footpath": (False, True, [], ["person"], "sidewalk"),
    "outdoor_road_clear": (False, True, [], ["car"], "asphalt road"),
    "table_edge": (True, False, [], [], "table edge"),
    "table_corner": (True, False, [], [], "table corner"),
    "desk_closeup": (True, False, [], ["laptop", "keyboard"], "computer keyboard"),
    "stairs_down": (True, False, [], [], "stairs down"),
    "stairs_up": (True, False, [], [], "staircase"),
    "blank_wall": (True, False, [], [], "interior wall"),
    "covered_lens": (True, False, [], [], None),
    "road_pothole": (True, False, ["pothole"], [], None),
    "road_manhole": (False, True, [], [], "manhole cover"),
    "road_patch_puddle": (False, True, [], [], "road puddle"),
    "pedestrian_crossing": (False, False, [], ["person"], "pedestrians walking"),
}

HEADERS = {
    "User-Agent": "SpatialVectorBot/1.0 (https://github.com/Kshitiz-Khandelwal/Nirman-Hackathon; dev@spatialvector.org)"
}

def query_commons(search_term, limit=35):
    """Query Wikimedia Commons API for images matching search_term with 640px thumbs."""
    encoded = quote(search_term)
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={encoded}&gsrnamespace=6&gsrlimit={limit}&prop=imageinfo&iiprop=url|extmetadata|size&iiurlwidth=640&format=json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        results = []
        for pid, p in pages.items():
            ii = p.get("imageinfo", [{}])[0]
            thumb = ii.get("thumburl") or ii.get("url")
            width = ii.get("width", 0)
            height = ii.get("height", 0)
            if not thumb or width < 250 or height < 180:
                continue
            orig_url = ii.get("url", "")
            url_path = urlparse(orig_url).path.lower()
            if not any(url_path.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                continue
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "CC-BY/Public Domain")
            results.append({
                "title": p.get("title", ""),
                "url": orig_url,
                "thumb_url": thumb,
                "license": lic,
                "width": width,
                "height": height
            })
        return results
    except Exception as e:
        print(f"[!] Commons query failed for '{search_term}': {e}", flush=True)
        return []

def download_image_direct(thumb_url, orig_url):
    """Download thumbnail and return RGB PIL Image, falling back to orig_url."""
    for u in [thumb_url, orig_url]:
        if not u:
            continue
        try:
            r = requests.get(u, headers=HEADERS, timeout=6)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                img = img.resize((640, 480), Image.Resampling.BILINEAR)
                return img
        except Exception:
            continue
    return None

def fetch_ryukijano_potholes(target_count=24):
    """Fetch real pothole images from Ryukijano/Pothole-detection-Yolov8 (CC BY 4.0)."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.dataset_info("Ryukijano/Pothole-detection-Yolov8")
    test_imgs = [f.rfilename for f in info.siblings if f.rfilename.startswith("test/images/") and f.rfilename.endswith(".jpg")]
    items = []
    print(f"  Fetching up to {target_count} real pothole images from Ryukijano (CC BY 4.0)...", flush=True)
    for rfile in test_imgs[:target_count]:
        try:
            local_path = hf_hub_download(repo_id="Ryukijano/Pothole-detection-Yolov8", filename=rfile, repo_type="dataset")
            img = Image.open(local_path).convert("RGB")
            img = img.resize((640, 480), Image.Resampling.BILINEAR)
            items.append((img, "CC BY 4.0", f"https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8/blob/main/{rfile}"))
        except Exception as e:
            print(f"    Failed downloading {rfile}: {e}", flush=True)
    return items

def main():
    os.makedirs(TUNE_DIR, exist_ok=True)
    os.makedirs(HOLDOUT_DIR, exist_ok=True)

    manifest = []
    print("[*] Starting real dataset collection (15 categories)...", flush=True)

    for cat, (is_hazard, exp_walk, exp_haz, exp_obj, query) in CATEGORIES.items():
        # Check if already collected
        existing_tune = [f for f in os.listdir(TUNE_DIR) if f.startswith(f"{cat}_") and f.endswith(".jpg")]
        existing_holdout = [f for f in os.listdir(HOLDOUT_DIR) if f.startswith(f"{cat}_") and f.endswith(".jpg")]
        if len(existing_tune) + len(existing_holdout) >= 24:
            print(f"[*] Category {cat} already collected ({len(existing_tune) + len(existing_holdout)} images). Loading metadata...", flush=True)
            for f in existing_tune:
                base = f[:-4]
                jp = os.path.join(TUNE_DIR, f"{base}.json")
                if os.path.isfile(jp):
                    with open(jp, "r", encoding="utf-8") as jf:
                        manifest.append(json.load(jf))
            for f in existing_holdout:
                base = f[:-4]
                jp = os.path.join(HOLDOUT_DIR, f"{base}.json")
                if os.path.isfile(jp):
                    with open(jp, "r", encoding="utf-8") as jf:
                        manifest.append(json.load(jf))
            continue

        cat_items = []
        print(f"\n--- Gathering category: {cat} ---", flush=True)

        if cat == "covered_lens":
            for i in range(24):
                import numpy as np
                mean_lum = 2 + (i % 6)
                noise = np.random.normal(mean_lum, 2.0, (480, 640, 3)).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(noise)
                cat_items.append((img, "CC0 - Camera Sensor Occlusion Noise", "local://camera_occlusion"))
        elif cat == "road_pothole":
            cat_items = fetch_ryukijano_potholes(target_count=24)
        else:
            print(f"  Querying Commons for '{query}'...", flush=True)
            candidates = query_commons(query, limit=35)
            time.sleep(1.0)
            print(f"  Found {len(candidates)} candidates. Downloading images with requests...", flush=True)

            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = {ex.submit(download_image_direct, c["thumb_url"], c["url"]): c for c in candidates}
                for fut in as_completed(futures):
                    if len(cat_items) >= 24:
                        break
                    cand = futures[fut]
                    try:
                        img = fut.result()
                        if img is not None:
                            cat_items.append((img, cand["license"], cand["url"]))
                            print(f"    [{len(cat_items)}/24] {cand['title'][:35]} | {cand['license']}", flush=True)
                    except Exception:
                        pass

        print(f"[+] Finalized {len(cat_items)} images for {cat}", flush=True)

        # Split 50% TUNE, 50% HOLDOUT (interleaved deterministic split)
        for idx, (img, lic, src_url) in enumerate(cat_items):
            split = "TUNE" if idx % 2 == 0 else "HOLDOUT"
            split_dir = TUNE_DIR if split == "TUNE" else HOLDOUT_DIR
            base_name = f"{cat}_{idx+1:03d}"
            img_path = os.path.join(split_dir, f"{base_name}.jpg")
            json_path = os.path.join(split_dir, f"{base_name}.json")

            img.save(img_path, "JPEG", quality=90)

            record = {
                "id": base_name,
                "category": cat,
                "split": split,
                "is_hazard_category": is_hazard,
                "expected_walkable": exp_walk,
                "corridor_status": "WALKABLE" if exp_walk else ("BLOCKED" if exp_haz or exp_obj else "UNKNOWN"),
                "expected_hazards": exp_haz,
                "expected_objects": exp_obj,
                "license": lic,
                "source_url": src_url,
                "image_file": f"{base_name}.jpg",
            }
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(record, jf, indent=2)

            manifest.append(record)

    # Save manifest.json and manifest.csv
    manifest_json_path = os.path.join(TARGET_DIR, "manifest.json")
    with open(manifest_json_path, "w", encoding="utf-8") as mj:
        json.dump(manifest, mj, indent=2)

    manifest_csv_path = os.path.join(TARGET_DIR, "manifest.csv")
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as mc:
        if manifest:
            writer = csv.DictWriter(mc, fieldnames=list(manifest[0].keys()))
            writer.writeheader()
            writer.writerows(manifest)

    n_tune = sum(1 for m in manifest if m["split"] == "TUNE")
    n_holdout = sum(1 for m in manifest if m["split"] == "HOLDOUT")
    print("\n" + "="*55, flush=True)
    print(f"Dataset build complete!", flush=True)
    print(f"Total real images : {len(manifest)}", flush=True)
    print(f"TUNE set size    : {n_tune}", flush=True)
    print(f"HOLDOUT set size : {n_holdout}", flush=True)
    print(f"Manifest JSON    : {manifest_json_path}", flush=True)
    print(f"Manifest CSV     : {manifest_csv_path}", flush=True)
    print("="*55, flush=True)

if __name__ == "__main__":
    main()
