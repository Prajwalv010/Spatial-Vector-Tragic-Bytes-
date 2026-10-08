"""Replace the 9 audited mismatched fixtures using original Wikimedia URLs."""
import json
from pathlib import Path
import urllib.parse
import urllib.request
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def fetch_orig_url(query: str) -> str:
    base_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f'filetype:bitmap {query}',
        "gsrnamespace": "6",
        "gsrlimit": "5",
        "prop": "imageinfo",
        "iiprop": "url",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "SpatialVector-HMI/2.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())
    pages = d.get("query", {}).get("pages", {})
    for pid, pdata in pages.items():
        ii = pdata.get("imageinfo", [{}])[0]
        u = ii.get("url")
        if u and not u.endswith(".svg") and not u.endswith(".tif"):
            return u
    return ""

def main():
    tune_dir = ROOT / "tests" / "fixtures" / "real" / "tune"
    
    replacements = {
        "indoor_floor_002.jpg": "parquet floor wood interior",
        "indoor_floor_004.jpg": "ceramic tile floor interior",
        "indoor_floor_010.jpg": "terrazzo floor tiles building",
        "indoor_floor_024.jpg": "vinyl floor interior surface",
        "indoor_floor_003.jpg": "hardwood flooring interior plank",
        "blank_wall_002.jpg": "white painted drywall wall interior",
        "table_edge_014.jpg": "wooden table edge close up surface",
        "table_edge_020.jpg": "kitchen table edge surface indoor",
        "table_edge_017.jpg": "office desk edge border surface",
    }

    print("[*] Downloading genuine category images for 9 mismatched fixtures...")
    for fname, query in replacements.items():
        url = fetch_orig_url(query)
        if not url:
            print(f"  [WARN] No URL found for {fname}")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "SpatialVector-HMI/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=12) as r:
                data = r.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                if max(h, w) > 960:
                    scale = 960.0 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                dest = tune_dir / fname
                cv2.imwrite(str(dest), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                print(f"  [REPLACED] {fname} with genuine {query} [URL: {url[:50]}...]")
        except Exception as e:
            print(f"  [WARN] Download failed for {fname}: {e}")

if __name__ == '__main__':
    main()
