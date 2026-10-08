import json
from pathlib import Path

def generate_review_html(split_name, manifest_path, output_path, img_dir_rel):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    html = [f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Dataset Label Review Sheet - {split_name.upper()}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }}
  h1 {{ margin-bottom: 8px; font-size: 24px; }}
  p.subtitle {{ color: #94a3b8; font-size: 14px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }}
  .thumb-wrap {{ width: 100%; height: 180px; background: #090d16; display: flex; align-items: center; justify-content: center; }}
  .thumb-wrap img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
  .info {{ padding: 12px; font-size: 12px; flex: 1; display: flex; flex-direction: column; gap: 4px; }}
  .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 10px; width: fit-content; }}
  .badge-walkable {{ background: #065f46; color: #34d399; }}
  .badge-blocked {{ background: #991b1b; color: #f87171; }}
  .badge-unknown {{ background: #854d0e; color: #fde047; }}
  .meta-row {{ display: flex; justify-content: space-between; color: #94a3b8; }}
  .objects {{ color: #38bdf8; word-break: break-all; margin-top: 4px; }}
  .hazards {{ color: #fb923c; margin-top: 2px; }}
</style>
</head>
<body>
<h1>Label Review Sheet: {split_name.upper()} ({len(items)} images)</h1>
<p class="subtitle">Verification sheet for visual inspection. Category, expected corridor status, walkable flag, expected hazards, and pseudo-labelled COCO objects.</p>
<div class="grid">
''']

    for item in items:
        status = item.get('corridor_status', 'UNKNOWN')
        badge_cls = 'badge-walkable' if status == 'WALKABLE' else ('badge-blocked' if status == 'BLOCKED' else 'badge-unknown')
        img_src = f"{img_dir_rel}/{item['category']}/{item['image_file']}"
        objs = ', '.join(item.get('expected_objects', [])) or 'none'
        hazards = ', '.join(item.get('expected_hazards', [])) or 'none'
        
        html.append(f'''  <div class="card">
    <div class="thumb-wrap">
      <img src="{img_src}" alt="{item['id']}" loading="lazy">
    </div>
    <div class="info">
      <div class="meta-row">
        <strong>{item['id']}</strong>
        <span class="badge {badge_cls}">{status}</span>
      </div>
      <div class="meta-row">
        <span>Cat: {item['category']}</span>
        <span>Walkable: {item.get('expected_walkable', False)}</span>
      </div>
      <div class="hazards">Hazards: {hazards}</div>
      <div class="objects">Objects (pseudo): {objs}</div>
      <div style="color:#64748b; font-size:10px; margin-top:auto;">Source: {item.get('label_source', 'yolov8x_prefill')} | Verified: {item.get('verified', False)}</div>
    </div>
  </div>''')

    html.append('''</div>
</body>
</html>''')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"Review sheet created: {output_path}")

if __name__ == "__main__":
    Path("tests/review_sheets").mkdir(parents=True, exist_ok=True)
    generate_review_html("tune", "tests/fixtures/real/tune/manifest.json", "tests/review_sheets/review_tune.html", "../fixtures/real/tune")
    generate_review_html("holdout", "tests/fixtures/real/holdout/manifest.json", "tests/review_sheets/review_holdout.html", "../fixtures/real/holdout")
