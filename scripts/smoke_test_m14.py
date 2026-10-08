"""Quick smoke-test for M14 v2 against synthetic scenarios."""
import sys, numpy as np, cv2
sys.path.insert(0, '.')
from spatialvector.freespace import FreeSpaceEstimator, CorridorStatus

rng = np.random.default_rng(0)

def tf(h, w, base=140, var=30):
    img = rng.integers(max(0, base-var), min(255, base+var), (h, w, 3), dtype=np.uint8)
    return np.clip(img.astype(np.int16) + rng.integers(-8, 8, (h, w, 3)), 0, 255).astype(np.uint8)

est = FreeSpaceEstimator(smoothing_window=1)
h, w = 480, 640

cases = []

# Table edge
img = tf(h, w, 180)
img[:280, :] = tf(280, w, 200)
img[280:, :] = tf(h-280, w, 120)
cv2.line(img, (0, 280), (w, 280), (10, 10, 10), 6)
r = est.estimate(img); est.reset()
cases.append(("TABLE EDGE", r.centre, r.centre_conf, r.reasons["centre"], CorridorStatus.UNKNOWN))

# Table corner
img2 = tf(h, w, 180)
img2[:310, :] = tf(310, w, 210)
cv2.line(img2, (0, 310), (w, 310), (5, 5, 5), 5)
cv2.line(img2, (w//2, 0), (w//2, 310), (5, 5, 5), 5)
r2 = est.estimate(img2); est.reset()
cases.append(("TABLE CORNER", r2.centre, r2.centre_conf, r2.reasons["centre"], CorridorStatus.UNKNOWN))

# Dark object on floor
img3 = tf(h, w, 160)
img3[320:420, 250:390] = 15
r3 = est.estimate(img3); est.reset()
cases.append(("DARK OBJECT", r3.centre, r3.centre_conf, r3.reasons["centre"], CorridorStatus.UNKNOWN))

# Clear textured corridor
img4 = rng.integers(100, 160, (h, w, 3), dtype=np.uint8)
r4 = est.estimate(img4); est.reset()
cases.append(("CLEAR CORRIDOR", r4.centre, r4.centre_conf, r4.reasons["centre"], CorridorStatus.WALKABLE))

# Black frame
r5 = est.estimate(np.zeros((h, w, 3), dtype=np.uint8)); est.reset()
cases.append(("BLACK FRAME", r5.centre, r5.centre_conf, r5.reasons["centre"], CorridorStatus.UNKNOWN))

# Outdoor footpath
img6 = np.full((h, w, 3), (130, 155, 180), dtype=np.uint8)
img6[260:, :] = tf(h-260, w, 140, 20)
r6 = est.estimate(img6); est.reset()
cases.append(("OUTDOOR PATH", r6.centre, r6.centre_conf, r6.reasons["centre"], CorridorStatus.WALKABLE))

print("\n=== M14 v2 Smoke Test ===")
all_pass = True
for name, status, conf, reason, expected in cases:
    ok = (status == expected) or (expected == CorridorStatus.UNKNOWN and status != CorridorStatus.WALKABLE)
    tag = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  {tag}  {name:<20} got={status.value:<10} exp={expected.value:<10} conf={conf:.2f}  | {reason}")
print()
print("ALL PASS" if all_pass else "FAILURES ABOVE")
sys.exit(0 if all_pass else 1)
