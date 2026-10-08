# SpatialVector-HMI: Mathematical & Geometric Specifications

This document details the exact mathematical formulas, collision prediction mechanics, and risk scoring algorithms used across the SpatialVector-HMI pipeline, along with their source code locations in this repository.

---

## 1. Coordinate System & Relative Motion

* **Reference Implementation:** [`spatialvector/decision/prediction.py`](../spatialvector/decision/prediction.py) (Lines 180–205)

The user is positioned at the local origin `(0.0, 0.0)`. Ego-motion compensation (from optical flow and IMU) is pre-subtracted upstream in module **M05**, isolating true obstacle motion relative to the user.

* **Relative Position Vector:**
  $$\mathbf{r} = [x_{\text{rel}},\, y_{\text{rel}}]$$
  Where:
  * `x_rel` = Lateral offset derived from obstacle bearing angle: `sin(bearing)`
  * `y_rel` = Forward distance estimated from bounding box proximity scale: `0.35 / proximity_scale`

* **Relative Velocity Vector:**
  $$\mathbf{v} = [v_x,\, -v_{\text{approach}}]$$
  Where:
  * `v_x` = Relative lateral velocity across the image frame.
  * `v_approach` = Forward approach speed, computed from looming bounding-box expansion rate: `y_rel × expansion_rate`.

---

## 2. Closest Point of Approach (CPA)

* **Reference Implementation:** [`spatialvector/decision/prediction.py`](../spatialvector/decision/prediction.py) (Lines 360–373)

The system computes the exact timestamp `t_cpa` where distance between user and obstacle reaches its global minimum (where the derivative of squared distance equals zero).

### Time to CPA ($t_{\text{cpa}}$)
```
t_cpa = - (r · v) / |v|²
```
Expanded form:
```
t_cpa = - (x_rel * v_x + y_rel * v_y) / (v_x² + v_y²)
```

### Motion Classification
| Condition | Interpretation | Pipeline Action |
| :--- | :--- | :--- |
| `t_cpa ≤ 0` | Obstacle is **receding** (moving away or passed) | `TTC = None`, `Intersection = False` |
| `t_cpa > 0` | Obstacle is **approaching** in forward time | Evaluates Corridor Intersection & TTC |

### Distance at CPA ($d_{\text{cpa}}$)
```
d_cpa = |r + v * t_cpa|
```

---

## 3. Path Intersection & Time-To-Collision (TTC)

* **Reference Implementation:** [`spatialvector/decision/prediction.py`](../spatialvector/decision/prediction.py) (Lines 374–429)

### Intersection Condition
An obstacle trajectory is marked as an **active intersection** (`intersection_flag = True`) if and only if both conditions are met:
1. **Corridor Containment:** `d_cpa < Corridor_Width` *(Default: 0.12 normalized units)*
2. **Horizon Containment:** `t_cpa ≤ Time_Horizon` *(Default: 5.0 seconds)*

### Quadratic Contact Envelope (TTC)
TTC is defined as the first moment `t > 0` when obstacle boundary touches the contact radius `R_contact = 0.05`:
```
|r + v * t|² = R_contact²
```
Solving the quadratic equation `a * t² + b * t + c = 0`:
```
a = |v|²
b = 2 * (r · v)
c = |r|² - R_contact²
Discriminant (Δ) = b² - 4*a*c
```

```
TTC = min { t > 0 | t = (-b - √Δ) / (2*a) }
```
*(If Δ < 0, trajectories do not touch within contact radius; TTC is set to None).*

### Visual Looming Expansion (Head-On Collision Fallback)
When an obstacle approaches directly head-on with minimal lateral velocity:
```
TTC_looming = 1.0 / expansion_rate
```

---

## 4. Multi-Factor Risk Engine (M08)

* **Reference Implementation:** [`spatialvector/decision/risk_engine.py`](../spatialvector/decision/risk_engine.py) (Lines 350–405)
* **Configuration Defaults:** [`spatialvector/config/default.yaml`](../spatialvector/config/default.yaml)

Risk is not based on arbitrary object labels (e.g. car vs person), but purely on geometric hazard metrics.

### Per-Object Risk Formula ($R_i$)
```
Risk_i = (w_ttc * S_ttc) + (w_miss * S_miss) + (w_int * S_int)
```

| Component | Weight | Formula | Description |
| :--- | :---: | :--- | :--- |
| **TTC Score ($S_{\text{ttc}}$)** | **0.50** (50%) | `clip(1.0 - TTC / Horizon, 0.0, 1.0)` | Closer in time = higher urgency |
| **Miss Distance ($S_{\text{miss}}$)** | **0.30** (30%) | `clip(1.0 - Miss_Distance / 0.5, 0.0, 1.0)` | Closer spatial approach = higher risk |
| **Intersection ($S_{\text{int}}$)** | **0.20** (20%) | `Prediction_Confidence` *(if intersecting)* | Confidence-weighted path crossing |

*Direct Proximity Override:* If an obstacle directly occupies the forward walking path (`proximity_risk > 0.25`), `Risk_i = max(Risk_i, proximity_risk)`.

---

## 5. Global Risk & Corridor Distribution

* **Reference Implementation:** [`spatialvector/decision/risk_engine.py`](../spatialvector/decision/risk_engine.py) (Lines 304–320, 407–439)

### Global Risk Aggregation
To prevent multiple safe background objects from diluting a single critical collision:
```
Global_Risk = max(Risk_1, Risk_2, ..., Risk_n)
```

### Corridor Risk Distribution
The camera Field of View is segmented into three bearing corridors:
* **Left:** `bearing < -30°` (`< -0.52 rad`)
* **Center:** `-30° ≤ bearing ≤ +30°`
* **Right:** `bearing > +30°` (`> +0.52 rad`)

Corridor scores combine multiple hazards using a dominant-plus-secondary weighted sum:
```
Corridor_Risk = clip(Primary_Hazard_Risk + 0.5 * Sum(Secondary_Hazards_Risk), 0.0, 1.0)
```

---

## 6. Hysteresis State Machine & Thresholds

* **Reference Implementation:** [`spatialvector/decision/risk_engine.py`](../spatialvector/decision/risk_engine.py) (Lines 480–525)

To prevent rapid haptic chattering and alert flickering, state transitions require temporal persistence.

```
       Global Risk ≥ 0.30            Global Risk ≥ 0.60            Global Risk ≥ 0.85
  SAFE ───────────────────► CAUTION ───────────────────► WARNING ───────────────────► CRITICAL
       ◄───────────────────         ◄───────────────────         ◄───────────────────
       Global Risk < 0.30            Global Risk < 0.60            Global Risk < 0.85
```

### Escalation & De-escalation Rules
| Alert State | Activation Threshold | Escalation Debounce | De-escalation Debounce |
| :--- | :---: | :---: | :---: |
| **SAFE** | `Risk < 0.30` | — | 5 consecutive frames below |
| **CAUTION** | `Risk ≥ 0.30` | 3 consecutive frames above | 5 consecutive frames below |
| **WARNING** | `Risk ≥ 0.60` | 3 consecutive frames above | 5 consecutive frames below |
| **CRITICAL** | `Risk ≥ 0.85` | 3 consecutive frames above | 5 consecutive frames below |
| **DEGRADED** | `Confidence < 0.25` | Immediate fallback flag | Cleared upon sensor recovery |
