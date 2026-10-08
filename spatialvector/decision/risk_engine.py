"""
M08 — Risk Engine & State Machine

The SINGLE SOURCE OF TRUTH for warning state. Every other module reads RiskState.state;
no other module is permitted to independently compute "is this dangerous."

Responsibilities:
  1. Aggregate per-object Predictions into one global_risk score (0..1).
  2. Map global_risk to a state string via configurable thresholds.
  3. Apply hysteresis/debounce so state does not flicker at threshold crossings.
  4. Compute per-corridor risk for M09 to consume.
  5. Produce explainable reason_codes on every state transition.
  6. Emit DEGRADED state when confidence is systemically too low to trust any output.

Rules enforced in code:
  - No class_name affects risk. A "person" is not inherently riskier than a "chair."
    Risk comes from geometry (TTC, CPA, intersection), never from class labels.
  - TTC alone does not determine state. A low TTC with intersection_flag=False is
    a near-miss, not a collision prediction.
  - Hysteresis is mandatory: escalation requires hysteresis_frames_up consecutive
    frames above threshold; de-escalation requires hysteresis_frames_down frames below.

All weights and thresholds live in config/default.yaml under "risk_engine:".
They are NOT hardcoded here — this is exactly the block that gets tuned in the
last hour of a hackathon and must be a config change, not a code change.

Config values (from config/default.yaml, section "risk_engine"):
  weight_ttc:                       0.5  — weight of TTC component in per-object risk
  weight_miss_distance:             0.3  — weight of miss distance component
  weight_intersection_confidence:   0.2  — weight of intersection + confidence signal
  state_thresholds.caution:         0.3
  state_thresholds.warning:         0.6
  state_thresholds.critical:        0.85
  hysteresis_frames_up:             3    — frames above threshold before escalating
  hysteresis_frames_down:           5    — frames below threshold before de-escalating
  degraded_confidence_threshold:    0.25 — if avg prediction_confidence below this,
                                           state → DEGRADED
  corridor_width_left:              [0.0, 0.33]  — normalized bearing range for left corridor
  corridor_width_center:            [0.33, 0.67] — center
  corridor_width_right:             [0.67, 1.0]  — right (by abs(bearing/pi))
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import math
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Union

import numpy as np

from spatialvector.decision.schemas import Prediction, RiskState
from spatialvector.decision.corridor_constants import CORRIDOR_CENTER_BEARING_RAD
from spatialvector.motion.temporal_smoother import AdaptiveEMA

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Default config values — mirror what's in default.yaml
# -------------------------------------------------------------------------
_DEFAULT_WEIGHT_TTC = 0.5
_DEFAULT_WEIGHT_MISS_DISTANCE = 0.3
_DEFAULT_WEIGHT_INTERSECTION_CONF = 0.2
_DEFAULT_THRESHOLDS = {"caution": 0.3, "warning": 0.6, "critical": 0.85}
_DEFAULT_HYSTERESIS_UP = 3
_DEFAULT_HYSTERESIS_DOWN = 5
_DEFAULT_DEGRADED_CONF_THRESHOLD = 0.25
_DEFAULT_HORIZON_S = 5.0

# Corridor bearing boundaries (radians)
# We map: left < -CORRIDOR_CENTER_BEARING_RAD, center -CORRIDOR_CENTER_BEARING_RAD..+CORRIDOR_CENTER_BEARING_RAD, right > +CORRIDOR_CENTER_BEARING_RAD
_LEFT_BOUNDARY = -CORRIDOR_CENTER_BEARING_RAD   # -30 degrees (-0.5236 rad)
_RIGHT_BOUNDARY = CORRIDOR_CENTER_BEARING_RAD   #  30 degrees ( 0.5236 rad)


@dataclass(frozen=True)
class RiskEngineConfig:
    """Immutable configuration snapshot for M08 RiskEngine.
    
    Provides thread-safe atomic updates without lock contention in the hot loop.
    """
    weight_ttc: float = _DEFAULT_WEIGHT_TTC
    weight_miss_distance: float = _DEFAULT_WEIGHT_MISS_DISTANCE
    weight_intersection_confidence: float = _DEFAULT_WEIGHT_INTERSECTION_CONF
    state_thresholds: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_THRESHOLDS))
    hysteresis_frames_up: int = _DEFAULT_HYSTERESIS_UP
    hysteresis_frames_down: int = _DEFAULT_HYSTERESIS_DOWN
    degraded_confidence_threshold: float = _DEFAULT_DEGRADED_CONF_THRESHOLD
    horizon_s: float = _DEFAULT_HORIZON_S

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskEngineConfig':
        thresholds = data.get("state_thresholds")
        if thresholds is not None and not isinstance(thresholds, dict):
            raise ValueError("state_thresholds must be a dictionary")
        thresh_dict = dict(thresholds) if thresholds else dict(_DEFAULT_THRESHOLDS)
        return cls(
            weight_ttc=float(data.get("weight_ttc", _DEFAULT_WEIGHT_TTC)),
            weight_miss_distance=float(data.get("weight_miss_distance", _DEFAULT_WEIGHT_MISS_DISTANCE)),
            weight_intersection_confidence=float(data.get("weight_intersection_confidence", _DEFAULT_WEIGHT_INTERSECTION_CONF)),
            state_thresholds=thresh_dict,
            hysteresis_frames_up=int(data.get("hysteresis_frames_up", _DEFAULT_HYSTERESIS_UP)),
            hysteresis_frames_down=int(data.get("hysteresis_frames_down", _DEFAULT_HYSTERESIS_DOWN)),
            degraded_confidence_threshold=float(data.get("degraded_confidence_threshold", _DEFAULT_DEGRADED_CONF_THRESHOLD)),
            horizon_s=float(data.get("horizon_s", _DEFAULT_HORIZON_S)),
        )


class RiskEngine:
    """Aggregates Predictions into a stable, explainable RiskState.

    Usage:
        engine = RiskEngine()
        risk_state = engine.update(predictions, fallback_active=False, timestamp=t)
    """

    def __init__(
        self,
        weight_ttc: float = _DEFAULT_WEIGHT_TTC,
        weight_miss_distance: float = _DEFAULT_WEIGHT_MISS_DISTANCE,
        weight_intersection_confidence: float = _DEFAULT_WEIGHT_INTERSECTION_CONF,
        state_thresholds: Optional[dict] = None,
        hysteresis_frames_up: int = _DEFAULT_HYSTERESIS_UP,
        hysteresis_frames_down: int = _DEFAULT_HYSTERESIS_DOWN,
        degraded_confidence_threshold: float = _DEFAULT_DEGRADED_CONF_THRESHOLD,
        horizon_s: float = _DEFAULT_HORIZON_S,
    ):
        initial_cfg = RiskEngineConfig(
            weight_ttc=weight_ttc,
            weight_miss_distance=weight_miss_distance,
            weight_intersection_confidence=weight_intersection_confidence,
            state_thresholds=dict(state_thresholds) if state_thresholds else dict(_DEFAULT_THRESHOLDS),
            hysteresis_frames_up=hysteresis_frames_up,
            hysteresis_frames_down=hysteresis_frames_down,
            degraded_confidence_threshold=degraded_confidence_threshold,
            horizon_s=horizon_s,
        )
        self.validate_config(initial_cfg)
        self._config = initial_cfg
        self._config_lock = threading.Lock()

        # State machine
        self._current_state = "SAFE"
        self._frames_above: dict[str, int] = {"caution": 0, "warning": 0, "critical": 0}
        self._frames_below: dict[str, int] = {"caution": 0, "warning": 0, "critical": 0}

        # Recent risk history for trend analysis (used by M09)
        self._risk_history: deque[float] = deque(maxlen=30)

        # Adaptive smoothers for continuous telemetry and hysteresis inputs
        self._global_risk_smoother = AdaptiveEMA(alpha_slow=0.20, alpha_fast=0.65, window_size=30, rising_is_dangerous=True)
        self._corridor_smoothers = {
            "left": AdaptiveEMA(alpha_slow=0.20, alpha_fast=0.65, window_size=30, rising_is_dangerous=True),
            "center": AdaptiveEMA(alpha_slow=0.20, alpha_fast=0.65, window_size=30, rising_is_dangerous=True),
            "right": AdaptiveEMA(alpha_slow=0.20, alpha_fast=0.65, window_size=30, rising_is_dangerous=True),
        }

        # Previous state for transition logging
        self._prev_state = "SAFE"

    # Properties mirroring config for backwards compatibility
    @property
    def config(self) -> RiskEngineConfig:
        with self._config_lock:
            return self._config

    @property
    def w_ttc(self) -> float:
        return self._config.weight_ttc

    @property
    def w_miss(self) -> float:
        return self._config.weight_miss_distance

    @property
    def w_int_conf(self) -> float:
        return self._config.weight_intersection_confidence

    @property
    def thresholds(self) -> dict[str, float]:
        return self._config.state_thresholds

    @property
    def hyst_up(self) -> int:
        return self._config.hysteresis_frames_up

    @property
    def hyst_down(self) -> int:
        return self._config.hysteresis_frames_down

    @property
    def degraded_conf_threshold(self) -> float:
        return self._config.degraded_confidence_threshold

    @property
    def horizon_s(self) -> float:
        return self._config.horizon_s

    @staticmethod
    def validate_config(cfg: Union[RiskEngineConfig, dict]) -> None:
        """Validates configuration parameters, raising ValueError on invalid values."""
        if isinstance(cfg, dict):
            cfg = RiskEngineConfig.from_dict(cfg)

        for name, val in [
            ("weight_ttc", cfg.weight_ttc),
            ("weight_miss_distance", cfg.weight_miss_distance),
            ("weight_intersection_confidence", cfg.weight_intersection_confidence),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")

        w_sum = cfg.weight_ttc + cfg.weight_miss_distance + cfg.weight_intersection_confidence
        if w_sum > 1.05:
            raise ValueError(f"Sum of weights must be <= 1.0 (got {w_sum:.3f})")

        th = cfg.state_thresholds
        for level in ("caution", "warning", "critical"):
            if level not in th:
                raise ValueError(f"state_thresholds missing required key '{level}'")
            if not (0.0 < th[level] <= 1.0):
                raise ValueError(f"Threshold '{level}' must be in (0.0, 1.0], got {th[level]}")

        if not (th["caution"] < th["warning"] < th["critical"]):
            raise ValueError(
                f"State thresholds must be strictly ascending caution < warning < critical "
                f"(got caution={th['caution']}, warning={th['warning']}, critical={th['critical']})"
            )

        if cfg.hysteresis_frames_up < 1:
            raise ValueError(f"hysteresis_frames_up must be >= 1, got {cfg.hysteresis_frames_up}")
        if cfg.hysteresis_frames_down < 1:
            raise ValueError(f"hysteresis_frames_down must be >= 1, got {cfg.hysteresis_frames_down}")
        if not (0.0 < cfg.degraded_confidence_threshold <= 1.0):
            raise ValueError(f"degraded_confidence_threshold must be in (0.0, 1.0], got {cfg.degraded_confidence_threshold}")
        if cfg.horizon_s <= 0.0:
            raise ValueError(f"horizon_s must be > 0.0, got {cfg.horizon_s}")

    def update_config(self, new_cfg: Union[RiskEngineConfig, dict]) -> RiskEngineConfig:
        """Thread-safe configuration update via atomic reference swap."""
        if isinstance(new_cfg, dict):
            new_cfg = RiskEngineConfig.from_dict(new_cfg)
        self.validate_config(new_cfg)
        with self._config_lock:
            self._config = new_cfg
        logger.info(f"[M08] Live configuration updated: {new_cfg}")
        return new_cfg

    def update(
        self,
        predictions: list[Prediction],
        fallback_active: bool = False,
        timestamp: Optional[float] = None,
    ) -> RiskState:
        """Produce one RiskState for the current frame.

        Args:
            predictions: all Prediction objects from M07 for this frame.
            fallback_active: True when M05 is in DEGRADED fallback mode.
                             Propagates into confidence scoring.
            timestamp: frame timestamp. Defaults to time.monotonic() if None.

        Returns:
            RiskState — the single source of truth for this frame's warning state.
        """
        ts = timestamp if timestamp is not None else time.monotonic()

        # Atomic config snapshot for the entire update() execution — zero torn reads
        cfg = self._config

        # --- DEGRADED check ---
        # If no predictions, or all predictions have very low confidence,
        # or upstream is in fallback mode and confidence is borderline, → DEGRADED.
        system_confidence = self._compute_system_confidence(predictions, fallback_active)
        if system_confidence < cfg.degraded_confidence_threshold:
            state = self._apply_degraded()
            corridor_risks = {"left": 0.0, "center": 0.0, "right": 0.0}
            reason_codes = [f"degraded:conf={system_confidence:.3f}"]
            if fallback_active:
                reason_codes.append("degraded:fallback_active")
            self._log_transition(state)
            return RiskState(
                timestamp=ts,
                global_risk=0.0,
                state=state,
                reason_codes=reason_codes,
                corridor_risks=corridor_risks,
                confidence=system_confidence,
                recommended_horizon_s=cfg.horizon_s,
            )

        # --- Per-object risk scores ---
        object_risks = {}   # track_id -> (risk_score, reason_parts)
        for pred in predictions:
            risk, parts = self._object_risk(pred, cfg=cfg)
            object_risks[pred.track_id] = (risk, parts)

        # --- Global risk = max of individual risks (most dangerous object drives state) ---
        # We use max rather than mean so that one critical object can't be diluted by
        # many safe objects in the background.
        if not object_risks:
            global_risk = 0.0
            reason_codes = []
        else:
            global_risk = max(r for r, _ in object_risks.values())
            # Collect reason codes from objects above a "notable" threshold
            reason_codes = []
            for tid, (risk, parts) in object_risks.items():
                if risk > cfg.state_thresholds["caution"] * 0.5:  # include anything meaningfully above baseline
                    for p in parts:
                        reason_codes.append(f"{p}:track_{tid}")

        # --- Corridor risk ---
        corridor_risks = self._compute_corridor_risks(predictions, object_risks)

        # Smooth continuous risk metrics with AdaptiveEMA
        smoothed_global_risk = float(self._global_risk_smoother.update(global_risk))
        smoothed_corridor_risks = {
            corr: float(self._corridor_smoothers[corr].update(score))
            for corr, score in corridor_risks.items()
        }

        self._risk_history.append(smoothed_global_risk)

        # --- State machine with hysteresis on smoothed signal ---
        raw_state = self._raw_state_for_risk(smoothed_global_risk, cfg=cfg)
        state = self._apply_hysteresis(raw_state, smoothed_global_risk, cfg=cfg)
        self._log_transition(state)

        return RiskState(
            timestamp=ts,
            global_risk=float(smoothed_global_risk),
            state=state,
            reason_codes=reason_codes,
            corridor_risks=smoothed_corridor_risks,
            confidence=float(system_confidence),
            recommended_horizon_s=cfg.horizon_s,
        )

    # ------------------------------------------------------------------
    # Risk computation
    # ------------------------------------------------------------------

    def _object_risk(self, pred: Prediction, cfg: Optional[RiskEngineConfig] = None) -> tuple[float, list[str]]:
        """Compute a 0..1 risk score for one Prediction, with reason codes.

        Components (each individually explainable):
          - TTC component:      higher when TTC is shorter. 0 if no TTC (ttc_s=None).
          - Miss distance:      higher when CPA distance is smaller.
          - Intersection×conf:  higher when paths cross AND prediction is confident.

        These are the three weighted terms. Weights come from config snapshot.
        """
        c = cfg or self._config
        reasons: list[str] = []
        risk = 0.0

        # --- TTC component ---
        # TTC is None when there's no finite/relevant TTC (receding, parallel, beyond horizon).
        # When TTC is valid, it should be inversely related to risk (low TTC → high risk).
        ttc_score = 0.0
        if pred.ttc_s is not None:
            # Normalize TTC: 0s → score 1.0, horizon_s → score 0.0
            ttc_score = float(np.clip(1.0 - pred.ttc_s / c.horizon_s, 0.0, 1.0))
            if ttc_score > 0.3:
                reasons.append(f"ttc_low:{pred.ttc_s:.2f}s")

        # --- Miss distance component ---
        # Smaller miss distance = higher risk. Normalized by corridor width (proxy for "close").
        # miss_distance_normalized is in the same units as corridor_width in M07.
        # We normalize against a "safe" distance of 0.5 (half of frame width).
        _safe_miss = 0.5
        miss_score = float(np.clip(1.0 - pred.miss_distance_normalized / _safe_miss, 0.0, 1.0))
        if miss_score > 0.5:
            reasons.append(f"miss_dist:{pred.miss_distance_normalized:.3f}")

        # --- Intersection × confidence component ---
        # Only contributes when paths actually cross. Scaled by prediction confidence
        # so low-confidence intersections don't drive state transitions.
        int_conf_score = 0.0
        if pred.intersection_flag:
            int_conf_score = float(np.clip(pred.prediction_confidence, 0.0, 1.0))
            reasons.append(f"intersection")

        # --- Weighted sum ---
        risk = (
            c.weight_ttc * ttc_score
            + c.weight_miss_distance * miss_score
            + c.weight_intersection_confidence * int_conf_score
        )
        # Factor in proximity risk directly (person/obstacle right in front of user)
        prox_score = getattr(pred, "proximity_risk", 0.0)
        if prox_score > 0.25:
            reasons.append(f"proximity:{prox_score:.2f}")
            risk = max(risk, prox_score)

        risk = float(np.clip(risk, 0.0, 1.0))

        return risk, reasons

    def _compute_corridor_risks(
        self,
        predictions: list[Prediction],
        object_risks: dict[int, tuple[float, list[str]]],
    ) -> dict[str, float]:
        """Compute risk per corridor.

        Uses ALL objects that fall into a given corridor, not just the closest one.
        A corridor with two moderate-risk objects can be worse than one with one low-risk object.
        Method: max-of-weighted-sum. Primary = max risk object; secondary = 0.5× second-highest.
        """
        corridor_scores: dict[str, list[float]] = {"left": [], "center": [], "right": []}

        # Corridor assignment uses Prediction.bearing directly — see _assign_corridor_from_prediction().

        for pred in predictions:
            risk_score = object_risks.get(pred.track_id, (0.0, []))[0]
            corridor = self._assign_corridor_from_prediction(pred)
            corridor_scores[corridor].append(risk_score)

        # For each corridor: use weighted sum where primary object = full weight, others = 0.5×
        result = {}
        for corr, scores in corridor_scores.items():
            if not scores:
                result[corr] = 0.0
            else:
                scores_sorted = sorted(scores, reverse=True)
                weighted = scores_sorted[0]
                for s in scores_sorted[1:]:
                    weighted += 0.5 * s
                result[corr] = float(np.clip(weighted, 0.0, 1.0))

        return result

    def _assign_corridor_from_prediction(self, pred: Prediction) -> str:
        """Assign a Prediction to left/center/right corridor.

        Uses the bearing field (from ObjectGeometry via M07) for accurate assignment.
        Corridor boundaries: left < -pi/6, center -pi/6..pi/6, right > pi/6.
        These boundaries (~30 degrees) split the FOV into three approximately equal thirds.
        """
        bearing = getattr(pred, 'bearing', 0.0)
        if bearing < _LEFT_BOUNDARY:
            return "left"
        elif bearing > _RIGHT_BOUNDARY:
            return "right"
        else:
            return "center"


    # ------------------------------------------------------------------
    # State machine with hysteresis
    # ------------------------------------------------------------------

    def _raw_state_for_risk(self, risk: float, cfg: Optional[RiskEngineConfig] = None) -> str:
        """Map risk score to state string without hysteresis."""
        c = cfg or self._config
        if risk >= c.state_thresholds["critical"]:
            return "CRITICAL"
        elif risk >= c.state_thresholds["warning"]:
            return "WARNING"
        elif risk >= c.state_thresholds["caution"]:
            return "CAUTION"
        else:
            return "SAFE"

    def _apply_hysteresis(self, raw_state: str, risk: float, cfg: Optional[RiskEngineConfig] = None) -> str:
        """Apply hysteresis: require sustained state before transitioning.

        Escalation:   need hyst_up consecutive frames above the NEXT level's threshold.
        De-escalation: need hyst_down consecutive frames below the CURRENT level's threshold.

        States advance and retreat exactly ONE level at a time, so SAFE→CRITICAL is impossible
        in a single frame (it must go SAFE→CAUTION→WARNING→CRITICAL over at least
        hyst_up * 2 frames).
        """
        c = cfg or self._config
        state_order = ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
        level_to_threshold = {
            "CAUTION": c.state_thresholds["caution"],
            "WARNING": c.state_thresholds["warning"],
            "CRITICAL": c.state_thresholds["critical"],
        }

        current_idx = state_order.index(self._current_state) if self._current_state in state_order else 0

        # --- Determine if we can escalate ONE level ---
        if current_idx < len(state_order) - 1:
            next_state = state_order[current_idx + 1]
            next_threshold = level_to_threshold.get(next_state, 1.0)
            if risk >= next_threshold:
                self._frames_above[next_state] = self._frames_above.get(next_state, 0) + 1
            else:
                self._frames_above[next_state] = 0

            if self._frames_above.get(next_state, 0) >= c.hysteresis_frames_up:
                # Escalate one level
                self._current_state = next_state
                self._frames_above[next_state] = 0   # reset after transition
                return self._current_state

        # --- Determine if we can de-escalate ONE level ---
        if current_idx > 0:
            current_state_name = state_order[current_idx]
            current_threshold = level_to_threshold.get(current_state_name, 0.0)
            if risk < current_threshold:
                self._frames_below[current_state_name] = self._frames_below.get(current_state_name, 0) + 1
            else:
                self._frames_below[current_state_name] = 0

            if self._frames_below.get(current_state_name, 0) >= c.hysteresis_frames_down:
                # De-escalate one level
                self._current_state = state_order[current_idx - 1]
                self._frames_below[current_state_name] = 0   # reset after transition
                return self._current_state

        return self._current_state


    def _apply_degraded(self) -> str:
        """Transition to DEGRADED state and reset hysteresis counters."""
        if self._current_state != "DEGRADED":
            self._current_state = "DEGRADED"
            for level in self._frames_above:
                self._frames_above[level] = 0
                self._frames_below[level] = 0
        return "DEGRADED"

    def _compute_system_confidence(
        self, predictions: list[Prediction], fallback_active: bool
    ) -> float:
        """Overall confidence in the current frame's predictions.

        Low when:
          - Upstream sensors or motion estimation are in fallback mode (fallback_active=True)
          - Tracked objects have low prediction_confidence
        When predictions is empty:
          - If fallback_active=True: sensors/flow failed → confidence=0.0 (DEGRADED)
          - If fallback_active=False: pipeline is healthy and scene is clear → confidence=1.0 (SAFE)
        """
        if not predictions:
            return 0.0 if fallback_active else 1.0

        avg_conf = float(np.mean([p.prediction_confidence for p in predictions]))

        if fallback_active:
            avg_conf *= 0.6   # fallback_caution_widen_factor ≈ 1/0.6 from config

        return float(np.clip(avg_conf, 0.0, 1.0))

    def _log_transition(self, new_state: str):
        """Log state transitions with reason codes for debugging."""
        if new_state != self._prev_state:
            logger.info(f"[M08] State transition: {self._prev_state} → {new_state}")
            self._prev_state = new_state
