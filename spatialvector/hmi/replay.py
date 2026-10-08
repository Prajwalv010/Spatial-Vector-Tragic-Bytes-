"""M12 — Session Replay & Offline Evaluation Harness.

Loads recorded JSONL sessions, validates schemas, and deterministically replays
them through the decision chain (M07 Prediction -> M08 Risk -> M09 Policy).

Supports offline A/B testing of different RiskEngine weights and thresholds
without re-recording live video footage.
"""

from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from spatialvector.decision.corridor_policy import CorridorPolicy
from spatialvector.decision.prediction import CollisionPredictor
from spatialvector.decision.risk_engine import RiskEngine
from spatialvector.decision.schemas import Prediction, RiskState, HapticCommand
from spatialvector.motion.schemas import ObjectGeometry
from spatialvector.perception.schemas import Track

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSION = 1


def parse_session_file(file_path: Union[str, Path]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parses a session.jsonl file, enforcing schema versioning and integrity.

    Raises:
        ValueError: On corrupt records, missing required fields, or unsupported schema versions.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")

    header = None
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                rec = json.loads(line_str)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt record at line {line_num}: Malformed JSON ({exc})")

            if not isinstance(rec, dict):
                raise ValueError(f"Corrupt record at line {line_num}: Record must be a JSON object")

            # Check required fields
            for req in ("session_id", "record_type", "ts"):
                if req not in rec:
                    raise ValueError(f"Corrupt record at line {line_num}: Missing required field '{req}'")

            if rec["record_type"] == "header":
                header = rec["payload"]
                version = header.get("schema_version")
                if version != SUPPORTED_SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported session schema_version {version} at line {line_num}. Expected {SUPPORTED_SCHEMA_VERSION}"
                    )
            else:
                records.append(rec)

    if header is None:
        raise ValueError(f"Session file {path} missing header record")

    return header, records


class SessionReplayer:
    """Replays recorded session events through the decision pipeline deterministically."""

    def __init__(self, session_path: Union[str, Path]):
        self.session_path = Path(session_path)
        self.header, self.records = parse_session_file(self.session_path)
        self.session_id = self.header.get("session_id", "unknown")

    def run_replay(
        self,
        risk_engine: Optional[RiskEngine] = None,
        corridor_policy: Optional[CorridorPolicy] = None,
        predictor: Optional[CollisionPredictor] = None,
    ) -> Dict[str, Any]:
        """Runs the recorded predictions or geometries through the decision chain.

        Returns replay summary with states, risk scores, and commands produced.
        """
        engine = risk_engine or RiskEngine()
        policy = corridor_policy or CorridorPolicy()

        # Group records by frame_id
        frames_data: Dict[int, Dict[str, Any]] = {}
        for rec in self.records:
            fid = rec.get("frame_id")
            if fid is None:
                continue
            if fid not in frames_data:
                frames_data[fid] = {
                    "ts": rec["ts"],
                    "predictions": [],
                    "tracks": [],
                    "geometries": [],
                    "fallback_active": False,
                }

            rtype = rec.get("record_type")
            payload = rec.get("payload", {})
            if rtype in ("prediction", "predictions", "prediction_batch"):
                pred_list = payload if isinstance(payload, list) else [payload]
                for p_dict in pred_list:
                    pred = Prediction(
                        track_id=p_dict.get("track_id", 0),
                        frame_id=fid,
                        ttc_s=p_dict.get("ttc_s"),
                        cpa_normalized=p_dict.get("cpa_normalized", 0.5),
                        miss_distance_normalized=p_dict.get("miss_distance_normalized", 0.5),
                        intersection_flag=p_dict.get("intersection_flag", False),
                        prediction_confidence=p_dict.get("prediction_confidence", 0.8),
                        bearing=p_dict.get("bearing", 0.0),
                    )
                    frames_data[fid]["predictions"].append(pred)


            elif rtype == "motion":
                frames_data[fid]["fallback_active"] = payload.get("fallback_active", False)

        ordered_frame_ids = sorted(frames_data.keys())
        states: List[str] = []
        global_risks: List[float] = []
        commands: List[HapticCommand] = []

        for fid in ordered_frame_ids:
            fdata = frames_data[fid]
            preds = fdata["predictions"]
            ts = fdata["ts"]
            fallback = fdata["fallback_active"]

            risk_state = engine.update(preds, fallback_active=fallback, timestamp=ts)
            cmd = policy.select(risk_state, timestamp=ts)

            states.append(risk_state.state)
            global_risks.append(risk_state.global_risk)
            commands.append(cmd)

        return {
            "session_id": self.session_id,
            "total_frames": len(ordered_frame_ids),
            "frame_ids": ordered_frame_ids,
            "states": states,
            "global_risks": global_risks,
            "commands": commands,
            "state_distribution": dict(Counter(states)),
        }


def compare_replays(replay_a: Dict[str, Any], replay_b: Dict[str, Any]) -> Dict[str, Any]:
    """Compares two replay evaluation runs (e.g. baseline vs modified risk engine)."""
    states_a = replay_a.get("states", [])
    states_b = replay_b.get("states", [])
    risks_a = replay_a.get("global_risks", [])
    risks_b = replay_b.get("global_risks", [])

    min_len = min(len(states_a), len(states_b))
    divergent_states = sum(1 for i in range(min_len) if states_a[i] != states_b[i])

    risk_diffs = [abs(risks_a[i] - risks_b[i]) for i in range(min_len)]
    mean_risk_diff = sum(risk_diffs) / len(risk_diffs) if risk_diffs else 0.0

    return {
        "compared_frames": min_len,
        "divergent_state_count": divergent_states,
        "divergence_percentage": (divergent_states / min_len * 100.0) if min_len else 0.0,
        "mean_risk_difference": mean_risk_diff,
        "distribution_a": replay_a.get("state_distribution", {}),
        "distribution_b": replay_b.get("state_distribution", {}),
    }
