"""SpatialVector-HMI — Demo Script Validation Harness (Section 12 / Gate G).

Runs all 6 thesis-proving demonstration scenes through the real M01-M09 pipeline,
records session traces with M12 SessionLogger, and asserts on:
- Track detection completeness (anti-vacuous test rule)
- Predictive trajectory intersection flags (TTC/CPA)
- Risk engine state convergence
- Haptic command dispatch patterns

Usage:
    python demo/validate_scenarios.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo.scenarios import SCENARIOS, DemoScenario
from demo.scene_generators import DemoScenarioRunner, ensure_fixtures
from spatialvector.hmi.logger import SessionLogger


def validate_scenario(scenario: DemoScenario, num_frames: int = 35) -> Tuple[bool, str, dict]:
    """Runs a single scenario through M01-M09 and validates against all expectations.

    Returns:
        (passed: bool, message: str, final_metrics: dict)
    """
    session_id = f"val_{scenario.id}_{int(time.time())}"
    logger = SessionLogger(session_id=session_id, base_dir="sessions")

    try:
        # Run real M01-M09 chain
        trace = DemoScenarioRunner.run_scenario(scenario.id, num_frames=num_frames)

        all_track_ids = set()
        predictions_history: List[dict] = []
        risk_states: List[dict] = []
        haptic_commands: List[dict] = []

        for fid, tracks, preds, risk_st, cmd in trace:
            for t in tracks:
                all_track_ids.add(t.track_id)
            for p in preds:
                predictions_history.append(asdict(p))

            risk_dict = asdict(risk_st)
            cmd_dict = asdict(cmd)

            risk_states.append(risk_dict)
            haptic_commands.append(cmd_dict)

            # Record in M12 Logger
            logger.log_event("predictions", preds, frame_id=fid)
            logger.log_event("risk", risk_st, frame_id=fid)
            logger.log_event("haptic", cmd, frame_id=fid)

        # 1. Anti-vacuous check: Verify track count
        if len(all_track_ids) < scenario.expect_min_tracks:
            err_msg = (
                f"FAILED: Observed {len(all_track_ids)} tracks, "
                f"expected at least {scenario.expect_min_tracks} (anti-vacuous rule violated)"
            )
            return False, err_msg, {"session_file": str(logger.session_file)}

        # 2. Steady-state check (latter half of scenario after hysteresis settling)
        settled_slice_start = max(0, len(risk_states) - 12)
        settled_risks = risk_states[settled_slice_start:]
        final_risk = risk_states[-1]
        final_state = final_risk.get("state", "UNKNOWN")

        if final_state not in scenario.expect_state_in:
            err_msg = (
                f"FAILED: Steady-state risk is '{final_state}', "
                f"expected one of {scenario.expect_state_in}"
            )
            return False, err_msg, {"session_file": str(logger.session_file), "final_state": final_state}

        # 3. Intersection flag check
        observed_intersection = None
        if scenario.expect_intersection is not None:
            # Check predictions in settled slice for the primary object
            settled_preds = [p for p in predictions_history if p.get("frame_id", 0) >= settled_slice_start]
            if settled_preds:
                observed_intersection = any(p.get("intersection_flag", False) for p in settled_preds)
            else:
                observed_intersection = False

            if observed_intersection != scenario.expect_intersection:
                err_msg = (
                    f"FAILED: Observed intersection_flag={observed_intersection}, "
                    f"expected {scenario.expect_intersection}"
                )
                return False, err_msg, {
                    "session_file": str(logger.session_file),
                    "final_state": final_state,
                    "intersection": observed_intersection,
                }

        # 4. Haptic pattern check
        haptic_matched = True
        matching_pattern = None
        if scenario.expect_haptic_pattern_prefix is not None:
            prefix = scenario.expect_haptic_pattern_prefix
            matching_commands = [
                c for c in haptic_commands if c.get("pattern_id", "").startswith(prefix) or c.get("direction", "") == prefix
            ]
            if not matching_commands:
                err_msg = (
                    f"FAILED: No haptic command matched prefix '{prefix}'. "
                    f"Produced: {[c.get('pattern_id') for c in haptic_commands[-5:]]}"
                )
                return False, err_msg, {
                    "session_file": str(logger.session_file),
                    "final_state": final_state,
                    "haptic": "NO_MATCH",
                }
            matching_pattern = matching_commands[-1].get("pattern_id")

        # Success summary
        details = [f"state={final_state}"]
        if observed_intersection is not None:
            details.append(f"intersection={observed_intersection}")
        if matching_pattern:
            details.append(f"haptic={matching_pattern}")
        elif haptic_commands:
            details.append(f"haptic={haptic_commands[-1].get('pattern_id')}")

        return True, ", ".join(details), {
            "session_file": str(logger.session_file),
            "final_state": final_state,
            "intersection": observed_intersection,
            "haptic_pattern": matching_pattern or (haptic_commands[-1].get("pattern_id") if haptic_commands else None),
            "tracks_observed": len(all_track_ids),
        }

    finally:
        logger.close()


import json


def run_all_validation(json_output: bool = False) -> Dict[str, dict]:
    """Runs all 6 demo scenarios, prints formatted report, and returns results dictionary."""
    if not json_output:
        print("=" * 76)
        print("      SpatialVector-HMI -- Demo Script Validation Harness (Gate G)      ")
        print("=" * 76)
        print(f"Validating {len(SCENARIOS)} demonstration scenes against real M01-M09 pipeline...\n")

    ensure_fixtures()

    results = {}
    all_passed = True

    for scen in SCENARIOS:
        passed, msg, metrics = validate_scenario(scen)
        status_tag = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
        results[scen.id] = {
            "passed": passed,
            "name": scen.name,
            "state": metrics.get("final_state"),
            "intersection": metrics.get("intersection"),
            "haptic": metrics.get("haptic_pattern"),
            "session_file": metrics.get("session_file"),
            "message": msg,
        }
        if not passed:
            all_passed = False

        if not json_output:
            print(f"{status_tag} {scen.id:<28} -- {msg}")
            print(f"       Thesis: \"{scen.judge_takeaway}\"")
            if not passed and metrics.get("session_file"):
                print(f"       Trace:  {metrics.get('session_file')}")
            print()

    if json_output:
        print(json.dumps(results, indent=2))
    else:
        print("-" * 76)
        if all_passed:
            print("\033[92m[OK] ALL 6 DEMO SCENARIOS VALIDATED SUCCESSFULLY.\033[0m")
            print("  All scene expectations verified end-to-end through perception, decision & haptics.")
        else:
            print("\033[91m[X] VALIDATION FAILED -- Review trace logs above for mismatched expectations.\033[0m")
        print("=" * 76)

    return results


def main():
    json_mode = "--json" in sys.argv
    results = run_all_validation(json_output=json_mode)
    all_passed = all(r["passed"] for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
