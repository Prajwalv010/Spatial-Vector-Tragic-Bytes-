"""SpatialVector-HMI — Gate G Cold-Start Reproducibility Check.

Section 12 / Gate G:
Verifies that the full 6-scene demo suite produces identical, deterministic,
and reproducible behavior when executed from a completely clean process cold-start.

This check executes two independent runs in separate OS subprocesses, diffs the
resulting steady-state risk states, trajectory intersections, and haptic outputs,
and fails loudly if any state fails to reproduce.

Usage:
    python demo/run_gate_g.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent


def execute_validation_subprocess(run_index: int) -> Dict[str, dict]:
    """Spawns an isolated Python subprocess to execute validate_scenarios.py."""
    print(f"[*] Spawning clean cold-start process for Run #{run_index}...")
    start_t = time.monotonic()

    proc = subprocess.run(
        [sys.executable, str(ROOT / "demo" / "validate_scenarios.py"), "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    elapsed = time.monotonic() - start_t
    if proc.returncode != 0:
        print(f"[!] Subprocess Run #{run_index} exited with error code {proc.returncode}!")
        if proc.stderr:
            print("--- STDERR ---")
            print(proc.stderr.strip())
        if proc.stdout:
            print("--- STDOUT ---")
            print(proc.stdout.strip())
        raise RuntimeError(f"Run #{run_index} failed execution.")

    try:
        data = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        print(f"[!] Failed to parse JSON output from Run #{run_index}: {exc}")
        print("Raw output:")
        print(proc.stdout)
        raise

    print(f"[+] Run #{run_index} completed cleanly in {elapsed:.2f}s.\n")
    return data


def run_gate_g():
    print("=" * 76)
    print("      SpatialVector-HMI -- Gate G: Cold-Start Verification Suite       ")
    print("=" * 76)
    print("Running two full independent passes via isolated OS subprocesses...\n")

    # Run 1
    run_1 = execute_validation_subprocess(1)

    # Cooldown between runs to guarantee completely released OS resources & file locks
    print("[*] Cold-start reset cooldown (500ms)...")
    time.sleep(0.5)

    # Run 2
    run_2 = execute_validation_subprocess(2)

    print("=" * 76)
    print("         Gate G Differential Analysis (Run 1 vs Run 2 Cold Start)      ")
    print("=" * 76)

    mismatches = []
    all_scenarios = list(run_1.keys())

    for scen_id in all_scenarios:
        res1 = run_1.get(scen_id, {})
        res2 = run_2.get(scen_id, {})

        state1, state2 = res1.get("state"), res2.get("state")
        pass1, pass2 = res1.get("passed"), res2.get("passed")
        int1, int2 = res1.get("intersection"), res2.get("intersection")
        hap1, hap2 = res1.get("haptic"), res2.get("haptic")

        has_diff = False
        diff_reasons = []

        if not pass1 or not pass2:
            has_diff = True
            diff_reasons.append(f"pass status mismatch (run1={pass1}, run2={pass2})")

        if state1 != state2:
            has_diff = True
            diff_reasons.append(f"state mismatch (run1={state1}, run2={state2})")

        if int1 != int2:
            has_diff = True
            diff_reasons.append(f"intersection mismatch (run1={int1}, run2={int2})")

        if hap1 != hap2:
            has_diff = True
            diff_reasons.append(f"haptic pattern mismatch (run1={hap1}, run2={hap2})")

        if has_diff:
            mismatches.append((scen_id, diff_reasons))
            print(f"[GATE G FAILURE] {scen_id:<28} -- NOT reproducible from cold start!")
            for r in diff_reasons:
                print(f"                 - {r}")
        else:
            print(f"[REPRODUCIBLE]   {scen_id:<28} -- state={state1}, haptic={hap1}")

    print("-" * 76)
    if mismatches:
        print(f"[X] GATE G VERIFICATION FAILED: {len(mismatches)} scenario(s) varied across cold starts.")
        sys.exit(1)
    else:
        print("[OK] GATE G VERIFICATION PASSED.")
        print("     All 6 demo scenarios produce 100% deterministic & reproducible output")
        print("     across fresh process cold-starts.")
        print("=" * 76)
        sys.exit(0)


if __name__ == "__main__":
    run_gate_g()
