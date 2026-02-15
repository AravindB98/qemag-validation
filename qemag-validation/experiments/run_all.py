"""
Run all QEMA-G validation experiments.

Usage:
    python experiments/run_all.py

This will execute Experiments 1-3 and save results to the results/ directory.
"""

import subprocess
import sys
import os
import time


def run_experiment(script_name: str):
    """Run a single experiment script."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n{'#' * 70}")
    print(f"# Running: {script_name}")
    print(f"{'#' * 70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {result.returncode}")
        return False
    else:
        print(f"\n[OK] {script_name} completed in {elapsed:.1f}s")
        return True


def main():
    print("=" * 70)
    print("QEMA-G Validation: Running All Experiments")
    print("=" * 70)

    experiments = [
        "experiment1_identity_vqc_path.py",
        "experiment2_trained_vqc_path.py",
        "experiment3_identity_vqc_cycle.py",
    ]

    results = {}
    total_start = time.time()

    for exp in experiments:
        success = run_experiment(exp)
        results[exp] = "PASS" if success else "FAIL"

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"{'=' * 70}")
    for exp, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {exp}: {status}")
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Results saved in: results/")

    if all(s == "PASS" for s in results.values()):
        print("\nAll experiments passed.")
        sys.exit(0)
    else:
        print("\nSome experiments failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
