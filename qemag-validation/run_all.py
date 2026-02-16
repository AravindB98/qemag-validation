"""
QEMA-G Validation: Run All Experiments
=======================================
Master script to reproduce all numerical results from the paper.

Usage:
    python run_all.py           # Run all experiments
    python run_all.py --quick   # Quick mode (fewer runs)
"""

import sys
import time

def main():
    quick = '--quick' in sys.argv
    
    print("╔" + "═" * 68 + "╗")
    print("║" + " QEMA-G: Complete Validation Suite".center(68) + "║")
    print("║" + " Quantum-Enhanced Memory Architectures for Graph-Based AI".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    if quick:
        print("\n⚡ Quick mode: reduced runs/shots for faster execution\n")
    
    start = time.time()
    
    # ── Experiment 1 ──
    print("\n" + "▶" * 35)
    print("  EXPERIMENT 1: Identity VQC on Path Graph P4")
    print("▶" * 35 + "\n")
    from experiment1.exp1_identity_path import run_experiment as exp1
    exp1()
    
    # ── Experiment 2 ──
    print("\n" + "▶" * 35)
    print("  EXPERIMENT 2: Trained VQC on Path Graph P4")
    print("▶" * 35 + "\n")
    from experiment2.exp2_trained_path import run_experiment as exp2
    exp2()
    
    # ── Experiment 3 ──
    print("\n" + "▶" * 35)
    print("  EXPERIMENT 3: Identity VQC on Cycle Graph C4")
    print("▶" * 35 + "\n")
    from experiment3.exp3_identity_cycle import run_experiment as exp3
    exp3()
    
    # ── Experiment 4 ──
    print("\n" + "▶" * 35)
    print("  EXPERIMENT 4: 8-Node Classification")
    print("▶" * 35 + "\n")
    from experiment4.exp4_8node_classification import run_experiment as exp4
    exp4()
    
    elapsed = time.time() - start
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " ALL EXPERIMENTS COMPLETE".center(68) + "║")
    print("║" + f" Total time: {elapsed:.1f}s".center(68) + "║")
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    main()
