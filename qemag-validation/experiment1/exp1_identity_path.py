"""
Experiment 1: Identity VQC on Path Graph P4
=============================================
Section 7.2 / Table 4, Row 1

Tests the quantum aggregation mechanism with U(θ) = I.
Validates that measurement probabilities match analytical ground truth
for node 1 (interior, degree 2, neighbors {0, 2}).

Expected output (Eq. 12):
  h_1 = (1/2)(x_0 + x_2) = [0.25, 0.604, 0.25, 0.604]^T
  P(|00>) = 0.0625, P(|01>) = 0.3650, P(|10>) = 0.0625, P(|11>) = 0.3650
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from utils.qemag_circuits import (
    FEATURES_P4, ADJ_PATH_P4, 
    compute_classical_aggregation, analytical_probabilities,
    build_aggregation_circuit_simple
)
from noise_models.brisbane_config import get_brisbane_noise_model

NUM_SHOTS = 10000
NUM_RUNS = 10
TARGET_NODE = 1


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 1: Identity VQC on Path Graph P4")
    print("=" * 70)
    
    # ── Classical Ground Truth ──
    h1 = compute_classical_aggregation(TARGET_NODE, ADJ_PATH_P4, FEATURES_P4)
    probs_analytical = analytical_probabilities(h1)
    
    print(f"\nNode {TARGET_NODE} neighbors: {ADJ_PATH_P4[TARGET_NODE]}")
    print(f"Classical aggregation h_1 = {h1}")
    print(f"Analytical probabilities: {probs_analytical}")
    print(f"  P(|00>) = {probs_analytical[0]:.4f}")
    print(f"  P(|01>) = {probs_analytical[1]:.4f}")
    print(f"  P(|10>) = {probs_analytical[2]:.4f}")
    print(f"  P(|11>) = {probs_analytical[3]:.4f}")
    
    # ── Build Circuit ──
    qc = build_aggregation_circuit_simple(
        TARGET_NODE, ADJ_PATH_P4, FEATURES_P4, vqc_params=None
    )
    
    # ── Noiseless Simulation ──
    print("\n--- Noiseless Simulation ---")
    sim = AerSimulator(method='statevector')
    qc_transpiled = transpile(qc, sim)
    result = sim.run(qc_transpiled, shots=NUM_SHOTS).result()
    counts = result.get_counts()
    
    total = sum(counts.values())
    basis_states = ['00', '01', '10', '11']
    probs_noiseless = {bs: counts.get(bs, 0) / total for bs in basis_states}
    
    print("Measured probabilities:")
    for bs in basis_states:
        print(f"  P(|{bs}>) = {probs_noiseless[bs]:.4f}  "
              f"(analytical: {probs_analytical[int(bs, 2)]:.4f})")
    
    # ── Noisy Simulation (10 runs) ──
    print(f"\n--- Noisy Simulation ({NUM_RUNS} runs, {NUM_SHOTS} shots each) ---")
    noise_model = get_brisbane_noise_model()
    sim_noisy = AerSimulator(noise_model=noise_model)
    
    all_probs = {bs: [] for bs in basis_states}
    
    for run in range(NUM_RUNS):
        qc_t = transpile(qc, sim_noisy)
        result = sim_noisy.run(qc_t, shots=NUM_SHOTS).result()
        counts = result.get_counts()
        total = sum(counts.values())
        for bs in basis_states:
            all_probs[bs].append(counts.get(bs, 0) / total)
    
    print("Noisy probabilities (mean ± std):")
    for bs in basis_states:
        arr = np.array(all_probs[bs])
        print(f"  P(|{bs}>) = {arr.mean():.3f} ± {arr.std():.3f}  "
              f"(analytical: {probs_analytical[int(bs, 2)]:.4f})")
    
    # ── Fidelity ──
    mean_probs = np.array([np.mean(all_probs[bs]) for bs in basis_states])
    fidelity = np.sum(np.sqrt(probs_analytical * mean_probs))**2
    print(f"\nState fidelity F = {fidelity:.3f}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 70)
    
    return probs_analytical, probs_noiseless, all_probs, fidelity


if __name__ == "__main__":
    run_experiment()
