"""
Experiment 3: Identity VQC on Cycle Graph C4
=============================================
Section 7.4 / Table 4, Row 2

Tests aggregation on uniform-degree topology (all nodes degree 2).
Validates that quantum aggregation distinguishes nodes based on
neighborhood content rather than degree.

Expected: F = 0.93 ± 0.01 under Brisbane noise.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from utils.qemag_circuits import (
    FEATURES_P4, ADJ_CYCLE_C4,
    compute_classical_aggregation, analytical_probabilities,
    build_aggregation_circuit_simple
)
from noise_models.brisbane_config import get_brisbane_noise_model

NUM_SHOTS = 10000
NUM_RUNS = 10


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 3: Identity VQC on Cycle Graph C4")
    print("=" * 70)
    
    basis_states = ['00', '01', '10', '11']
    sim = AerSimulator(method='statevector')
    noise_model = get_brisbane_noise_model()
    sim_noisy = AerSimulator(noise_model=noise_model)
    
    for node_id in range(4):
        h = compute_classical_aggregation(node_id, ADJ_CYCLE_C4, FEATURES_P4)
        probs_analytical = analytical_probabilities(h)
        
        print(f"\nNode {node_id} | neighbors: {ADJ_CYCLE_C4[node_id]}")
        print(f"  h = {h}")
        
        qc = build_aggregation_circuit_simple(
            node_id, ADJ_CYCLE_C4, FEATURES_P4, vqc_params=None
        )
        
        # Noiseless
        qc_t = transpile(qc, sim)
        result = sim.run(qc_t, shots=NUM_SHOTS).result()
        counts = result.get_counts()
        total = sum(counts.values())
        probs_noiseless = [counts.get(bs, 0) / total for bs in basis_states]
        
        # Noisy (10 runs)
        all_probs = {bs: [] for bs in basis_states}
        for run in range(NUM_RUNS):
            qc_t = transpile(qc, sim_noisy)
            result = sim_noisy.run(qc_t, shots=NUM_SHOTS).result()
            counts = result.get_counts()
            total = sum(counts.values())
            for bs in basis_states:
                all_probs[bs].append(counts.get(bs, 0) / total)
        
        print("  Probabilities:")
        for i, bs in enumerate(basis_states):
            arr = np.array(all_probs[bs])
            print(f"    |{bs}>: analytical={probs_analytical[i]:.4f}  "
                  f"noiseless={probs_noiseless[i]:.4f}  "
                  f"noisy={arr.mean():.3f}±{arr.std():.3f}")
        
        # Fidelity
        mean_probs = np.array([np.mean(all_probs[bs]) for bs in basis_states])
        fidelity = np.sum(np.sqrt(probs_analytical * np.clip(mean_probs, 0, None)))**2
        print(f"  Fidelity F = {fidelity:.3f}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
