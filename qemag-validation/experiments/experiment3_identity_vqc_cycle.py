"""
Experiment 3: Identity VQC on Cycle Graph (C4)

Tests the quantum aggregation mechanism on a cycle graph where all nodes
have identical degree (d_i = 2) but different neighbor sets. Validates
that quantum aggregation distinguishes nodes by neighborhood content
rather than degree.

Reference: Section 7.4 of the paper.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix

from noise_models.brisbane_noise import get_brisbane_noise_model


# =============================================================================
# Feature vectors (same as Experiments 1-2)
# =============================================================================
FEATURES = {
    0: np.array([1, 1, 1, 1]) / 2.0,
    1: np.array([1, 0, 1, 0]) / np.sqrt(2),
    2: np.array([0, 1, 0, 1]) / np.sqrt(2),
    3: np.array([1, -1, 1, -1]) / 2.0,
}

# Cycle graph C4: edges {(0,1), (1,2), (2,3), (3,0)}
ADJACENCY = {
    0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [2, 0],
}


def compute_classical_aggregation(node: int) -> np.ndarray:
    """Compute classical mean aggregation for a node."""
    neighbors = ADJACENCY[node]
    return np.mean([FEATURES[j] for j in neighbors], axis=0)


def amplitude_encoding_angles(feature_vec: np.ndarray) -> list:
    """Compute Ry rotation angles for amplitude encoding."""
    a = feature_vec
    norm_01 = np.sqrt(a[0] ** 2 + a[1] ** 2)
    norm_23 = np.sqrt(a[2] ** 2 + a[3] ** 2)

    theta_top = 2 * np.arccos(np.clip(norm_01, -1, 1))
    theta_left = 2 * np.arccos(np.clip(a[0] / max(norm_01, 1e-12), -1, 1))
    theta_right = 2 * np.arccos(np.clip(a[2] / max(norm_23, 1e-12), -1, 1))

    return [theta_top, theta_left, theta_right]


def build_statevector_circuit(target_node: int) -> QuantumCircuit:
    """
    Build aggregation circuit for C4 (all nodes have degree 2).
    q0: address qubit (superposition of 2 neighbors)
    q1: unused
    q2, q3: feature qubits
    """
    neighbors = ADJACENCY[target_node]
    qc = QuantumCircuit(4)

    # All nodes in C4 have degree 2
    qc.h(0)

    angles_n0 = amplitude_encoding_angles(FEATURES[neighbors[0]])
    angles_n1 = amplitude_encoding_angles(FEATURES[neighbors[1]])

    # Controlled feature loading
    qc.x(0)
    qc.cry(angles_n0[0], 0, 2)
    qc.x(0)
    qc.cry(angles_n1[0], 0, 2)

    qc.x(0)
    qc.cry(angles_n0[1], 0, 3)
    qc.x(0)
    qc.cry(angles_n1[1], 0, 3)

    return qc


def build_measurement_circuit(target_node: int) -> QuantumCircuit:
    """Build circuit with measurement for noisy simulation."""
    neighbors = ADJACENCY[target_node]
    qc = QuantumCircuit(4, 2)

    qc.h(0)

    angles_n0 = amplitude_encoding_angles(FEATURES[neighbors[0]])
    angles_n1 = amplitude_encoding_angles(FEATURES[neighbors[1]])

    qc.x(0)
    qc.cry(angles_n0[0], 0, 2)
    qc.x(0)
    qc.cry(angles_n1[0], 0, 2)

    qc.x(0)
    qc.cry(angles_n0[1], 0, 3)
    qc.x(0)
    qc.cry(angles_n1[1], 0, 3)

    qc.measure([0, 1], [0, 1])

    return qc


def run_noiseless(target_node: int) -> dict:
    """Run noiseless statevector simulation."""
    qc = build_statevector_circuit(target_node)
    sv = Statevector.from_instruction(qc)
    dm = DensityMatrix(sv)
    dm_feat = partial_trace(dm, [0, 1])

    probs = np.real(np.diag(dm_feat.data))
    return {f"|{i:02b}>": float(probs[i]) for i in range(4)}


def run_noisy(target_node: int, num_runs: int = 10, shots: int = 10000) -> dict:
    """Run noisy simulation with Brisbane noise model."""
    noise_model = get_brisbane_noise_model()
    backend = AerSimulator(noise_model=noise_model)

    all_probs = {f"|{i:02b}>": [] for i in range(4)}

    for run in range(num_runs):
        qc = build_measurement_circuit(target_node)
        qc_t = transpile(qc, backend, optimization_level=1)
        result = backend.run(qc_t, shots=shots).result()
        counts = result.get_counts()

        total = sum(counts.values())
        for i in range(4):
            key = f"{i:02b}"
            all_probs[f"|{key}>"].append(counts.get(key, 0) / total)

    summary = {}
    for key, vals in all_probs.items():
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return summary


def compute_fidelity(noiseless_probs: dict, noisy_probs: dict) -> float:
    """Compute fidelity between noiseless and noisy probability distributions."""
    p_ideal = np.array([noiseless_probs[f"|{i:02b}>"] for i in range(4)])
    p_noisy = np.array([noisy_probs[f"|{i:02b}>"]["mean"] for i in range(4)])
    # Classical fidelity: F = (sum sqrt(p_i * q_i))^2
    return float(np.sum(np.sqrt(p_ideal * p_noisy)) ** 2)


def main():
    print("=" * 60)
    print("Experiment 3: Identity VQC on Cycle Graph (C4)")
    print("=" * 60)

    all_results = {}

    for target_node in range(4):
        print(f"\n--- Node {target_node} ---")
        print(f"Neighbors: {ADJACENCY[target_node]}")

        # Classical ground truth
        classical = compute_classical_aggregation(target_node)
        print(f"Classical aggregation: {classical}")

        # Analytical probabilities
        analytical = {f"|{i:02b}>": float(classical[i] ** 2) for i in range(4)}

        # Noiseless
        noiseless = run_noiseless(target_node)
        print(f"Noiseless probabilities:")
        for k, v in noiseless.items():
            print(f"  {k}: {v:.4f}")

        # Noisy
        noisy = run_noisy(target_node)
        print(f"Noisy probabilities:")
        for k, v in noisy.items():
            print(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f}")

        # Fidelity
        fidelity = compute_fidelity(noiseless, noisy)
        print(f"Fidelity: {fidelity:.3f}")

        all_results[f"node_{target_node}"] = {
            "neighbors": ADJACENCY[target_node],
            "classical_aggregation": classical.tolist(),
            "analytical": analytical,
            "noiseless": noiseless,
            "noisy": noisy,
            "fidelity": fidelity,
        }

    # Average fidelity
    fidelities = [all_results[f"node_{i}"]["fidelity"] for i in range(4)]
    avg_fidelity = np.mean(fidelities)
    std_fidelity = np.std(fidelities)
    print(f"\n{'='*60}")
    print(f"Average fidelity across all nodes: {avg_fidelity:.3f} ± {std_fidelity:.3f}")

    # Consistency check: node 1 should match Experiment 1
    print(f"\nConsistency check (node 1 vs Experiment 1):")
    print(f"  C4 node 1 noiseless: {all_results['node_1']['noiseless']}")
    print(f"  (Should match P4 node 1: |00>=0.0625, |01>=0.3650, |10>=0.0625, |11>=0.3650)")

    # Save
    output = {
        "experiment": "3_identity_vqc_cycle",
        "topology": "C4",
        "nodes": all_results,
        "summary": {
            "avg_fidelity": float(avg_fidelity),
            "std_fidelity": float(std_fidelity),
            "all_fidelities": [float(f) for f in fidelities],
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/experiment3_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results/experiment3_results.json")


if __name__ == "__main__":
    main()
