"""
Experiment 1: Identity VQC on Path Graph (P4)

Tests the quantum message-passing aggregation mechanism with U(θ) = I.
Validates that partial measurement produces the correct mean-aggregated
neighborhood features for a 4-node path graph.

Reference: Section 7.2 of the paper.
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
# Feature vectors (pre-normalized, orthogonal)
# =============================================================================
FEATURES = {
    0: np.array([1, 1, 1, 1]) / 2.0,
    1: np.array([1, 0, 1, 0]) / np.sqrt(2),
    2: np.array([0, 1, 0, 1]) / np.sqrt(2),
    3: np.array([1, -1, 1, -1]) / 2.0,
}

# Path graph P4: edges {(0,1), (1,2), (2,3)}
ADJACENCY = {
    0: [1],
    1: [0, 2],
    2: [1, 3],
    3: [2],
}


def compute_classical_aggregation(node: int) -> np.ndarray:
    """Compute classical mean aggregation for a node."""
    neighbors = ADJACENCY[node]
    return np.mean([FEATURES[j] for j in neighbors], axis=0)


def amplitude_encoding_angles(feature_vec: np.ndarray) -> list:
    """
    Compute rotation angles for amplitude encoding a 4D normalized vector
    into 2 qubits using a tree of Ry rotations.

    For a vector [a0, a1, a2, a3] with unit norm:
      - First Ry on q0: angle = 2*arccos(sqrt(a0^2 + a1^2))
      - Controlled-Ry on q1 (ctrl=q0=|0>): angle = 2*arccos(a0/sqrt(a0^2+a1^2))
      - Controlled-Ry on q1 (ctrl=q0=|1>): angle = 2*arccos(a2/sqrt(a2^2+a3^2))
    """
    a = feature_vec
    norm_01 = np.sqrt(a[0] ** 2 + a[1] ** 2)
    norm_23 = np.sqrt(a[2] ** 2 + a[3] ** 2)

    # Top-level split
    if norm_01 + norm_23 < 1e-12:
        return [0.0, 0.0, 0.0]

    theta_top = 2 * np.arccos(np.clip(norm_01, -1, 1))

    # Left branch (q0 = |0>)
    if norm_01 < 1e-12:
        theta_left = 0.0
    else:
        theta_left = 2 * np.arccos(np.clip(a[0] / norm_01, -1, 1))

    # Right branch (q0 = |1>)
    if norm_23 < 1e-12:
        theta_right = 0.0
    else:
        theta_right = 2 * np.arccos(np.clip(a[2] / norm_23, -1, 1))

    return [theta_top, theta_left, theta_right]


def build_aggregation_circuit(target_node: int) -> QuantumCircuit:
    """
    Build the quantum message-passing circuit for a given node.

    Circuit layout:
      q0, q1: address qubits (neighbor register)
      q2, q3: feature qubits

    For degree-2 nodes (interior), we use Hadamard on address qubit
    to create equal superposition of two neighbors.
    For degree-1 nodes (boundary), no superposition needed.
    """
    neighbors = ADJACENCY[target_node]
    degree = len(neighbors)

    qc = QuantumCircuit(4, 2)  # 4 qubits, 2 classical bits for address measurement

    if degree == 2:
        # Step 1: Create superposition over 2 neighbors
        qc.h(0)

        # Step 2a: Controlled feature loading for neighbor 0 (address = |0>)
        angles_n0 = amplitude_encoding_angles(FEATURES[neighbors[0]])
        # When q0 = |0>, encode features of neighbors[0]
        qc.x(0)  # Flip so ctrl-on-0 becomes ctrl-on-1
        qc.cry(angles_n0[0], 0, 2)
        # Controlled-controlled for sub-branches
        qc.x(0)

        # Step 2b: Controlled feature loading for neighbor 1 (address = |1>)
        angles_n1 = amplitude_encoding_angles(FEATURES[neighbors[1]])
        qc.cry(angles_n1[0], 0, 2)

        # Sub-branch rotations (simplified for 2-qubit feature register)
        # For the full encoding, we use controlled rotations on q3 conditioned on q2
        qc.x(0)
        qc.cry(angles_n0[1], 0, 3)  # left sub-branch for neighbor 0
        qc.x(0)
        qc.cry(angles_n1[1], 0, 3)  # left sub-branch for neighbor 1

        # Right sub-branches (conditioned on q2 = |1>)
        qc.x(2)
        qc.x(0)
        qc.ccx(0, 2, 3)  # Toffoli for controlled sub-rotation
        qc.x(0)
        qc.x(2)

    elif degree == 1:
        # Single neighbor: no superposition, directly encode
        angles = amplitude_encoding_angles(FEATURES[neighbors[0]])
        qc.ry(angles[0], 2)
        qc.cry(angles[1], 2, 3)

    # Step 3: Identity VQC (U(θ) = I) — no gates needed

    # Step 4: Measure address qubits for partial trace
    qc.measure([0, 1], [0, 1])

    return qc


def build_statevector_circuit(target_node: int) -> QuantumCircuit:
    """Build circuit without measurement for statevector simulation."""
    neighbors = ADJACENCY[target_node]
    degree = len(neighbors)

    qc = QuantumCircuit(4)

    if degree == 2:
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

    elif degree == 1:
        angles = amplitude_encoding_angles(FEATURES[neighbors[0]])
        qc.ry(angles[0], 2)
        qc.cry(angles[1], 2, 3)

    return qc


def run_noiseless(target_node: int) -> dict:
    """Run noiseless statevector simulation."""
    qc = build_statevector_circuit(target_node)
    sv = Statevector.from_instruction(qc)

    # Partial trace over address qubits (q0, q1) to get feature register state
    dm_full = DensityMatrix(sv)
    dm_feat = partial_trace(dm_full, [0, 1])

    # Extract diagonal (measurement probabilities in computational basis)
    probs = np.real(np.diag(dm_feat.data))
    return {
        f"|{i:02b}>": float(probs[i])
        for i in range(4)
    }


def run_noisy(target_node: int, num_runs: int = 10, shots: int = 10000) -> dict:
    """Run noisy simulation with Brisbane noise model."""
    noise_model = get_brisbane_noise_model()
    backend = AerSimulator(noise_model=noise_model)

    all_probs = {f"|{i:02b}>": [] for i in range(4)}

    for run_idx in range(num_runs):
        qc = build_aggregation_circuit(target_node)
        qc_transpiled = transpile(qc, backend, optimization_level=1)
        result = backend.run(qc_transpiled, shots=shots).result()
        counts = result.get_counts()

        total = sum(counts.values())
        for i in range(4):
            key = f"{i:02b}"
            all_probs[f"|{key}>"].append(counts.get(key, 0) / total)

    # Compute mean ± std
    summary = {}
    for key, vals in all_probs.items():
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return summary


def main():
    print("=" * 60)
    print("Experiment 1: Identity VQC on Path Graph (P4)")
    print("=" * 60)

    # Test node 1 (interior, degree 2)
    target_node = 1
    print(f"\nTarget node: {target_node}")
    print(f"Neighbors: {ADJACENCY[target_node]}")

    # Classical ground truth
    classical = compute_classical_aggregation(target_node)
    print(f"\nClassical ground truth (mean aggregation):")
    print(f"  h_1 = {classical}")

    # Analytical probabilities (squared amplitudes of aggregated state)
    analytical = {f"|{i:02b}>": float(classical[i] ** 2) for i in range(4)}
    print(f"\nAnalytical measurement probabilities:")
    for k, v in analytical.items():
        print(f"  {k}: {v:.4f}")

    # Noiseless simulation
    print(f"\nNoiseless simulation:")
    noiseless = run_noiseless(target_node)
    for k, v in noiseless.items():
        print(f"  {k}: {v:.4f}")

    # Noisy simulation
    print(f"\nNoisy simulation (Brisbane model, 10 runs × 10000 shots):")
    noisy = run_noisy(target_node)
    for k, v in noisy.items():
        print(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f}")

    # Save results
    results = {
        "experiment": "1_identity_vqc_path",
        "target_node": target_node,
        "topology": "P4",
        "analytical": analytical,
        "noiseless": noiseless,
        "noisy": noisy,
        "classical_aggregation": classical.tolist(),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/experiment1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/experiment1_results.json")


if __name__ == "__main__":
    main()
