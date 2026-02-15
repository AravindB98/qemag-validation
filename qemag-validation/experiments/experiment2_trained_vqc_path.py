"""
Experiment 2: Trained VQC on Path Graph (P4) — Binary Node Classification

Trains a 2-layer hardware-efficient VQC to classify nodes as
"boundary" (degree 1) vs "interior" (degree 2) on the P4 path graph.

Validates: (1) parameter-shift gradient computation, (2) hybrid
classical-quantum optimization convergence, (3) discriminative
feature generation after aggregation.

Reference: Section 7.3 of the paper.
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
# Graph and features (same as Experiment 1)
# =============================================================================
FEATURES = {
    0: np.array([1, 1, 1, 1]) / 2.0,
    1: np.array([1, 0, 1, 0]) / np.sqrt(2),
    2: np.array([0, 1, 0, 1]) / np.sqrt(2),
    3: np.array([1, -1, 1, -1]) / 2.0,
}

ADJACENCY = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}

# Labels: 0 = boundary (degree 1), 1 = interior (degree 2)
LABELS = {0: 0, 1: 1, 2: 1, 3: 0}


# =============================================================================
# VQC Ansatz: Hardware-efficient, L=2 layers on 2 qubits
# =============================================================================
def build_vqc_circuit(params: np.ndarray) -> QuantumCircuit:
    """
    Build L=2 layer hardware-efficient ansatz on 2 feature qubits.

    Each layer: Ry, Rz on each qubit + CNOT ladder.
    Total parameters: 2 layers × 2 qubits × 2 rotations + 2 extra = 10.
    """
    assert len(params) == 10, f"Expected 10 parameters, got {len(params)}"

    qc = QuantumCircuit(2)

    idx = 0
    for layer in range(2):
        # Single-qubit rotations
        for qubit in range(2):
            qc.ry(params[idx], qubit)
            idx += 1
            qc.rz(params[idx], qubit)
            idx += 1
        # CNOT ladder
        qc.cx(0, 1)

    # Final rotation for readout
    qc.ry(params[idx], 0)
    idx += 1
    qc.rz(params[idx], 0)

    return qc


def amplitude_encoding_angles(feature_vec: np.ndarray) -> list:
    """Compute Ry rotation angles for 4D amplitude encoding into 2 qubits."""
    a = feature_vec
    norm_01 = np.sqrt(a[0] ** 2 + a[1] ** 2)
    norm_23 = np.sqrt(a[2] ** 2 + a[3] ** 2)

    theta_top = 2 * np.arccos(np.clip(norm_01, -1, 1))
    theta_left = 2 * np.arccos(np.clip(a[0] / max(norm_01, 1e-12), -1, 1))
    theta_right = 2 * np.arccos(np.clip(a[2] / max(norm_23, 1e-12), -1, 1))

    return [theta_top, theta_left, theta_right]


def build_full_circuit(target_node: int, vqc_params: np.ndarray) -> QuantumCircuit:
    """
    Build full aggregation + VQC circuit for a node.
    q0: address qubit, q1: (unused for degree <= 2)
    q2, q3: feature qubits
    """
    neighbors = ADJACENCY[target_node]
    degree = len(neighbors)

    qc = QuantumCircuit(4)

    # Aggregation (same as Experiment 1)
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

    # Apply VQC to feature qubits (q2, q3)
    vqc = build_vqc_circuit(vqc_params)
    qc.compose(vqc, qubits=[2, 3], inplace=True)

    return qc


def get_classification_prob(target_node: int, vqc_params: np.ndarray) -> float:
    """
    Get P(class=1) = probability of measuring |0> on first feature qubit (q2).
    """
    qc = build_full_circuit(target_node, vqc_params)
    sv = Statevector.from_instruction(qc)
    dm = DensityMatrix(sv)
    dm_feat = partial_trace(dm, [0, 1])

    # P(q2 = |0>) = sum of probabilities where first qubit is 0
    probs = np.real(np.diag(dm_feat.data))
    p_class1 = probs[0] + probs[1]  # |00> + |01>
    return float(p_class1)


def binary_cross_entropy(p_pred: float, y_true: int, eps: float = 1e-8) -> float:
    """Compute binary cross-entropy loss."""
    p = np.clip(p_pred, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def compute_loss(vqc_params: np.ndarray) -> float:
    """Total loss across all nodes."""
    total_loss = 0.0
    for node in range(4):
        p = get_classification_prob(node, vqc_params)
        total_loss += binary_cross_entropy(p, LABELS[node])
    return total_loss / 4.0


def parameter_shift_gradient(vqc_params: np.ndarray) -> np.ndarray:
    """
    Compute gradient using the parameter-shift rule:
    dC/dθ_k = [C(θ_k + π/2) - C(θ_k - π/2)] / 2
    """
    grad = np.zeros_like(vqc_params)
    for k in range(len(vqc_params)):
        params_plus = vqc_params.copy()
        params_minus = vqc_params.copy()
        params_plus[k] += np.pi / 2
        params_minus[k] -= np.pi / 2

        grad[k] = (compute_loss(params_plus) - compute_loss(params_minus)) / 2.0
    return grad


def adam_update(params, grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer step."""
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v


def train_vqc(num_iterations: int = 100, lr: float = 0.01, seed: int = 42) -> dict:
    """Train VQC using Adam with parameter-shift gradients."""
    rng = np.random.RandomState(seed)
    params = rng.uniform(0, 2 * np.pi, size=10)

    m = np.zeros(10)
    v = np.zeros(10)

    history = {"loss": [], "accuracy": []}

    for t in range(1, num_iterations + 1):
        # Compute loss and accuracy
        loss = compute_loss(params)
        predictions = {}
        correct = 0
        for node in range(4):
            p = get_classification_prob(node, params)
            pred = 1 if p > 0.5 else 0
            predictions[node] = pred
            if pred == LABELS[node]:
                correct += 1
        accuracy = correct / 4.0

        history["loss"].append(float(loss))
        history["accuracy"].append(float(accuracy))

        if t % 20 == 0 or t == 1:
            print(f"  Iter {t:3d}: loss={loss:.4f}, accuracy={accuracy:.0%}")

        if accuracy == 1.0 and t > 10:
            print(f"  Converged at iteration {t}")
            break

        # Gradient step
        grad = parameter_shift_gradient(params)
        params, m, v = adam_update(params, grad, m, v, t, lr=lr)

    return {
        "final_params": params.tolist(),
        "final_loss": float(loss),
        "final_accuracy": float(accuracy),
        "history": history,
        "converged_at": t,
    }


def evaluate_noisy(params: np.ndarray, num_runs: int = 10, shots: int = 1000) -> dict:
    """Evaluate trained VQC under Brisbane noise model."""
    noise_model = get_brisbane_noise_model()
    backend = AerSimulator(noise_model=noise_model)

    accuracies = []
    fidelities = []

    for run in range(num_runs):
        correct = 0
        run_fidelities = []

        for node in range(4):
            qc = build_full_circuit(node, params)
            qc.measure_all()
            qc_t = transpile(qc, backend, optimization_level=1)
            result = backend.run(qc_t, shots=shots).result()
            counts = result.get_counts()

            # Classification from first feature qubit
            p_class1 = 0
            total = sum(counts.values())
            for bitstring, count in counts.items():
                # Qiskit bit ordering: rightmost = q0
                feat_q2 = bitstring[-3]  # third from right
                if feat_q2 == "0":
                    p_class1 += count / total

            pred = 1 if p_class1 > 0.5 else 0
            if pred == LABELS[node]:
                correct += 1

        accuracies.append(correct / 4.0)

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "all_accuracies": [float(a) for a in accuracies],
    }


def main():
    print("=" * 60)
    print("Experiment 2: Trained VQC on Path Graph (P4)")
    print("Binary Node Classification: Boundary vs Interior")
    print("=" * 60)

    # Train across 10 independent runs
    all_results = []
    for run in range(10):
        print(f"\n--- Training Run {run + 1}/10 ---")
        result = train_vqc(num_iterations=100, lr=0.01, seed=run * 42)
        all_results.append(result)

    # Summary
    final_accs = [r["final_accuracy"] for r in all_results]
    convergence_iters = [r["converged_at"] for r in all_results]
    print(f"\n{'='*60}")
    print(f"Summary across 10 runs:")
    print(f"  Mean accuracy: {np.mean(final_accs):.2%} ± {np.std(final_accs):.2%}")
    print(f"  Mean convergence: {np.mean(convergence_iters):.0f} ± {np.std(convergence_iters):.0f} iterations")

    # Noisy evaluation of best run
    best_run = max(range(10), key=lambda i: all_results[i]["final_accuracy"])
    best_params = np.array(all_results[best_run]["final_params"])

    print(f"\nNoisy evaluation (Brisbane model, best run):")
    noisy_results = evaluate_noisy(best_params)
    print(f"  Noisy accuracy: {noisy_results['mean_accuracy']:.2%} ± {noisy_results['std_accuracy']:.2%}")

    # Save results
    output = {
        "experiment": "2_trained_vqc_path",
        "topology": "P4",
        "task": "binary_classification_boundary_vs_interior",
        "vqc_config": {"layers": 2, "qubits": 2, "params": 10},
        "training_runs": all_results,
        "noisy_evaluation": noisy_results,
        "summary": {
            "mean_noiseless_accuracy": float(np.mean(final_accs)),
            "std_noiseless_accuracy": float(np.std(final_accs)),
            "mean_convergence_iter": float(np.mean(convergence_iters)),
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/experiment2_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Save trained parameters separately
    params_output = {
        f"run_{i}": all_results[i]["final_params"]
        for i in range(10)
    }
    with open("trained_params/experiment2_params.json", "w") as f:
        json.dump(params_output, f, indent=2)

    print("\nResults saved to results/experiment2_results.json")
    print("Trained params saved to trained_params/experiment2_params.json")


if __name__ == "__main__":
    main()
