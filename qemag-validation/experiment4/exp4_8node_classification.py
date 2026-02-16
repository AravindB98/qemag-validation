"""
Experiment 4: 8-Node Noiseless VQC Classification
==================================================
Section 7.5

Multi-class node classification on G8 (3 structural classes).
Tests L=2 (10 params) and L=1 (5 params) VQC configurations.
Includes classical baseline comparison and initialization sensitivity.

Results (from paper):
  L=2: 100% accuracy within 60 iterations
  L=1: 87.5% accuracy (7/8 correct)
  Classical 8-param baseline: 75.0%
  Classical 5-param PCA baseline: 62.5%
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from utils.qemag_circuits import amplitude_encode_angles, _add_hardware_efficient_vqc

NUM_SHOTS = 1000
NUM_ITERATIONS = 100
NUM_RUNS = 10
LEARNING_RATE = 0.01

# Load graph spec
with open(os.path.join(os.path.dirname(__file__), 'graph_spec.json')) as f:
    G8 = json.load(f)

ADJ = {int(k): v for k, v in G8['adjacency_list'].items()}
FEATURES = {int(k): np.array(v) for k, v in G8['feature_vectors'].items()}
LABELS = {int(k): v for k, v in G8['class_labels'].items()}
NUM_CLASSES = 3


def classical_mean_aggregate(node_id):
    """Compute classical mean-aggregated feature for a node."""
    neighbors = ADJ[node_id]
    return np.mean([FEATURES[j] for j in neighbors], axis=0)


def build_classification_circuit_8node(node_id, params, n_params):
    """Build aggregation + VQC for 8-node classification."""
    neighbors = ADJ[node_id]
    agg_feat = classical_mean_aggregate(node_id)
    # Normalize for amplitude encoding
    norm = np.linalg.norm(agg_feat)
    if norm > 1e-10:
        agg_feat_norm = agg_feat / norm
    else:
        agg_feat_norm = agg_feat
    
    qc = QuantumCircuit(2, 2)  # 2 feature qubits only (classical pre-aggregation)
    
    # Amplitude encode the aggregated feature
    angles = amplitude_encode_angles(agg_feat_norm)
    qc.ry(angles[0], 0)
    qc.ry(angles[1], 1)
    
    qc.barrier()
    
    # VQC
    _add_hardware_efficient_vqc(qc, params, feature_qubits=[0, 1])
    
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    
    return qc


def evaluate_8node(params, n_params, simulator, shots=NUM_SHOTS):
    """Evaluate 3-class accuracy on all 8 nodes."""
    correct = 0
    for node in range(8):
        qc = build_classification_circuit_8node(node, params, n_params)
        qc_t = transpile(qc, simulator)
        result = simulator.run(qc_t, shots=shots).result()
        counts = result.get_counts()
        
        # 3-class prediction: |00>=hub, |01>=bridge, |10>=peripheral
        p = {}
        total = sum(counts.values())
        for bs in ['00', '01', '10', '11']:
            p[bs] = counts.get(bs, 0) / total
        
        # Map to class
        class_probs = [p['00'], p['01'], p['10'] + p['11']]
        predicted = np.argmax(class_probs)
        
        if predicted == LABELS[node]:
            correct += 1
    
    return correct / 8


def compute_loss_8node(params, n_params, simulator, shots=NUM_SHOTS):
    """Cross-entropy loss for 3-class classification."""
    loss = 0.0
    for node in range(8):
        qc = build_classification_circuit_8node(node, params, n_params)
        qc_t = transpile(qc, simulator)
        result = simulator.run(qc_t, shots=shots).result()
        counts = result.get_counts()
        total = sum(counts.values())
        
        p = {}
        for bs in ['00', '01', '10', '11']:
            p[bs] = counts.get(bs, 0) / total
        
        class_probs = [
            np.clip(p['00'], 1e-7, 1),
            np.clip(p['01'], 1e-7, 1),
            np.clip(p['10'] + p['11'], 1e-7, 1)
        ]
        
        label = LABELS[node]
        loss -= np.log(class_probs[label])
    
    return loss / 8


def parameter_shift_gradient_8node(params, n_params, simulator, shots=NUM_SHOTS):
    """Parameter-shift gradient for 8-node classification."""
    grad = np.zeros(n_params)
    for k in range(n_params):
        p_plus = params.copy()
        p_plus[k] += np.pi / 2
        p_minus = params.copy()
        p_minus[k] -= np.pi / 2
        
        l_plus = compute_loss_8node(p_plus, n_params, simulator, shots)
        l_minus = compute_loss_8node(p_minus, n_params, simulator, shots)
        
        grad[k] = (l_plus - l_minus) / 2.0
    return grad


def classical_baseline():
    """Run classical baselines for comparison."""
    print("\n--- Classical Baselines ---")
    
    # Compute aggregated features for all nodes
    agg_features = np.array([classical_mean_aggregate(i) for i in range(8)])
    labels = np.array([LABELS[i] for i in range(8)])
    
    # Baseline (a): 8-parameter rank-1 factored model
    best_acc_8 = 0
    for trial in range(100):
        np.random.seed(trial)
        u = np.random.randn(4)
        v = np.random.randn(3)
        bias = np.random.randn(1)[0]
        
        # Simple optimization
        lr = 0.01
        for _ in range(200):
            logits = agg_features @ u.reshape(-1, 1) @ v.reshape(1, -1) + bias
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            
            # Gradient step (simplified)
            for i in range(8):
                target = np.zeros(3)
                target[labels[i]] = 1
                err = probs[i] - target
                u -= lr * agg_features[i] * (err @ v) * 0.1
                v -= lr * (agg_features[i] @ u) * err * 0.1
                bias -= lr * err.sum() * 0.01
        
        # Evaluate
        logits = agg_features @ u.reshape(-1, 1) @ v.reshape(1, -1) + bias
        preds = logits.argmax(axis=1)
        acc = (preds == labels).mean()
        best_acc_8 = max(best_acc_8, acc)
    
    print(f"  8-parameter rank-1 baseline: {best_acc_8:.1%}")
    
    # Baseline (b): 5-parameter PCA model
    from numpy.linalg import svd
    U, S, Vt = svd(agg_features - agg_features.mean(axis=0), full_matrices=False)
    X_pca = U[:, :2] * S[:2]  # 2D PCA projection
    
    best_acc_5 = 0
    for trial in range(100):
        np.random.seed(trial + 1000)
        W = np.random.randn(2, 3) * 0.1
        b = np.zeros(3)
        
        for _ in range(500):
            logits = X_pca @ W + b
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            
            for i in range(8):
                target = np.zeros(3)
                target[labels[i]] = 1
                err = probs[i] - target
                W -= 0.01 * X_pca[i].reshape(-1, 1) @ err.reshape(1, -1)
                b -= 0.01 * err
        
        logits = X_pca @ W + b
        preds = logits.argmax(axis=1)
        acc = (preds == labels).mean()
        best_acc_5 = max(best_acc_5, acc)
    
    print(f"  5-parameter PCA baseline: {best_acc_5:.1%}")
    
    return best_acc_8, best_acc_5


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 4: 8-Node Noiseless VQC Classification")
    print("=" * 70)
    
    sim = AerSimulator(method='statevector')
    
    for n_params, label in [(10, "L=2 (10 params)"), (5, "L=1 (5 params)")]:
        print(f"\n{'='*50}")
        print(f"VQC Configuration: {label}")
        print(f"{'='*50}")
        
        all_accs = []
        all_convergence = []
        
        for run in range(NUM_RUNS):
            np.random.seed(42 + run)
            params = np.random.uniform(0, 2 * np.pi, n_params)
            
            m = np.zeros(n_params)
            v = np.zeros(n_params)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            converged_at = NUM_ITERATIONS
            
            for iteration in range(NUM_ITERATIONS):
                grad = parameter_shift_gradient_8node(params, n_params, sim)
                
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**(iteration + 1))
                v_hat = v / (1 - beta2**(iteration + 1))
                params -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)
                
                if (iteration + 1) % 20 == 0:
                    acc = evaluate_8node(params, n_params, sim)
                    print(f"  Run {run+1}, Iter {iteration+1}: acc={acc:.1%}")
                    
                    if acc == 1.0 and converged_at == NUM_ITERATIONS:
                        converged_at = iteration + 1
            
            final_acc = evaluate_8node(params, n_params, sim)
            all_accs.append(final_acc)
            all_convergence.append(converged_at)
        
        acc_arr = np.array(all_accs)
        conv_arr = np.array(all_convergence)
        print(f"\n  Summary ({label}):")
        print(f"    Accuracy: {acc_arr.mean():.1%} "
              f"({np.sum(acc_arr >= 0.875)}/{NUM_RUNS} runs >= 87.5%)")
        print(f"    Convergence: {conv_arr.mean():.0f} ± {conv_arr.std():.0f} iterations")
    
    # Classical baselines
    classical_baseline()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
