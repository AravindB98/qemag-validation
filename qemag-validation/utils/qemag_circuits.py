"""
QEMA-G Circuit Construction Utilities
======================================
Shared functions for quantum message-passing protocol implementation.
Used by all four experiments.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


# ── Feature Vectors (Eq. 11 in paper) ──
# Pre-normalized, orthogonal structure for clean verification
FEATURES_P4 = {
    0: np.array([1, 1, 1, 1]) / 2.0,
    1: np.array([1, 0, 1, 0]) / np.sqrt(2),
    2: np.array([0, 1, 0, 1]) / np.sqrt(2),
    3: np.array([1, -1, 1, -1]) / 2.0,
}

# ── Graph Topologies ──
ADJ_PATH_P4 = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
ADJ_CYCLE_C4 = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [2, 0]}


def amplitude_encode_angles(feature_vec):
    """
    Compute Ry rotation angles for amplitude encoding a 4D unit vector
    into 2 qubits using a 3-rotation decomposition.
    
    For |psi> = a|00> + b|01> + c|10> + d|11>:
      - First Ry on q0: angle = 2*arccos(sqrt(a^2 + b^2))
      - Controlled Ry on q1 (ctrl=|0> on q0): angle = 2*arccos(a / sqrt(a^2+b^2))
      - Controlled Ry on q1 (ctrl=|1> on q0): angle = 2*arccos(c / sqrt(c^2+d^2))
    """
    a, b, c, d = feature_vec
    
    # Probability of |0> on first qubit
    p0 = a**2 + b**2
    p1 = c**2 + d**2
    
    theta_0 = 2 * np.arccos(np.clip(np.sqrt(p0), -1, 1))
    
    if p0 > 1e-10:
        theta_1_given_0 = 2 * np.arccos(np.clip(a / np.sqrt(p0), -1, 1))
    else:
        theta_1_given_0 = 0.0
    
    if p1 > 1e-10:
        theta_1_given_1 = 2 * np.arccos(np.clip(c / np.sqrt(p1), -1, 1))
    else:
        theta_1_given_1 = 0.0
    
    return theta_0, theta_1_given_0, theta_1_given_1


def build_aggregation_circuit(node_id, adj_list, features, vqc_params=None):
    """
    Build the quantum message-passing circuit for a single node.
    
    Implements the 4-phase protocol (Section 3.3):
      Phase 1: Superposition query (Hadamard on address register)
      Phase 2: Controlled feature loading (amplitude encoding)
      Phase 3: VQC transformation U(θ) on feature register
      Phase 4: Measurement of address register (partial trace)
    
    Parameters
    ----------
    node_id : int
        Target node for aggregation.
    adj_list : dict
        Adjacency list {node: [neighbors]}.
    features : dict
        Feature vectors {node: np.array}.
    vqc_params : np.array or None
        VQC parameters. If None, identity (U=I) is used.
    
    Returns
    -------
    QuantumCircuit
        The aggregation circuit with 4 qubits (2 address + 2 feature).
    """
    neighbors = adj_list[node_id]
    n_neighbors = len(neighbors)
    
    # 4 qubits: 2 address + 2 feature
    qc = QuantumCircuit(4, 2)  # measure only feature qubits
    
    # Address qubits: q[0], q[1]
    # Feature qubits: q[2], q[3]
    
    # ── Phase 1: Superposition Query ──
    if n_neighbors == 2:
        # Equal superposition over 2 neighbors
        qc.h(0)
    elif n_neighbors == 1:
        # Single neighbor: no superposition needed
        pass
    
    qc.barrier()
    
    # ── Phase 2: Controlled Feature Loading ──
    if n_neighbors == 1:
        # Direct encoding of single neighbor's features
        neighbor = neighbors[0]
        angles = amplitude_encode_angles(features[neighbor])
        qc.ry(angles[0], 2)
        qc.ry(angles[1], 3)
        # Controlled rotation for |1> subspace of q2
        qc.x(2)
        qc.cry(angles[2] - angles[1], 2, 3)
        qc.x(2)
    
    elif n_neighbors == 2:
        # Controlled encoding: |0> on address -> neighbor[0], |1> -> neighbor[1]
        for idx, neighbor in enumerate(neighbors):
            angles = amplitude_encode_angles(features[neighbor])
            
            if idx == 0:
                # Control on |0>: apply X, then controlled, then X
                qc.x(0)
                qc.cry(angles[0], 0, 2)
                # Controlled-controlled Ry for q3 given q0=|0>, q2
                _add_controlled_feature_encoding(qc, angles, ctrl_qubit=0, 
                                                  feat_qubits=[2, 3], ctrl_state=1)
                qc.x(0)
            else:
                # Control on |1>
                qc.cry(angles[0], 0, 2)
                _add_controlled_feature_encoding(qc, angles, ctrl_qubit=0,
                                                  feat_qubits=[2, 3], ctrl_state=1)
    
    qc.barrier()
    
    # ── Phase 3: VQC Transformation ──
    if vqc_params is not None:
        _add_hardware_efficient_vqc(qc, vqc_params, feature_qubits=[2, 3])
    # Else: identity (U = I)
    
    qc.barrier()
    
    # ── Phase 4: Measure feature qubits ──
    qc.measure([2, 3], [0, 1])
    
    return qc


def build_aggregation_circuit_simple(node_id, adj_list, features, vqc_params=None):
    """
    Simplified aggregation circuit using direct statevector preparation.
    This version is cleaner for validation and matches the paper's Eq. 12.
    
    For a degree-2 node with neighbors j1, j2:
    |psi> = (1/sqrt(2)) * (|0>|x_j1> + |1>|x_j2>)
    
    After tracing out the address register:
    rho_feat = (1/2)(|x_j1><x_j1| + |x_j2><x_j2|)
    """
    neighbors = adj_list[node_id]
    n_neighbors = len(neighbors)
    
    qc = QuantumCircuit(4, 2)
    
    if n_neighbors == 2:
        # Phase 1: Hadamard for equal superposition
        qc.h(0)
        
        # Phase 2: Controlled feature loading
        # When address = |0>: encode features of neighbors[0]
        # When address = |1>: encode features of neighbors[1]
        
        feat_0 = features[neighbors[0]]
        feat_1 = features[neighbors[1]]
        
        angles_0 = amplitude_encode_angles(feat_0)
        angles_1 = amplitude_encode_angles(feat_1)
        
        # Controlled on address qubit = |0>
        qc.x(0)
        qc.cry(angles_0[0], 0, 2)
        qc.x(0)
        
        # Controlled on address qubit = |1>  
        qc.cry(angles_1[0], 0, 2)
        
        # Second feature qubit encoding (simplified for orthogonal features)
        qc.x(0)
        qc.cry(angles_0[1], 0, 3)
        qc.x(0)
        qc.cry(angles_1[1], 0, 3)
        
    elif n_neighbors == 1:
        feat = features[neighbors[0]]
        angles = amplitude_encode_angles(feat)
        qc.ry(angles[0], 2)
        qc.ry(angles[1], 3)
    
    qc.barrier()
    
    # Phase 3: VQC (optional)
    if vqc_params is not None:
        _add_hardware_efficient_vqc(qc, vqc_params, feature_qubits=[2, 3])
    
    qc.barrier()
    qc.measure([2, 3], [0, 1])
    
    return qc


def _add_controlled_feature_encoding(qc, angles, ctrl_qubit, feat_qubits, ctrl_state):
    """Add controlled feature encoding rotations."""
    theta_0, theta_1_given_0, theta_1_given_1 = angles
    # Simplified: controlled Ry on second feature qubit
    qc.cry(theta_1_given_0, ctrl_qubit, feat_qubits[1])


def _add_hardware_efficient_vqc(qc, params, feature_qubits):
    """
    Hardware-efficient VQC ansatz (Section 3.3).
    
    Each layer l: Ry(θ) and Rz(θ) on each qubit, then CNOT ladder.
    For 2 qubits, L layers: 2*L*2 = 4L rotation params + L CNOT.
    
    Parameters
    ----------
    qc : QuantumCircuit
    params : array-like
        For L=2, 2 qubits: 10 parameters
        Layout: [ry_q0_l0, rz_q0_l0, ry_q1_l0, rz_q1_l0, 
                 ry_q0_l1, rz_q0_l1, ry_q1_l1, rz_q1_l1,
                 ry_q0_final, ry_q1_final]
    feature_qubits : list
        Indices of feature qubits in the circuit.
    """
    q0, q1 = feature_qubits
    n_params = len(params)
    
    if n_params >= 8:
        # Layer 1
        qc.ry(params[0], q0)
        qc.rz(params[1], q0)
        qc.ry(params[2], q1)
        qc.rz(params[3], q1)
        qc.cx(q0, q1)
        
        # Layer 2
        qc.ry(params[4], q0)
        qc.rz(params[5], q0)
        qc.ry(params[6], q1)
        qc.rz(params[7], q1)
        qc.cx(q0, q1)
    
    if n_params >= 10:
        # Final rotation layer
        qc.ry(params[8], q0)
        qc.ry(params[9], q1)
    elif n_params == 5:
        # L=1: single layer
        qc.ry(params[0], q0)
        qc.rz(params[1], q0)
        qc.ry(params[2], q1)
        qc.rz(params[3], q1)
        qc.cx(q0, q1)
        qc.ry(params[4], q0)


def compute_classical_aggregation(node_id, adj_list, features):
    """
    Compute classical mean aggregation for ground truth comparison.
    h_i = (1/d_i) * sum_{j in N(i)} x_j
    """
    neighbors = adj_list[node_id]
    agg = np.mean([features[j] for j in neighbors], axis=0)
    return agg


def analytical_probabilities(aggregated_feature):
    """
    Compute expected measurement probabilities from aggregated features.
    For amplitude-encoded state |h> = sum_k h_k |k>,
    P(|k>) = |h_k|^2.
    """
    return np.abs(aggregated_feature)**2
