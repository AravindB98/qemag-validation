# QEMA-G Validation Code

**Quantum-Enhanced Memory Architectures for Graph-Based AI Systems**

Aravind Balaji and Nik Bear Brown  
Northeastern University, Boston, MA 02115, USA  
balaji.ara@northeastern.edu, ni.brown@northeastern.edu

---

## Overview

This repository contains the complete Qiskit simulation code for reproducing all numerical results reported in the QEMA-G paper. The code validates the quantum message-passing protocol on toy-scale graphs under both noiseless and noisy (IBM Brisbane) simulation conditions.

## Repository Structure

```
qemag-validation/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── experiment1/                   # Identity VQC on Path Graph P4
│   └── exp1_identity_path.py
├── experiment2/                   # Trained VQC on Path Graph P4
│   └── exp2_trained_path.py
├── experiment3/                   # Identity VQC on Cycle Graph C4
│   └── exp3_identity_cycle.py
├── experiment4/                   # 8-Node VQC Classification
│   ├── exp4_8node_classification.py
│   └── graph_spec.json            # Complete G8 specification
├── noise_models/
│   └── brisbane_config.py         # IBM Brisbane noise model config
├── utils/
│   └── qemag_circuits.py          # Shared circuit construction utilities
└── run_all.py                     # Run all experiments
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python run_all.py

# Or run individual experiments
python experiment1/exp1_identity_path.py
python experiment2/exp2_trained_path.py
python experiment3/exp3_identity_cycle.py
python experiment4/exp4_8node_classification.py
```

## Requirements

- Python >= 3.9
- Qiskit == 1.0
- Qiskit Aer == 0.14
- NumPy >= 1.24
- SciPy >= 1.11

## Experiments

### Experiment 1: Identity VQC on Path Graph (Table 4, Row 1)
Tests the quantum aggregation mechanism with U(θ) = I on P4.
Validates that measurement probabilities match analytical ground truth.

### Experiment 2: Trained VQC on Path Graph (Section 7.3)
Binary classification (boundary vs. interior nodes) with L=2 VQC.
Validates the hybrid classical-quantum training pipeline.

### Experiment 3: Identity VQC on Cycle Graph (Table 4, Row 2)
Tests aggregation on C4 with uniform-degree topology.
Validates neighborhood-content-based discrimination.

### Experiment 4: 8-Node Classification (Section 7.5)
Multi-class classification on G8 with 3 structural classes.
Includes classical baseline comparison and initialization sensitivity.

## Noise Model

All noisy simulations use the IBM Brisbane backend noise model.
- Calibration date: January 15, 2026
- Two-qubit gate error rate: ε_g ≈ 5 × 10⁻³
- Single-qubit gate error rate: ε_1 ≈ 10⁻⁴
- Readout error: ~1-2% per qubit

## Citation

```bibtex
@article{balaji2026qemag,
  title={Quantum-Enhanced Memory Architectures for Graph-Based AI Systems: 
         A Theoretical Framework with Feasibility Analysis},
  author={Balaji, Aravind and Brown, Nik Bear},
  year={2026},
  institution={Northeastern University}
}
```

## License

MIT License. See LICENSE for details.
