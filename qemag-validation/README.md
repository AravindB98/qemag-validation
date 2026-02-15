# QEMA-G Validation: Toy-Scale Numerical Experiments

Supplementary code for the paper:

> **Quantum-Enhanced Memory Architectures for Graph-Based AI Systems: A Theoretical Framework with Feasibility Analysis**
> Aravind Balaji and Nik Bear Brown, Northeastern University

This repository contains all simulation code, trained parameters, and noise model configurations needed to reproduce the numerical validation results in Section 7 of the paper.

## Repository Structure

```
qemag-validation/
├── README.md
├── requirements.txt
├── experiments/
│   ├── experiment1_identity_vqc_path.py    # Exp 1: Identity VQC on P4
│   ├── experiment2_trained_vqc_path.py     # Exp 2: Trained VQC on P4
│   ├── experiment3_identity_vqc_cycle.py   # Exp 3: Identity VQC on C4
│   └── run_all.py                          # Run all experiments
├── noise_models/
│   └── brisbane_noise.py                   # IBM Brisbane noise model config
├── trained_params/
│   └── experiment2_params.json             # Trained VQC parameters (10 runs)
├── results/
│   └── (generated after running experiments)
└── LICENSE
```

## Requirements

- Python >= 3.9
- IBM Qiskit >= 1.0
- NumPy >= 1.24
- SciPy >= 1.10

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run all three experiments:

```bash
python experiments/run_all.py
```

Or run individually:

```bash
python experiments/experiment1_identity_vqc_path.py
python experiments/experiment2_trained_vqc_path.py
python experiments/experiment3_identity_vqc_cycle.py
```

## Experiments

### Experiment 1: Identity VQC on Path Graph (P4)
- Tests the aggregation mechanism with U(θ) = I (identity)
- Validates quantum message-passing produces correct mean-aggregated features
- Compares noiseless simulation against analytical ground truth

### Experiment 2: Trained VQC on Path Graph (P4)
- Binary node classification: boundary (degree 1) vs. interior (degree 2)
- L=2 layer hardware-efficient ansatz, 10 parameterized angles
- Adam optimizer with parameter-shift gradients, η=0.01, 100 iterations
- Reports classification accuracy and fidelity across 10 independent runs

### Experiment 3: Identity VQC on Cycle Graph (C4)
- Tests aggregation under uniform-degree (non-tree) topology
- Validates that quantum aggregation distinguishes nodes by neighborhood content

## Expected Results

Results should match Table 4 in the paper:

| Basis state | |00⟩ | |01⟩ | |10⟩ | |11⟩ |
|---|---|---|---|---|
| Analytical (ideal) | 0.0625 | 0.3650 | 0.0625 | 0.3650 |
| Noiseless simulation | 0.0625 | 0.3650 | 0.0625 | 0.3650 |
| Noisy (εg ≈ 5×10⁻³) | 0.068±0.005 | 0.353±0.009 | 0.067±0.005 | 0.356±0.009 |

Noisy results may vary slightly due to stochastic noise simulation.

## Noise Model

The Brisbane noise model uses:
- Two-qubit gate error rate: εg ≈ 5 × 10⁻³
- T1 relaxation: ~100 μs
- T2 dephasing: ~80 μs
- Readout error: ~1.5%

See `noise_models/brisbane_noise.py` for the full configuration.

## Citation

If you use this code, please cite:

```bibtex
@article{balaji2026qemag,
  title={Quantum-Enhanced Memory Architectures for Graph-Based AI Systems:
         A Theoretical Framework with Feasibility Analysis},
  author={Balaji, Aravind and Brown, Nik Bear},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
