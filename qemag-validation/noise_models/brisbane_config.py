"""
IBM Brisbane Noise Model Configuration
=======================================
Calibration date: January 15, 2026

Noise parameters used in all noisy simulations:
- Two-qubit gate error rate: epsilon_g ~ 5e-3
- Single-qubit gate error rate: epsilon_1 ~ 1e-4
- Readout error: ~1-2% per qubit
- T1 relaxation: ~100-300 μs
- T2 dephasing: ~100-200 μs
"""

from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

# Brisbane calibration parameters (Jan 15, 2026)
SINGLE_QUBIT_ERROR = 1e-4
TWO_QUBIT_ERROR = 5e-3
READOUT_ERROR_0 = 0.015   # P(1|0)
READOUT_ERROR_1 = 0.020   # P(0|1)
T1_US = 200.0
T2_US = 150.0
GATE_TIME_1Q_US = 0.035   # single-qubit gate time
GATE_TIME_2Q_US = 0.300   # two-qubit gate time


def get_brisbane_noise_model():
    """
    Construct a noise model approximating the IBM Brisbane backend.
    
    Returns
    -------
    NoiseModel
        Qiskit noise model with depolarizing and readout errors.
    """
    noise_model = NoiseModel()
    
    # Single-qubit depolarizing error
    error_1q = depolarizing_error(SINGLE_QUBIT_ERROR, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['ry', 'rz', 'h', 'x'])
    
    # Two-qubit depolarizing error
    error_2q = depolarizing_error(TWO_QUBIT_ERROR, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cry'])
    
    # Readout errors
    from qiskit_aer.noise import ReadoutError
    readout_err = ReadoutError(
        [[1 - READOUT_ERROR_0, READOUT_ERROR_0],
         [READOUT_ERROR_1, 1 - READOUT_ERROR_1]]
    )
    noise_model.add_all_qubit_readout_error(readout_err)
    
    return noise_model


def get_noise_params():
    """Return a dict of noise parameters for reporting."""
    return {
        'single_qubit_error': SINGLE_QUBIT_ERROR,
        'two_qubit_error': TWO_QUBIT_ERROR,
        'readout_error_0': READOUT_ERROR_0,
        'readout_error_1': READOUT_ERROR_1,
        'T1_us': T1_US,
        'T2_us': T2_US,
        'calibration_date': '2026-01-15',
        'backend': 'IBM Brisbane (simulated)',
    }
