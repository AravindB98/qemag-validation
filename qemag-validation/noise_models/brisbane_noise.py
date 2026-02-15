"""
IBM Brisbane Noise Model Configuration for QEMA-G Validation

Approximates the noise characteristics of the IBM Brisbane backend
as used in the paper's toy-scale validation experiments.
"""

from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit_aer.noise import ReadoutError
import numpy as np


def get_brisbane_noise_model(
    two_qubit_error: float = 5e-3,
    single_qubit_error: float = 3e-4,
    t1_us: float = 100.0,
    t2_us: float = 80.0,
    readout_error_prob: float = 0.015,
    gate_time_single_us: float = 0.06,
    gate_time_two_us: float = 0.66,
) -> NoiseModel:
    """
    Construct a noise model approximating IBM Brisbane backend.

    Parameters
    ----------
    two_qubit_error : float
        Depolarizing error rate for two-qubit (CX) gates.
    single_qubit_error : float
        Depolarizing error rate for single-qubit gates.
    t1_us : float
        T1 relaxation time in microseconds.
    t2_us : float
        T2 dephasing time in microseconds.
    readout_error_prob : float
        Probability of measurement bit-flip error.
    gate_time_single_us : float
        Duration of single-qubit gates in microseconds.
    gate_time_two_us : float
        Duration of two-qubit gates in microseconds.

    Returns
    -------
    NoiseModel
        Configured Qiskit Aer noise model.
    """
    noise_model = NoiseModel()

    # Depolarizing errors
    error_1q = depolarizing_error(single_qubit_error, 1)
    error_2q = depolarizing_error(two_qubit_error, 2)

    # Thermal relaxation errors
    thermal_1q = thermal_relaxation_error(
        t1_us * 1e-6, t2_us * 1e-6, gate_time_single_us * 1e-6
    )
    thermal_2q_single = thermal_relaxation_error(
        t1_us * 1e-6, t2_us * 1e-6, gate_time_two_us * 1e-6
    )

    # Compose depolarizing + thermal for single-qubit gates
    combined_1q = error_1q.compose(thermal_1q)
    noise_model.add_all_qubit_quantum_error(combined_1q, ["ry", "rz", "h", "x", "z", "s", "sdg"])

    # Compose depolarizing + thermal for two-qubit gates
    combined_2q = error_2q.compose(thermal_2q_single.tensor(thermal_2q_single))
    noise_model.add_all_qubit_quantum_error(combined_2q, ["cx", "cz"])

    # Readout errors
    p0_given_1 = readout_error_prob  # P(measure 0 | state is 1)
    p1_given_0 = readout_error_prob  # P(measure 1 | state is 0)
    readout_err = ReadoutError(
        [[1 - p1_given_0, p1_given_0], [p0_given_1, 1 - p0_given_1]]
    )
    noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model


def get_noise_model_summary() -> dict:
    """Return a summary dict of noise parameters for logging."""
    return {
        "backend": "ibm_brisbane (approximate)",
        "two_qubit_gate_error": 5e-3,
        "single_qubit_gate_error": 3e-4,
        "T1_us": 100.0,
        "T2_us": 80.0,
        "readout_error": 0.015,
        "gate_time_1q_us": 0.06,
        "gate_time_2q_us": 0.66,
    }


if __name__ == "__main__":
    model = get_brisbane_noise_model()
    summary = get_noise_model_summary()
    print("Brisbane Noise Model Configuration:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nNoise model basis gates: {model.basis_gates}")
