"""Estimate the operator-norm approximate 2-design epsilon for Figure 9.

This script uses the same local Strongly Entangling block distribution as
``figure9_reproduction.ipynb``.  The full superoperator matrix would have size
65536 x 65536 for n=4, so the script applies the superoperator implicitly and
uses power iteration on A^dagger A, where A = M^(2) - H^(2).

Run from the repository root:

    python Paper_work/operator_norm_epsilon.py

Optional controls:

    OP_NORM_SAMPLES=5000 OP_NORM_ITERS=30 python Paper_work/operator_norm_epsilon.py
    OP_NORM_M_VALUES=4 OP_NORM_SAMPLES=5000 python Paper_work/operator_norm_epsilon.py
"""

import os
import time

import numpy as np


N_QUBITS = 6
D = 2**N_QUBITS
D2 = D * D
SUB_L = 5
M_VALUES = tuple(int(value) for value in os.environ.get("OP_NORM_M_VALUES", "6").split(","))
N_SAMPLES = int(os.environ.get("OP_NORM_SAMPLES", "1"))
N_ITERS = int(os.environ.get("OP_NORM_ITERS", "1"))
N_RESTARTS = int(os.environ.get("OP_NORM_RESTARTS", "1"))
SEED = int(os.environ.get("OP_NORM_SEED", "20260520"))

I2 = np.eye(2, dtype=complex)


def rz(theta):
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


def ry(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rot(phi, theta, omega):
    return rz(omega) @ ry(theta) @ rz(phi)


def kron_all(mats):
    out = np.array([[1.0 + 0.0j]])
    for mat in mats:
        out = np.kron(out, mat)
    return out


def full_single_qubit_gate(gate, wire, n_qubits):
    mats = [I2] * n_qubits
    mats[wire] = gate
    return kron_all(mats)


_CNOT_CACHE = {}


def cnot_matrix(n_qubits, control, target):
    key = (n_qubits, control, target)
    if key in _CNOT_CACHE:
        return _CNOT_CACHE[key]

    size = 2**n_qubits
    matrix = np.zeros((size, size), dtype=complex)
    for column in range(size):
        bits = [(column >> (n_qubits - 1 - idx)) & 1 for idx in range(n_qubits)]
        out_bits = bits.copy()
        if bits[control]:
            out_bits[target] ^= 1
        row = 0
        for bit in out_bits:
            row = (row << 1) | bit
        matrix[row, column] = 1.0

    _CNOT_CACHE[key] = matrix
    return matrix


def strongly_entangling_unitary(m_wires, rng):
    weights = 2 * np.pi * rng.standard_normal((SUB_L, m_wires, 3))
    unitary = np.eye(2**m_wires, dtype=complex)

    for layer in range(SUB_L):
        for wire in range(m_wires):
            unitary = full_single_qubit_gate(rot(*weights[layer, wire]), wire, m_wires) @ unitary
        if m_wires > 1:
            gate_range = (layer % (m_wires - 1)) + 1
            for wire in range(m_wires):
                unitary = cnot_matrix(m_wires, wire, (wire + gate_range) % m_wires) @ unitary

    return unitary


def local_trainable_layer(m_wires, rng):
    blocks = [strongly_entangling_unitary(m_wires, rng) for _ in range(N_QUBITS // m_wires)]
    return kron_all(blocks)


def build_global_swap():
    swap = np.zeros((D2, D2), dtype=complex)
    for a in range(D):
        for b in range(D):
            swap[b * D + a, a * D + b] = 1.0
    return swap


SWAP = build_global_swap()
IDENTITY_D2 = np.eye(D2, dtype=complex)
P_SYM = (IDENTITY_D2 + SWAP) / 2
P_ASYM = (IDENTITY_D2 - SWAP) / 2
DIM_SYM = D * (D + 1) / 2
DIM_ASYM = D * (D - 1) / 2


def haar_twirl_second_moment(x):
    """Haar second-moment channel on operators over H \\otimes H."""
    coeff_sym = np.trace(P_SYM @ x) / DIM_SYM
    coeff_asym = np.trace(P_ASYM @ x) / DIM_ASYM
    return coeff_sym * P_SYM + coeff_asym * P_ASYM


def apply_q_conjugation(x, unitary, dagger=False):
    """Apply (U otimes U) X (U^dagger otimes U^dagger), or its adjoint."""
    if dagger:
        unitary = unitary.conj().T
    unitary_conj = unitary.conj()
    x4 = x.reshape(D, D, D, D)
    tmp = np.einsum("ia,abcd->ibcd", unitary, x4, optimize=True)
    tmp = np.einsum("jb,ibcd->ijcd", unitary, tmp, optimize=True)
    tmp = np.einsum("kc,ijcd->ijkd", unitary_conj, tmp, optimize=True)
    out = np.einsum("ld,ijkd->ijkl", unitary_conj, tmp, optimize=True)
    return out.reshape(D2, D2)


def apply_moment(x, unitaries, dagger=False):
    out = np.zeros_like(x)
    for unitary in unitaries:
        out += apply_q_conjugation(x, unitary, dagger=dagger)
    return out / len(unitaries)


def apply_difference(x, unitaries, dagger=False):
    return apply_moment(x, unitaries, dagger=dagger) - haar_twirl_second_moment(x)


def estimate_spectral_norm(unitaries, seed):
    rng = np.random.default_rng(seed)
    best_sigma = -1.0
    best_history = []

    for _ in range(N_RESTARTS):
        x = rng.standard_normal((D2, D2)) + 1j * rng.standard_normal((D2, D2))
        x /= np.linalg.norm(x)

        history = []
        for _ in range(N_ITERS):
            y = apply_difference(x, unitaries, dagger=False)
            z = apply_difference(y, unitaries, dagger=True)
            z_norm = np.linalg.norm(z)
            if z_norm == 0:
                break
            x = z / z_norm
            history.append(float(np.linalg.norm(y)))

        sigma = float(np.linalg.norm(apply_difference(x, unitaries, dagger=False)))
        if sigma > best_sigma:
            best_sigma = sigma
            best_history = history

    return best_sigma, best_history


def main():
    print("Operator-norm epsilon for Figure 9 local-block circuits")
    print(
        "n_qubits={}, sub_l={}, m_values={}, samples={}, power_iters={}, restarts={}".format(
            N_QUBITS,
            SUB_L,
            M_VALUES,
            N_SAMPLES,
            N_ITERS,
            N_RESTARTS,
        )
    )
    print()

    results = {}
    total_start = time.time()
    for m_wires in M_VALUES:
        start = time.time()
        rng = np.random.default_rng(SEED + 37 * m_wires)
        unitaries = [local_trainable_layer(m_wires, rng) for _ in range(N_SAMPLES)]
        build_elapsed = time.time() - start
        epsilon_op, history = estimate_spectral_norm(unitaries, SEED + 1000 * m_wires)
        elapsed = time.time() - start
        results[m_wires] = epsilon_op

        print(
            "m={}: epsilon_op ~= {:.8f}; build_seconds={:.2f}; total_seconds={:.2f}; last five iterates={}".format(
                m_wires,
                epsilon_op,
                build_elapsed,
                elapsed,
                [round(value, 8) for value in history[-5:]],
            )
        )

    print()
    print("summary:", {m_wires: round(value, 8) for m_wires, value in results.items()})
    print("total_elapsed_seconds={:.2f}".format(time.time() - total_start))
    print()
    print("The same values apply to M_1^(2)-H^(2) and tilde(M)_2^(2)-H^(2),")
    print("because tilde(M)_2^(2) is the Hilbert-Schmidt adjoint channel for the same unitary ensemble.")


if __name__ == "__main__":
    main()
