"""Config-driven Pauli/exponential encoding bound comparisons.

This module estimates two approximate-design errors for the chosen local-block
circuit ensemble:

1. ``epsilon_monomial``: the paper-style monomial error used for Theorem 7.
2. ``epsilon_op``: the operator norm used in ``MSQE/main.tex``.

It then plots Pauli-encoding and exponential-encoding versions of the bound
comparison.
"""

from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np


@dataclass
class AutoBoundConfig:
    n_qubits: int = 4
    sub_l: int = 2
    m_values: tuple = (4,)
    error_samples: int = 300
    op_iters: int = 14
    op_restarts: int = 1
    simulation_samples: int = 1000
    seed: int = 20260520
    output_dir: Path = Path(".")


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


def rx(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


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


def strongly_entangling_unitary(m_wires, sub_l, rng):
    weights = 2 * pi * rng.standard_normal((sub_l, m_wires, 3))
    unitary = np.eye(2**m_wires, dtype=complex)

    for layer in range(sub_l):
        for wire in range(m_wires):
            unitary = full_single_qubit_gate(rot(*weights[layer, wire]), wire, m_wires) @ unitary
        if m_wires > 1:
            gate_range = (layer % (m_wires - 1)) + 1
            for wire in range(m_wires):
                unitary = cnot_matrix(m_wires, wire, (wire + gate_range) % m_wires) @ unitary

    return unitary


def local_trainable_layer(n_qubits, m_wires, sub_l, rng):
    if n_qubits % m_wires != 0:
        raise ValueError("Each m_wires value must divide n_qubits.")
    blocks = [strongly_entangling_unitary(m_wires, sub_l, rng) for _ in range(n_qubits // m_wires)]
    return kron_all(blocks)


def weighted_encoding_redundancies(betas):
    counts = {0: 1}
    for beta in betas:
        beta = int(beta)
        next_counts = {}
        for freq, multiplicity in counts.items():
            for delta, delta_mult in ((-beta, 1), (0, 2), (beta, 1)):
                next_counts[freq + delta] = next_counts.get(freq + delta, 0) + multiplicity * delta_mult
        counts = next_counts
    freqs = np.array(sorted(freq for freq in counts if freq >= 0), dtype=int)
    redundancies = np.array([counts[int(freq)] for freq in freqs], dtype=float)
    return freqs, redundancies


def theorem5_global_projector_variance(redundancies, d):
    trace_o = 1.0
    norm2_sq = 1.0
    variances = (
        (d * norm2_sq - trace_o**2)
        / (d * (d**2 - 1))
        * redundancies
        / (d * (d + 1))
    )
    variances = variances.copy()
    variances[0] += (trace_o**2 - d * norm2_sq) / (d**2 * (d**2 - 1))
    return variances


def theorem7_monomial_bound(var_2design, redundancies, d, epsilon_monomial):
    trace_o = 1.0
    norm2_sq = 1.0
    sum_abs_o_tensor_sq = 1.0
    c1 = (d * norm2_sq - trace_o**2) / (d * (d**2 - 1))
    c2 = sum_abs_o_tensor_sq / d**2
    linear_term = (c1 / d**2 + c2 / (d * (d + 1))) * epsilon_monomial * redundancies
    quadratic_term = (c2 / d**2) * (epsilon_monomial * redundancies) ** 2
    return var_2design + linear_term + quadratic_term


def theorem7_spectral_bound(var_2design, redundancies, d, epsilon_infty):
    """Original Theorem 7 spectral/operator-norm-style bound.

    For this comparison we set epsilon_infty equal to the estimated operator
    norm error.  In the global projector case, Tr(O)=||O||_2^2=1.
    """
    trace_o = 1.0
    norm2_sq = 1.0
    c1 = (d * norm2_sq - trace_o**2) / (d * (d**2 - 1))
    leading = (c1 + norm2_sq / (d * (d + 1))) * epsilon_infty * np.sqrt(redundancies)
    quadratic = norm2_sq * (epsilon_infty**2) * redundancies
    return var_2design + leading + quadratic


def operator_norm_bound(
    var_2design,
    freqs,
    redundancies,
    d,
    epsilon_op,
    observable_frobenius_norm_sq=1.0,
    trace_o=1.0,
):
    leading_prefactor = (
        (2 * d - 1) * observable_frobenius_norm_sq - trace_o**2
    ) / (d * (d**2 - 1))
    bound = (
        var_2design
        + epsilon_op * np.sqrt(redundancies) * leading_prefactor
        + (epsilon_op**2 + epsilon_op**4) * observable_frobenius_norm_sq
    )
    bound = bound.copy()
    bound[freqs == 0] = np.nan
    return bound


def build_swap(d):
    d2 = d * d
    swap = np.zeros((d2, d2), dtype=complex)
    for a in range(d):
        for b in range(d):
            swap[b * d + a, a * d + b] = 1.0
    return swap


def haar_data(d):
    d2 = d * d
    swap = build_swap(d)
    identity = np.eye(d2, dtype=complex)
    p_sym = (identity + swap) / 2
    p_asym = (identity - swap) / 2
    dim_sym = d * (d + 1) / 2
    dim_asym = d * (d - 1) / 2
    return p_sym, p_asym, dim_sym, dim_asym


def haar_twirl(x, p_sym, p_asym, dim_sym, dim_asym):
    return (np.trace(p_sym @ x) / dim_sym) * p_sym + (np.trace(p_asym @ x) / dim_asym) * p_asym


def q_conjugation(x, unitary, d, dagger=False):
    if dagger:
        unitary = unitary.conj().T
    unitary_conj = unitary.conj()
    x4 = x.reshape(d, d, d, d)
    tmp = np.einsum("ia,abcd->ibcd", unitary, x4, optimize=True)
    tmp = np.einsum("jb,ibcd->ijcd", unitary, tmp, optimize=True)
    tmp = np.einsum("kc,ijcd->ijkd", unitary_conj, tmp, optimize=True)
    out = np.einsum("ld,ijkd->ijkl", unitary_conj, tmp, optimize=True)
    return out.reshape(d * d, d * d)


def moment_action(x, unitaries, d, dagger=False):
    out = np.zeros_like(x)
    for unitary in unitaries:
        out += q_conjugation(x, unitary, d, dagger=dagger)
    return out / len(unitaries)


def estimate_operator_epsilon(unitaries, d, op_iters, op_restarts, seed):
    d2 = d * d
    p_sym, p_asym, dim_sym, dim_asym = haar_data(d)

    def diff(x, dagger=False):
        return moment_action(x, unitaries, d, dagger=dagger) - haar_twirl(x, p_sym, p_asym, dim_sym, dim_asym)

    rng = np.random.default_rng(seed)
    best = -1.0
    for _ in range(op_restarts):
        x = rng.standard_normal((d2, d2)) + 1j * rng.standard_normal((d2, d2))
        x /= np.linalg.norm(x)
        for _ in range(op_iters):
            y = diff(x, dagger=False)
            z = diff(y, dagger=True)
            z_norm = np.linalg.norm(z)
            if z_norm == 0:
                break
            x = z / z_norm
        best = max(best, float(np.linalg.norm(diff(x, dagger=False))))
    return best


def estimate_paper_monomial_epsilon(unitaries, d, batch_size=16):
    """Estimate the monomial epsilon used in the original Figure 9 code.

    This follows the same reduced set of monomials as ``get_epsilon`` in
    ``statisticsreuploading.py``: for each basis index p=(a,b), it compares the
    channel action on |p><swap(p)| against the Haar value and takes
    d^2 times the largest matrix-entry deviation.
    """
    d2 = d * d
    p_sym, p_asym, dim_sym, dim_asym = haar_data(d)
    vv = np.stack([np.kron(unitary, unitary) for unitary in unitaries], axis=0)
    swapped = np.array([(idx % d) * d + (idx // d) for idx in range(d2)], dtype=int)

    max_distance = 0.0
    for start in range(0, d2, batch_size):
        stop = min(start + batch_size, d2)
        p_batch = np.arange(start, stop)
        q_batch = swapped[p_batch]

        cols_p = vv[:, :, p_batch]
        cols_q = vv[:, :, q_batch]
        empirical_blocks = np.einsum("sib,sjb->bij", cols_p, np.conj(cols_q), optimize=True) / len(unitaries)

        for block_idx, (p_idx, q_idx) in enumerate(zip(p_batch, q_batch)):
            haar_block = (
                (p_sym[q_idx, p_idx] / dim_sym) * p_sym
                + (p_asym[q_idx, p_idx] / dim_asym) * p_asym
            )
            max_distance = max(max_distance, float(np.max(np.abs(empirical_blocks[block_idx] - haar_block))))

    return d2 * max_distance


def encoding_layers_from_betas(betas):
    max_freq = int(np.sum(betas))
    steps = 2 * max_freq + 1
    x_grid = 2 * np.pi * np.arange(steps) / steps
    return [kron_all([rx(float(beta) * x) for beta in betas]) for x in x_grid]


def simulate_variances(n_qubits, m_wires, sub_l, encoding_layers, n_samples, seed):
    d = 2**n_qubits
    ket_zero = np.zeros(d, dtype=complex)
    ket_zero[0] = 1.0
    rng = np.random.default_rng(seed)
    steps = len(encoding_layers)
    coeffs = []
    for _ in range(n_samples):
        v_left = local_trainable_layer(n_qubits, m_wires, sub_l, rng)
        v_right = local_trainable_layer(n_qubits, m_wires, sub_l, rng)
        values = []
        for encoding in encoding_layers:
            state = v_right @ (encoding @ (v_left @ ket_zero))
            values.append(abs(state[0]) ** 2)
        coeffs.append(np.fft.rfft(np.array(values)) / steps)
    coeffs = np.array(coeffs)
    return np.mean(np.abs(coeffs) ** 2, axis=0) - np.abs(np.mean(coeffs, axis=0)) ** 2


def plot_encoding(
    name,
    freqs,
    var_2design,
    monomial_bounds,
    spectral_bounds,
    new_bounds,
    simulations,
    errors,
    output_path,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    m_values = tuple(errors)
    ncols = len(m_values)
    fig, axs = plt.subplots(1, ncols, figsize=(max(3.0 * ncols, 4.0), 1.9), sharey=True)
    if ncols == 1:
        axs = [axs]

    colors = {
        "simulation": "#35b897",
        "spectral": "#7E57C2",
        "new": "#E69F00",
    }
    nonzero = freqs > 0
    for ax, m_wires in zip(axs, m_values):
        ax.plot(freqs, simulations[m_wires], marker="o", color=colors["simulation"], label="Simulation", markersize=2.5, linewidth=0.8)
        ax.plot(freqs, spectral_bounds[m_wires], marker="d", color=colors["spectral"], label=r"Old spectral, $\epsilon_\infty=\epsilon_{op}$", markersize=2.5, linewidth=0.8)
        ax.plot(freqs[nonzero], new_bounds[m_wires][nonzero], marker="^", color=colors["new"], label=r"New bound, $\epsilon_{op}$", markersize=2.5, linewidth=0.8)
        ax.set_yscale("log")
        y_values = np.concatenate(
            [
                simulations[m_wires][np.isfinite(simulations[m_wires])],
                spectral_bounds[m_wires][np.isfinite(spectral_bounds[m_wires])],
                new_bounds[m_wires][np.isfinite(new_bounds[m_wires])],
            ]
        )
        y_values = y_values[y_values > 0]
        if len(y_values) > 0:
            y_min = 10 ** np.floor(np.log10(np.min(y_values)) - 0.2)
            y_max = 10 ** np.ceil(np.log10(np.max(y_values)) + 0.2)
            ax.set_ylim(y_min, y_max)
        ax.grid(linewidth=0.3, alpha=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", which="major", labelsize=6, pad=0)
        ax.set_xlabel(r"Frequency $\omega$", fontsize=10)
        ax.text(
            0.04,
            0.82,
            r"$m={}$, $\epsilon_M={:.3g}$, $\epsilon_{{op}}={:.3g}$".format(
                m_wires,
                errors[m_wires]["monomial"],
                errors[m_wires]["op"],
            ),
            transform=ax.transAxes,
            fontsize=6.5,
        )

    axs[0].set_ylabel(r"$\mathrm{Var}[c_\omega]$", fontsize=10)
    axs[min(1, ncols - 1)].legend(loc="upper center", bbox_to_anchor=(0.5, 1.34), ncol=3, fontsize=6.5, frameon=True)
    fig.suptitle(name, fontsize=10, y=1.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def run_auto_bound_comparison(config):
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    d = 2**config.n_qubits
    errors = {}
    for m_wires in config.m_values:
        rng = np.random.default_rng(config.seed + 37 * m_wires + 10000 * config.sub_l)
        unitaries = [
            local_trainable_layer(config.n_qubits, m_wires, config.sub_l, rng)
            for _ in range(config.error_samples)
        ]
        epsilon_monomial = estimate_paper_monomial_epsilon(unitaries, d)
        epsilon_op = estimate_operator_epsilon(
            unitaries,
            d,
            config.op_iters,
            config.op_restarts,
            config.seed + 1000 * m_wires + 10000 * config.sub_l,
        )
        errors[m_wires] = {"monomial": epsilon_monomial, "op": epsilon_op}
        print("m={}: epsilon_M ~= {:.6g}, epsilon_op ~= {:.6g}".format(m_wires, epsilon_monomial, epsilon_op))

    outputs = {}
    for encoding_name, betas in (
        ("Pauli encoding", np.ones(config.n_qubits, dtype=int)),
        ("Exponential encoding", 3 ** np.arange(config.n_qubits)),
    ):
        freqs, redundancies = weighted_encoding_redundancies(betas)
        var_2design = theorem5_global_projector_variance(redundancies, d)
        monomial_bounds = {
            m: theorem7_monomial_bound(var_2design, redundancies, d, errors[m]["monomial"])
            for m in config.m_values
        }
        spectral_bounds = {
            m: theorem7_spectral_bound(var_2design, redundancies, d, errors[m]["op"])
            for m in config.m_values
        }
        new_bounds = {
            m: operator_norm_bound(var_2design, freqs, redundancies, d, errors[m]["op"])
            for m in config.m_values
        }
        encoding_layers = encoding_layers_from_betas(betas)
        simulations = {
            m: simulate_variances(
                config.n_qubits,
                m,
                config.sub_l,
                encoding_layers,
                config.simulation_samples,
                config.seed + 17 * m + 1000 * len(freqs),
            )
            for m in config.m_values
        }

        file_stem = encoding_name.lower().replace(" ", "_")
        output_path = config.output_dir / "auto_{}_sub_l_{}_bounds.png".format(file_stem, config.sub_l)
        plot_encoding(
            encoding_name,
            freqs,
            var_2design,
            monomial_bounds,
            spectral_bounds,
            new_bounds,
            simulations,
            errors,
            output_path,
        )
        print("saved to", output_path.resolve())
        outputs[encoding_name] = output_path

    return {"errors": errors, "outputs": outputs}
