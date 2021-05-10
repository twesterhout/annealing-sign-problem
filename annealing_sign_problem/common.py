import lattice_symmetries as ls
import ising_glass_annealer as sa
import numpy as np
import scipy.sparse


def _int_to_ls_bits512(x: int, n: int = 8) -> np.ndarray:
    assert n > 0
    if n == 1:
        return x
    x = int(x)
    bits = np.empty(n, dtype=np.uint64)
    for i in range(n):
        bits[i] = x & 0xFFFFFFFFFFFFFFFF
        x >>= 64
    return bits


def _ls_bits512_to_int(bits) -> int:
    n = len(bits)
    if n == 0:
        return 0
    x = int(bits[n - 1])
    for i in range(n - 2, -1, -1):
        x <<= 64
        x |= int(bits[i])
    return x


def batched_apply(spins, hamiltonian):
    spins = np.asarray(spins, dtype=np.uint64, order="C")
    if spins.ndim < 2:
        spins = spins.reshape(-1, 1)
    out_spins = []
    out_coeffs = []
    out_counts = []
    for σ in spins:
        out_counts.append(0)
        for (other_σ, coeff) in zip(*hamiltonian.apply(_ls_bits512_to_int(σ))):
            assert coeff.imag == 0
            out_spins.append(other_σ)
            out_coeffs.append(coeff.real)
            out_counts[-1] += 1
    out_spins = np.stack(out_spins)
    out_coeffs = np.asarray(out_coeffs, dtype=np.float64)
    out_counts = np.asarray(out_counts, dtype=np.int64)
    return out_spins, out_coeffs, out_counts


def extract_classical_ising_model(spins, hamiltonian, log_ψ, sampled: bool = False):
    spins = np.asarray(spins, dtype=np.uint64, order="C")
    if spins.ndim < 2:
        spins = spins.reshape(-1, 1)
    spins, counts = np.unique(spins, return_counts=True, axis=0)
    if not sampled:
        assert np.all(counts == 1)
    ψs = log_ψ(spins).numpy().squeeze(axis=1)
    spins = [_ls_bits512_to_int(σ) for σ in spins]

    other_spins, coeffs, part_lengths = batched_apply(spins, hamiltonian)
    other_ψs = log_ψ(other_spins).numpy().squeeze(axis=1)
    other_spins = [_ls_bits512_to_int(σ) for σ in other_spins]

    scale = np.max(other_ψs.real)
    other_ψs.real -= scale
    other_ψs = np.exp(other_ψs)
    ψs.real -= scale
    ψs = np.exp(ψs)

    matrix = []
    field = np.zeros(len(spins), dtype=np.float64)

    offset = 0
    x0 = np.zeros((len(spins) + 63) // 64, dtype=np.uint64)
    for i, σ in enumerate(spins):
        ψ = ψs[i]
        assert np.isclose(ψ.imag, 0)
        if ψ > 0:
            x0[i // 64] |= np.uint64(1) << np.uint64(i % 64)
        for (other_σ, c, other_ψ) in zip(
            other_spins[offset : offset + part_lengths[i]],
            coeffs[offset : offset + part_lengths[i]],
            other_ψs[offset : offset + part_lengths[i]],
        ):
            assert np.isclose(other_ψ.imag, 0)
            if other_σ in spins:
                j = spins.index(other_σ)
                if sampled:
                    matrix.append((i, j, c * counts[i] * abs(other_ψ.real) / abs(ψ.real)))
                else:
                    matrix.append((i, j, c * abs(other_ψ.real) * abs(ψ.real)))
            else:
                if sampled:
                    field[i] += c * counts[i] * other_ψ.real / abs(ψ.real)
                else:
                    field[i] += c * other_ψ.real * abs(ψ.real)
        offset += part_lengths[i]
    Jmax = max(map(lambda t: abs(t[2]), matrix))
    matrix = list(filter(lambda t: abs(t[2]) > 1e-8 * Jmax, matrix))

    # print(matrix, field)
    matrix = scipy.sparse.coo_matrix(
        ([d for (_, _, d) in matrix],
        ([i for (i, _, _) in matrix], [j for (_, j, _) in matrix])),
        shape=(len(field), len(field))
    )
    spins = np.stack([_int_to_ls_bits512(σ) for σ in spins])
    return sa.Hamiltonian(matrix, field), spins, x0


def optimize_sign_structure(epochs: int = 2):
    for epoch in range(epochs):
        pass


