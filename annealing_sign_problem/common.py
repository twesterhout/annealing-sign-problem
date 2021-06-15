import lattice_symmetries as ls
import ising_glass_annealer as sa
import numpy as np
import scipy.sparse
from loguru import logger
from ctypes import CDLL, POINTER, c_int64, c_uint32, c_uint64, c_double
import os

_lib = CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)), "libbuild_matrix.so"))


def __preprocess_library():
    # fmt: off
    info = [
        ("build_matrix", [c_uint64, POINTER(c_uint64), POINTER(c_int64), POINTER(c_double),
            POINTER(c_uint64), POINTER(c_double), POINTER(c_int64), POINTER(c_double),
            POINTER(c_uint32), POINTER(c_uint32), POINTER(c_double), POINTER(c_double)], c_uint64),
        ("extract_signs", [c_uint64, POINTER(c_double), POINTER(c_uint64)], None),
    ]
    # fmt: on
    for (name, argtypes, restype) in info:
        f = getattr(_lib, name)
        f.argtypes = argtypes
        f.restype = restype


__preprocess_library()


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
    logger.debug(
        "max_buffer_size = {}, used = {}, max = {}",
        hamiltonian.max_buffer_size,
        out_spins.shape[0] / spins.shape[0],
        max(out_counts),
    )
    return out_spins, out_coeffs, out_counts


def extract_classical_ising_model(spins, hamiltonian, log_ψ, sampled: bool = False):
    logger.info("Constructing classical Ising model...")
    spins = np.asarray(spins, dtype=np.uint64, order="C")
    if spins.ndim == 1:
        spins = np.hstack([spins.reshape(-1, 1), np.zeros((spins.shape[0], 7), dtype=np.uint64)])
    elif spins.ndim == 2:
        if spins.shape[1] != 8:
            raise ValueError("'x' has wrong shape: {}; expected (?, 8)".format(x.shape))
        spins = np.ascontiguousarray(spins)
    else:
        raise ValueError("'x' has wrong shape: {}; expected a 2D array".format(x.shape))
    spins, counts = np.unique(spins, return_counts=True, axis=0)
    if not sampled:
        assert np.all(counts == 1)
    ψs = log_ψ(spins).numpy().squeeze(axis=1)

    other_spins, other_coeffs, other_counts = hamiltonian.batched_apply(spins)
    assert np.allclose(other_coeffs.imag, 0)
    other_coeffs = np.ascontiguousarray(other_coeffs.real)
    other_ψs = log_ψ(other_spins).numpy().squeeze(axis=1)

    scale = np.max(other_ψs.real)
    other_ψs.real -= scale
    other_ψs = np.exp(other_ψs, dtype=np.complex128)
    assert np.allclose(other_ψs.imag, 0)
    other_ψs = np.ascontiguousarray(other_ψs.real)
    ψs.real -= scale
    ψs = np.exp(ψs, dtype=np.complex128)
    assert np.allclose(ψs.imag, 0)
    ψs = np.ascontiguousarray(ψs.real)

    if sampled:
        ψs = 1 / ψs

    field = np.zeros(spins.shape[0], dtype=np.float64)
    row_indices = np.empty(other_spins.shape[0], dtype=np.uint32)
    col_indices = np.empty(other_spins.shape[0], dtype=np.uint32)
    elements = np.empty(other_spins.shape[0], dtype=np.float64)
    written = _lib.build_matrix(
        spins.shape[0],
        spins.ctypes.data_as(POINTER(c_uint64)),
        counts.ctypes.data_as(POINTER(c_int64)),
        ψs.ctypes.data_as(POINTER(c_double)),
        other_spins.ctypes.data_as(POINTER(c_uint64)),
        other_coeffs.ctypes.data_as(POINTER(c_double)),
        other_counts.ctypes.data_as(POINTER(c_int64)),
        other_ψs.ctypes.data_as(POINTER(c_double)),
        row_indices.ctypes.data_as(POINTER(c_uint32)),
        col_indices.ctypes.data_as(POINTER(c_uint32)),
        elements.ctypes.data_as(POINTER(c_double)),
        field.ctypes.data_as(POINTER(c_double)),
    )
    row_indices = row_indices[:written]
    col_indices = col_indices[:written]
    elements = elements[:written]

    matrix = scipy.sparse.coo_matrix(
        (elements, (row_indices, col_indices)),
        shape=(spins.shape[0], spins.shape[0]),
    )
    h = sa.Hamiltonian(matrix, field)

    print("Max field", field.max(), np.abs(field).max())
    print("Min field", field.min(), np.abs(field).min())
    print("Max coupling", matrix.data.max(), np.abs(matrix.data).max())
    print("Min coupling", matrix.data.min(), np.abs(matrix.data).min())

    x0 = np.empty((spins.shape[0] + 63) // 64, dtype=np.uint64)
    _lib.extract_signs(
        spins.shape[0],
        ψs.ctypes.data_as(POINTER(c_double)),
        x0.ctypes.data_as(POINTER(c_uint64)),
    )
    logger.info("Done! The Hamiltonian contains {} non-zero elements", written)
#    return h, spins, x0, matrix, field
    return h, spins, x0


# def extract_classical_ising_model(spins, hamiltonian, log_ψ, sampled: bool = False):
#     logger.info("Constructing classical Ising model...")
#     spins = np.asarray(spins, dtype=np.uint64, order="C")
#     if spins.ndim < 2:
#         spins = spins.reshape(-1, 1)
#     spins, counts = np.unique(spins, return_counts=True, axis=0)
#     if not sampled:
#         assert np.all(counts == 1)
#     ψs = log_ψ(spins).numpy().squeeze(axis=1)
#     spins = [_ls_bits512_to_int(σ) for σ in spins]
#     spins_set = set(spins)
#
#     other_spins, coeffs, part_lengths = batched_apply(spins, hamiltonian)
#     assert np.allclose(coeffs.imag, 0)
#     coeffs = coeffs.real
#     other_ψs = log_ψ(other_spins).numpy().squeeze(axis=1)
#     other_spins = [_ls_bits512_to_int(σ) for σ in other_spins]
#
#     scale = np.max(other_ψs.real)
#     other_ψs.real -= scale
#     other_ψs = np.exp(other_ψs)
#     ψs.real -= scale
#     ψs = np.exp(ψs)
#     assert np.allclose(ψs.imag, 0)
#
#     matrix = []
#     field = np.zeros(len(spins), dtype=np.float64)
#
#     offset = 0
#     x0 = np.zeros((len(spins) + 63) // 64, dtype=np.uint64)
#     for i, σ in enumerate(spins):
#         ψ = ψs[i]
#         if ψ > 0:
#             x0[i // 64] |= np.uint64(1) << np.uint64(i % 64)
#         for (other_σ, c, other_ψ) in zip(
#             other_spins[offset : offset + part_lengths[i]],
#             coeffs[offset : offset + part_lengths[i]],
#             other_ψs[offset : offset + part_lengths[i]],
#         ):
#             assert np.isclose(other_ψ.imag, 0)
#             if other_σ in spins_set:
#                 j = spins.index(other_σ)
#                 if sampled:
#                     matrix.append((i, j, c * counts[i] * abs(other_ψ.real) / abs(ψ.real)))
#                 else:
#                     matrix.append((i, j, c * abs(other_ψ.real) * abs(ψ.real)))
#             else:
#                 if sampled:
#                     field[i] += c * counts[i] * other_ψ.real / abs(ψ.real)
#                 else:
#                     field[i] += c * other_ψ.real * abs(ψ.real)
#         offset += part_lengths[i]
#     Jmax = max(map(lambda t: abs(t[2]), matrix))
#     matrix = list(filter(lambda t: abs(t[2]) > 1e-8 * Jmax, matrix))
#
#     logger.info("len(matrix) = {}", len(matrix))
#     matrix = scipy.sparse.coo_matrix(
#         ([d for (_, _, d) in matrix], ([i for (i, _, _) in matrix], [j for (_, j, _) in matrix])),
#         shape=(len(field), len(field)),
#     )
#     spins = np.stack([_int_to_ls_bits512(σ) for σ in spins])
#     h = sa.Hamiltonian(matrix, field)
#     logger.info("Done!")
#     return h, spins, x0
