import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import h5py
import ising_glass_annealer as sa
import lattice_symmetries as ls
import networkx
import numpy as np
import scipy.sparse
import yaml
from numba import njit, prange
from loguru import logger
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components


class AlmostInfiniteGraph:
    quantum_hamiltonian: ls.Operator
    ground_state: NDArray[np.float64]

    def __init__(self, quantum_hamiltonian, ground_state):
        self.quantum_hamiltonian = quantum_hamiltonian
        self.ground_state = ground_state

    def neighbours(self, spin: int):
        other_spins, other_coeffs = self.quantum_hamiltonian.apply(spin)
        if not np.allclose(other_coeffs.imag, 0, atol=1e-6):
            raise ValueError("expected all Hamiltonian matrix elements to be real")
        other_coeffs = other_coeffs.real
        if other_spins.ndim > 1:
            other_spins = other_spins[:, 0]

        basis = self.quantum_hamiltonian.basis
        psi = np.abs(self.ground_state[basis.index(spin)])
        other_psis = np.abs(self.ground_state[basis.batched_index(other_spins)])

        nodes = other_spins.tolist()
        edges = (psi * other_coeffs * other_psis).tolist()

        return sorted(zip(nodes, edges), reverse=True, key=lambda t: abs(t[1]))


@dataclass
class IsingModel:
    spins: NDArray[np.uint64]
    quantum_hamiltonian: ls.Operator
    ising_hamiltonian: sa.Hamiltonian
    initial_signs: NDArray[np.uint64]

    @property
    def size(self):
        return self.spins.shape[0]


def _normalize_spins(spins) -> NDArray[np.uint64]:
    spins = np.asarray(spins, dtype=np.uint64, order="C")
    if spins.ndim <= 1:
        spins = np.hstack([spins.reshape(-1, 1), np.zeros((spins.shape[0], 7), dtype=np.uint64)])
    elif spins.ndim == 2:
        if spins.shape[1] != 8:
            raise ValueError("'spins' has wrong shape: {}; expected (?, 8)".format(x.shape))
        spins = np.ascontiguousarray(spins)
    else:
        raise ValueError("'spins' has wrong shape: {}; expected a 2D array".format(x.shape))
    return spins


@njit
def _make_ising_model_compute_elements(
    belong_to_spins, psi, other_indices, other_coeffs, other_counts
):
    n = len(psi)
    other_psi = np.where(belong_to_spins, psi[other_indices], 0)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(other_counts)
    elements = other_coeffs * np.abs(other_psi)
    for i in range(n):
        elements[offsets[i] : offsets[i + 1]] *= np.abs(psi[i])
    return elements, offsets


def _batched_apply(hamiltonian, spins, chunk_size=10000):
    assert hamiltonian.basis.number_spins <= 64, "TODO: only works with up to 64 bits"
    n = spins.shape[0]
    out_spins = []
    out_coeffs = []
    out_counts = []

    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        x = _normalize_spins(spins[start:end])
        other_spins, other_coeffs, other_counts = hamiltonian.batched_apply(x)
        if not np.allclose(other_coeffs.imag, 0, atol=1e-6):
            raise ValueError("expected all Hamiltonian matrix elements to be real")
        other_coeffs = np.ascontiguousarray(other_coeffs.real)
        other_spins = np.ascontiguousarray(other_spins[:, 0])
        out_spins.append(other_spins)
        out_coeffs.append(other_coeffs)
        out_counts.append(other_counts)
        start = end

    return np.hstack(out_spins), np.hstack(out_coeffs), np.hstack(out_counts)


@njit
def invert_permutation(p):
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


@njit(parallel=True)
def _clipped_search_sorted(haystack, needles, chunk_size=5000):
    n = needles.size
    number_chunks = (n + chunk_size - 1) // chunk_size
    indices = np.zeros(n, dtype=np.int64)
    for i in prange(number_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        needles_chunk = needles[start:end]
        order = np.argsort(needles_chunk)
        index_chunk = np.searchsorted(haystack, needles_chunk[order])
        indices[start:end] = np.clip(index_chunk[invert_permutation(order)], 0, haystack.size - 1)
    return indices


def make_ising_model(
    spins: NDArray[np.uint64],
    quantum_hamiltonian: ls.Operator,
    log_psi: Optional[NDArray[np.float64]] = None,
    log_psi_fn: Optional[Callable[[NDArray[np.uint64]], NDArray[np.float64]]] = None,
    external_field: bool = False,
    debug: bool = False,
):
    start_time = time.time()
    if log_psi is None and log_psi_fn is None:
        raise ValueError("at least one of log_psi or log_psi_fn should be specified")
    if external_field and log_psi_fn is None:
        raise ValueError("log_psi_fn should be specified when external_field=True")

    spins = np.asarray(spins, dtype=np.uint64)
    spins, indices, counts = np.unique(spins, return_index=True, return_counts=True, axis=0)
    if np.any(counts != 1):
        logger.warning("'spins' were not unique, are you sure this is what you want?")
        spins = spins[indices]
        if log_psi is not None:
            log_psi = log_psi[indices]
    if log_psi is None:
        assert log_psi_fn is not None
        log_psi = log_psi_fn(spins)
    tick = time.time()
    n = spins.shape[0]
    other_spins, other_coeffs, other_counts = _batched_apply(quantum_hamiltonian, spins)
    # quantum_hamiltonian.batched_apply(spins)
    # if not np.allclose(other_coeffs.imag, 0, atol=1e-6):
    #     raise ValueError("expected all Hamiltonian matrix elements to be real")
    # other_coeffs = other_coeffs.real
    tock = time.time()
    logger.debug("batched_apply took {:.2f} seconds", tock - tick)

    tick = time.time()
    # assert quantum_hamiltonian.basis.number_spins <= 64, "TODO: only works with up to 64 bits"
    # spins = spins[:, 0]
    # other_spins = other_spins[:, 0]
    # other_indices1 = np.clip(np.searchsorted(spins, other_spins), 0, n - 1)
    other_indices = _clipped_search_sorted(spins, other_spins)
    # assert np.array_equal(_indices1, _indices2)
    # other_indices = _indices1
    belong_to_spins = other_spins == spins[other_indices]
    tock = time.time()
    logger.debug("searchsorted took {:.2f} seconds", tock - tick)

    psi = np.exp(log_psi, dtype=np.complex128)
    if not np.allclose(psi.imag, 0, atol=1e-6):
        raise ValueError("expected all wavefunction coefficients to be real")
    psi = np.ascontiguousarray(psi.real)
    psi /= np.linalg.norm(psi)

    elements, offsets = _make_ising_model_compute_elements(
        belong_to_spins, psi, other_indices, other_coeffs, other_counts
    )
    # other_psi = np.where(belong_to_spins, psi[other_indices], 0)
    # offsets = np.zeros(n + 1, dtype=np.int64)
    # offsets[1:] = np.cumsum(other_counts)
    # elements = other_coeffs * np.abs(other_psi)
    # for i in range(n):
    #     elements[offsets[i] : offsets[i + 1]] *= np.abs(psi[i])

    matrix = scipy.sparse.csr_matrix((elements, other_indices, offsets), shape=(n, n))
    matrix = 0.5 * (matrix + matrix.T)
    matrix.sort_indices()
    matrix = matrix.tocoo()

    field = np.zeros(n, dtype=np.float64)
    if external_field:
        assert False
        # external_other_spins = other_spins[not belong_to_spins]
        # external_other_log_coeff = log_coeff_fn(_normalize_spins(external_other_spins))

    ising_hamiltonian = sa.Hamiltonian(matrix, field)
    x0 = sa.signs_to_bits(np.sign(psi))
    end_time = time.time()
    logger.debug("Took {:.2} seconds", end_time - start_time)
    return IsingModel(spins, quantum_hamiltonian, ising_hamiltonian, x0)


def compute_accuracy_and_overlap(
    predicted: NDArray[np.uint64],
    exact: NDArray[np.uint64],
    weights: Optional[NDArray[np.float64]] = None,
    number_spins: Optional[int] = None,
) -> Tuple[float, float]:
    if weights is None and number_spins is None:
        raise ValueError("'weights' and 'number_spins' cannot be both None")
    if number_spins is None:
        number_spins = len(weights)
    if weights is None:
        weights = np.ones(number_spins, dtype=np.float64)

    predicted_signs = sa.bits_to_signs(predicted, number_spins)
    exact_signs = sa.bits_to_signs(exact, number_spins)
    accuracy = np.mean(exact_signs == predicted_signs)
    accuracy = max(accuracy, 1 - accuracy)
    overlap = abs(np.dot(exact_signs * predicted_signs, weights / np.sum(weights)))
    return accuracy, overlap


def solve_ising_model(
    model: IsingModel,
    mode: str = "sa",
    frozen_spins: Optional[NDArray[np.uint64]] = None,
    seed: int = 12345,
    number_sweeps: int = 5120,
    repetitions: int = 64,
    only_best: bool = True,
) -> NDArray[np.uint64]:
    if mode == "sa":
        x, _ = sa.anneal(
            model.ising_hamiltonian,
            seed=seed,
            number_sweeps=number_sweeps,
            repetitions=repetitions,
            only_best=only_best,
        )
    elif mode == "greedy":
        x, _ = sa.greedy_solve(model.ising_hamiltonian)
    else:
        raise ValueError(
            "invalid mode specified: '{}'; expected either 'sa' or 'greedy'".format(mode)
        )

    if frozen_spins is not None:
        frozen_indices = binary_search(model.spins, frozen_spins)
        frozen_signs = sa.bits_to_signs(x, count=model.spins.size)
        frozen_signs = frozen_signs[frozen_indices]
        x = sa.signs_to_bits(frozen_signs)
    return x


@dataclass
class SamplingResult:
    spins: NDArray[np.uint64]
    weights: Optional[NDArray[np.float64]]


def monte_carlo_sampling(
    states: NDArray[np.uint64],
    ground_state: NDArray[np.float64],
    number_samples: int,
    sampled_power: float = 2,
) -> SamplingResult:
    p = np.abs(ground_state) ** sampled_power
    p /= np.sum(p)
    indices = np.random.choice(len(states), size=number_samples, replace=True, p=p)
    return SamplingResult(spins=states[indices], weights=None)  # p[indices])


def determine_exact_solution(spins, quantum_hamiltonian, ground_state):
    indices = quantum_hamiltonian.basis.batched_index(spins)
    psi = ground_state[indices]
    return sa.signs_to_bits(np.sign(psi))


def compute_distribution_of_couplings(spins, quantum_hamiltonian, ground_state):
    infinite_graph = AlmostInfiniteGraph(quantum_hamiltonian, ground_state)
    histogram = np.zeros(1000, dtype=np.float64)
    for s in spins:
        couplings = np.array([c for _, c in infinite_graph.neighbours(s)])
        histogram[: couplings.size] += couplings
    histogram /= spins.size
    return histogram


# def strongest_coupling_greedy_color(
#     spins, quantum_hamiltonian, ground_state, frozen_spins, number_largest=1
# ):
#     log_psi_fn = ground_state_to_log_coeff_fn(ground_state, quantum_hamiltonian.basis)
#     ising = make_ising_model(spins, quantum_hamiltonian, log_psi_fn=log_psi_fn)
#     infinite_graph = AlmostInfiniteGraph(quantum_hamiltonian, ground_state)
#
#     matrix = ising.ising_hamiltonian.exchange.tocoo()
#     matrix.setdiag(np.zeros(spins.size))
#     matrix.eliminate_zeros()
#
#     number_components, _ = connected_components(matrix, directed=False)
#     assert number_components == 1
#
#     def ising_edges():
#         order = np.argsort(np.abs(matrix.data))[::-1]
#         for k in order:
#             s1 = matrix.row[k]
#             s2 = matrix.col[k]
#             if s1 < s2:
#                 c = float(matrix.data[k])
#                 yield (s1, s2, c)
#
#     @dataclass
#     class Cluster:
#         spins: Set[int]
#         signs: Dict[int, float]
#
#     csr_matrix = matrix.tocsr()
#
#     def merge_energy(cluster1, cluster2):
#         if len(cluster1.spins) > len(cluster2.spins):
#             return merge_energy(cluster2, cluster1)
#         energy = 0
#         for (i1, sign1) in zip(cluster1.spins, cluster1.signs):
#             for k in range(csr_matrix.indptr[i1], csr_matrix.indptr[i1 + 1]):
#                 i2 = csr_matrix.indices[k]
#                 if i2 in cluster2.spins:
#                     coupling = csr_matrix.data[k]
#                     sign2 = cluster2.signs[i2]
#                     energy += sign1 * sign2 * coupling
#         return energy
#
#     # signs = dict()
#     # np.zeros(spins.size, dtype=np.float64)
#     number_clusters = 0
#     # next_cluster_index = 0
#     # cluster_indices = dict()
#     clusters = dict()
#     # np.full(spins.size, -1, dtype=np.int32)
#
#     for (s1, s2, coupling) in ising_edges():
#         # Both spins already have colors
#         if (s1 in clusters) and (s2 in clusters):
#             cluster1 = clusters[s1]
#             cluster2 = clusters[s2]
#             if cluster1 == cluster2:
#                 # Spins belong to the same cluster. There is no reason
#                 # to flip spins because all previous couplings were stronger
#                 # than our current one
#                 pass
#             else:
#                 # Check whether we need to flip one of the clusters
#                 # should_flip = merge_energy(cluster1, cluster2) > 0
#                 is_frustrated = cluster1.signs[s1] * cluster2.signs[s2] * coupling > 0
#                 # if is_frustrated != should_flip:
#                 #     logger.debug("should_flip and is_frustrated do not agree")
#                 should_flip = is_frustrated
#                 keys = list(clusters.keys())
#                 for key in keys:
#                     if clusters[key] == cluster2:
#                         sign = clusters[key].signs[key]
#                         if should_flip:
#                             sign *= -1
#
#                         clusters[key] = cluster1
#                         cluster1.spins.add(key)
#                         cluster1.signs[key] = sign
#                 number_clusters -= 1
#         elif s1 in clusters:
#             cluster1 = clusters[s1]
#             cluster2 = Cluster({s2}, {s2: 1})
#             if merge_energy(cluster2, cluster1) > 0:
#                 cluster2.signs[s2] *= -1
#
#             clusters[s2] = cluster1
#             cluster1.spins.add(s2)
#             cluster1.signs[s2] = cluster2.signs[s2]
#
#         elif s2 in clusters:
#             cluster2 = clusters[s2]
#             cluster1 = Cluster({s1}, {s1: 1})
#             if merge_energy(cluster1, cluster2) > 0:
#                 cluster1.signs[s1] *= -1
#
#             clusters[s1] = cluster2
#             cluster2.spins.add(s1)
#             cluster2.signs[s1] = cluster1.signs[s1]
#
#         else:
#             # Neither of the spins has a color
#             sign = -int(np.sign(coupling))
#             cluster = Cluster({s1, s2}, {s1: 1, s2: sign})
#             clusters[s1] = cluster
#             clusters[s2] = cluster
#             number_clusters += 1
#
#     # print(number_clusters)
#     # print(cluster_indices)
#
#     assert number_clusters == 1
#     assert len(clusters) == spins.size
#
#     mega_cluster = next(iter(clusters.values()))
#     for cluster in clusters.values():
#         assert mega_cluster == cluster
#
#     signs = mega_cluster.signs
#
#     count = 0
#     while True:
#         changed = False
#         count += 1
#         for s1 in signs.keys():
#             e = 0
#             for k in range(csr_matrix.indptr[s1], csr_matrix.indptr[s1 + 1]):
#                 s2 = csr_matrix.indices[k]
#                 coupling = csr_matrix.data[k]
#                 e += signs[s2] * coupling
#             e *= signs[s1]
#             if e > 0:
#                 # logger.debug("{} is locally non-optimal", ising.spins[s1])
#                 changed = True
#                 signs[s1] *= -1
#         if not changed:
#             break
#     print(count)
#     # signs = np.array([2 * colors[s] - 1 for s in frozen_spins])
#     frozen_indices = binary_search(ising.spins, frozen_spins)
#     frozen_signs = np.array([signs[s] for s in frozen_indices])
#     return sa.signs_to_bits(frozen_signs)
def cluster_statistics(spins, quantum_hamiltonian, ground_state):
    log_psi_fn = ground_state_to_log_coeff_fn(ground_state, quantum_hamiltonian.basis)
    ising = make_ising_model(spins, quantum_hamiltonian, log_psi_fn=log_psi_fn)

    signs = sa.bits_to_signs(ising.initial_signs, spins.size)
    matrix = ising.ising_hamiltonian.exchange.tocoo()
    matrix.setdiag(np.zeros(spins.size))
    matrix.eliminate_zeros()
    number_bonds = matrix.nnz

    is_frustrated = [
        coupling * signs[i] * signs[j] > 0
        for (coupling, i, j) in zip(matrix.data, matrix.row, matrix.col)
    ]
    is_frustrated = np.asarray(is_frustrated, dtype=np.uint8)

    matrix = matrix.tocsr()
    positive_matrix = matrix.tocsr().copy()
    positive_matrix.data = np.abs(positive_matrix.data)
    largest_coupling_indices = matrix.argmax(axis=1).A.squeeze(axis=1)
    is_largest_frustrated = [
        matrix[i, largest_coupling_indices[i]] * signs[i] * signs[largest_coupling_indices[i]] > 0
        for i in range(spins.size)
    ]
    is_largest_frustrated = np.asarray(is_largest_frustrated, dtype=np.uint8)

    # is_largest_frustrated = np.zeros(spins.size, dtype=np.uint8)
    # is_largest_within = np.zeros(spins.size, dtype=np.uint8)
    # for k in range(spins.size):
    #     spin = spins[k]
    #     neighbours = graph.neighbours(spin)
    #     for (n, c) in neighbo

    logger.info(
        "Stats: spins={}, bonds={}, frustrated={}, largest_frustrated={}",
        spins.size,
        matrix.nnz,
        np.mean(is_frustrated),
        np.mean(is_largest_frustrated),
    )


def create_small_cluster_around_point(
    s0: int,
    hamiltonian: ls.Operator,
    required_size: int = 20,
    keep_probability: float = 0.5,
) -> List[int]:
    assert hamiltonian.basis.number_spins <= 64
    s0 = int(s0)
    spins = {s0}

    def children_of(s):
        xs, _ = hamiltonian.apply(s)
        if xs.ndim > 1:
            xs = xs[:, 0]
        children = []
        for x in xs:
            if x in spins:
                continue
            if np.random.rand() <= keep_probability:
                children.append(int(x))
        return children

    children = children_of(s0)
    while len(spins) < required_size and len(children) > 0:
        new_children = set()
        for child in children:
            spins.add(child)
            if len(spins) >= required_size:
                break
            new_children |= set(children_of(child))
        children = new_children

    return sorted(list(spins))


def make_hamiltonian_extension(
    model: IsingModel,
    log_psi_fn: Callable[[NDArray[np.uint64]], NDArray[np.float64]],
) -> IsingModel:
    spins, _, _ = _batched_apply(model.quantum_hamiltonian, model.spins)
    spins = np.unique(spins, axis=0)
    return make_ising_model(spins, model.quantum_hamiltonian, log_psi_fn=log_psi_fn)


@njit
def _get_strongest_off_diag_impl(data, indices, indptr):
    n = indptr.size - 1
    out = np.zeros(n, dtype=data.dtype)
    for i in range(n):
        max_c = 0
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if i != j:
                max_c = max(max_c, abs(data[k]))
        out[i] = max_c
    return out


def get_strongest_off_diag(matrix: scipy.sparse.spmatrix) -> NDArray[Any]:
    matrix = matrix.tocsr()
    return _get_strongest_off_diag_impl(matrix.data, matrix.indices, matrix.indptr)


def binary_search(haystack, needles):
    assert np.all(np.sort(haystack) == haystack)
    indices = np.searchsorted(haystack, needles)
    assert np.all(haystack[indices] == needles)
    return indices


# def sparsify_using_local_cutoff(
#     model: IsingModel, cutoff: int, frozen_spins: NDArray[np.uint64]
# ) -> IsingModel:
#     frozen_indices = binary_search(model.spins, frozen_spins)
#     is_spin_frozen = np.zeros(model.spins.shape, dtype=np.uint8)
#     is_spin_frozen[frozen_indices] = 1
#
#     matrix = model.ising_hamiltonian.exchange.tocsr().copy()
#     original_nnz = matrix.nnz
#     original_spins = model.size
#     # print(matrix.nnz, np.abs(matrix.data).min(), np.abs(matrix.data).max())
#     # strongest = get_strongest_off_diag(matrix)
#     # print(strongest)
#
#     tick = time.time()
#     for i in range(matrix.shape[0]):
#         start = matrix.indptr[i]
#         end = matrix.indptr[i + 1]
#         # js = matrix.indices[start:end]
#         cs = np.abs(matrix.data[start:end])
#         order = np.argsort(cs)[::-1]
#         for k in start + order[cutoff:]:
#             j = matrix.indices[k]
#             if is_spin_frozen[i] and is_spin_frozen[j]:
#                 continue
#             matrix.data[k] = 0
#     tock = time.time()
#     logger.debug("Python for loop took {:.2f}", tock - tick)
#
#     # matrix.data[start + order[cutoff:]] = 0
#     # if i == 0:
#     #     print(matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]])
#     # c = cutoff * strongest[i]
#     # for k in range(matrix.indptr[i], matrix.indptr[i + 1]):
#     #     j = matrix.indices[k]
#     #     coupling = abs(matrix.data[k])
#     #     if coupling < c:
#     #         if is_spin_frozen[i] and is_spin_frozen[j]:
#     #             pass
#     #         else:
#     #             matrix.data[k] = 0
#     matrix.eliminate_zeros()
#     matrix = 0.5 * (matrix + matrix.transpose())
#
#     tick = time.time()
#     _, component_indices = connected_components(matrix, directed=False)
#     tock = time.time()
#     logger.debug("connected_components took {:.2f}", tock - tick)
#
#     magic_component_index = component_indices[frozen_indices[0]]
#     # print(component_indices[frozen_spin_indices])
#     assert np.all(component_indices[frozen_indices] == magic_component_index)
#     mask = component_indices == magic_component_index
#     print(np.sum(mask))
#
#     spins = model.spins[mask]
#     initial_signs = sa.signs_to_bits(sa.bits_to_signs(model.initial_signs, model.size)[mask])
#
#     tick = time.time()
#     matrix = model.ising_hamiltonian.exchange[mask][:, mask]
#     tock = time.time()
#     logger.debug("matrix slicing took {:.2f}", tock - tick)
#     field = model.ising_hamiltonian.field[mask]
#
#     new_nnz = matrix.nnz
#     new_spins = spins.size
#     logger.debug(
#         "number of spins: {} -> {}; number of connections: {} -> {}",
#         original_spins,
#         new_spins,
#         original_nnz,
#         new_nnz,
#     )
#
#     new_model = IsingModel(
#         spins,
#         model.quantum_hamiltonian,
#         sa.Hamiltonian(matrix, field),
#         initial_signs,
#     )
#     return new_model


@njit
def _sparsify_using_global_cutoff(data, indices, indptr, is_spin_frozen, reltol):
    max_coupling = np.max(np.abs(data))
    n = indptr.size - 1
    for i in range(n):
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if is_spin_frozen[i] and is_spin_frozen[j]:
                continue
            if abs(data[k]) < reltol * max_coupling:
                data[k] = 0


def sparsify_using_global_cutoff(
    model: IsingModel, reltol: float, frozen_spins: NDArray[np.uint64]
) -> IsingModel:
    frozen_indices = binary_search(model.spins, frozen_spins)
    is_spin_frozen = np.zeros(model.spins.shape, dtype=np.uint8)
    is_spin_frozen[frozen_indices] = 1

    matrix = model.ising_hamiltonian.exchange.tocsr()
    original_nnz = matrix.nnz
    original_spins = model.size

    data = matrix.data.copy()
    _sparsify_using_global_cutoff(data, matrix.indices, matrix.indptr, is_spin_frozen, reltol)

    matrix = scipy.sparse.csr_matrix((data, matrix.indices, matrix.indptr), shape=matrix.shape)
    matrix = 0.5 * (matrix + matrix.transpose())
    matrix.eliminate_zeros()

    _, component_indices = connected_components(matrix, directed=False)
    magic_component_index = component_indices[frozen_indices[0]]
    assert np.all(component_indices[frozen_indices] == magic_component_index)
    is_magic_component = component_indices == magic_component_index

    spins = model.spins[is_magic_component]
    initial_signs = sa.bits_to_signs(model.initial_signs, model.size)[is_magic_component]
    initial_signs = sa.signs_to_bits(initial_signs)

    matrix = model.ising_hamiltonian.exchange[is_magic_component][:, is_magic_component]
    field = model.ising_hamiltonian.field[is_magic_component]

    new_nnz = matrix.nnz
    new_model = IsingModel(
        spins,
        model.quantum_hamiltonian,
        sa.Hamiltonian(matrix, field),
        initial_signs,
    )

    logger.info(
        "number of spins: {} -> {}; number of connections: {} -> {}",
        original_spins,
        new_model.size,
        original_nnz,
        new_nnz,
    )
    return new_model


# def optimize_signs_on_cluster(
#     cluster: SamplingResult,
#     quantum_hamiltonian: ls.Operator,
#     log_psi_fn: Callable[[NDArray[np.uint64]], NDArray[np.float64]],
#     extension_order: int,
#     cutoff: float,
#     number_sweeps: int = 5192,
#     repetitions: int = 64,
# ):
#     h = make_ising_model(cluster.spins, quantum_hamiltonian, log_psi_fn=log_psi_fn)
#     x = solve_ising_model(h, seed=None, number_sweeps=number_sweeps, repetitions=repetitions)
#     results = [x]
#     logger.debug("Starting with a cluster of {} spins", h.size)
#
#     h_coo = h.ising_hamiltonian.exchange.tocoo()
#     frozen_spins = cluster.spins
#
#     for i in range(extension_order):
#         h = make_hamiltonian_extension(h, log_psi_fn)
#
#         # graph = networkx.convert_matrix.from_scipy_sparse_matrix(h.ising_hamiltonian.exchange)
#         # print(list(graph.edges())[:100])
#         # print(networkx.is_weighted(graph))
#         # weights = networkx.get_edge_attributes(graph, "weight")
#         # print(list(weights.items())[:100])
#
#         logger.debug("Extension №{}: there are now {} spins in the cluster", i + 1, h.size)
#         h = sparsify_based_on_cutoff(h, cutoff=cutoff, frozen_spins=frozen_spins)
#         logger.debug("After sparsifying: {} spins in the cluster", h.size)
#         number_components, component_indices = connected_components(
#             h.ising_hamiltonian.exchange, directed=False
#         )
#         component_sizes = [np.sum(component_indices == k) for k in range(number_components)]
#         # print(sorted(component_sizes))
#         assert number_components == 1
#
#         # with h5py.File("test_cluster_{}.h5".format(i), "w") as f:
#         #     f["data"] = h.ising_hamiltonian.exchange.data
#         #     f["indices"] = h.ising_hamiltonian.exchange.indices
#         #     f["indptr"] = h.ising_hamiltonian.exchange.indptr
#         #     f["field"] = h.ising_hamiltonian.field
#
#         x = solve_ising_model(h, seed=None, number_sweeps=number_sweeps, repetitions=repetitions)
#         signs = extract_signs_from_bits(x, number_spins=h.size)
#
#         indices = np.searchsorted(h.spins, cluster.spins)
#         print(indices.shape)
#         assert np.all(h.spins[indices] == cluster.spins)
#         print(signs.shape)
#         part = extract_bits_from_signs(signs[indices])
#         results.append(part)
#
#     return results


def dump_ising_model_to_hdf5(model: IsingModel, ground_state: NDArray[np.float64], filename: str):
    matrix = model.ising_hamiltonian.exchange
    # offset = np.sum(matrix.diagonal())
    # matrix.setdiag(np.zeros(model.size))
    matrix = matrix.tocsr()
    field = model.ising_hamiltonian.field

    x = np.sign(ground_state)
    print(np.dot(x, matrix @ x))
    energy = model.quantum_hamiltonian.expectation(ground_state).real
    print(energy)

    with h5py.File(filename, "w") as out:
        out["elements"] = np.asarray(matrix.data, dtype=np.float64)
        out["indices"] = np.asarray(matrix.indices, dtype=np.int32)
        out["indptr"] = np.asarray(matrix.indptr, dtype=np.int32)
        out["field"] = np.asarray(field, dtype=np.float64)
        out["energy"] = energy
        # out["offset"] = offset
        out["signs"] = sa.signs_to_bits(np.sign(ground_state))


def load_ground_state(filename: str) -> Tuple[NDArray[np.float64], float, NDArray[np.uint64]]:
    with h5py.File(filename, "r") as f:
        ground_state = np.asarray(f["/hamiltonian/eigenvectors"], dtype=np.float64)
        ground_state = ground_state.squeeze()
        if ground_state.ndim > 1:
            ground_state = ground_state[0, :]
        energy = float(f["/hamiltonian/eigenvalues"][0])
        basis_representatives = np.asarray(f["/basis/representatives"], dtype=np.uint64)
    return ground_state, energy, basis_representatives


def load_hamiltonian(filename: str) -> ls.Operator:
    with open(filename, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    basis = ls.SpinBasis.load_from_yaml(config["basis"])
    hamiltonian = ls.Operator.load_from_yaml(config["hamiltonian"], basis)
    return hamiltonian


def load_input_files(args):
    yaml_filename = args.yaml
    if args.hdf5 is not None:
        hdf5_filename = args.hdf5
    else:
        hdf5_filename = yaml_filename.replace(".yaml", ".h5")

    logger.info("Loading the ground state ...")
    hamiltonian = load_hamiltonian(yaml_filename)
    ground_state, energy, _representatives = load_ground_state(hdf5_filename)
    hamiltonian.basis.build(_representatives)
    logger.info("Ground state energy is {}", energy)
    return hamiltonian, ground_state


def ground_state_to_log_coeff_fn(ground_state: np.ndarray, basis: ls.SpinBasis):
    ground_state = np.asarray(ground_state, dtype=np.float64, order="C")
    assert ground_state.ndim == 1

    log_amplitudes = np.log(np.abs(ground_state))
    phases = np.where(ground_state >= 0, 0, np.pi)

    def log_coeff_fn(spins: np.ndarray) -> np.ndarray:
        spins = np.asarray(spins, dtype=np.uint64, order="C")
        if spins.ndim > 1:
            spins = spins[:, 0]
        indices = ls.batched_index(basis, spins)
        a = log_amplitudes[indices]
        b = phases[indices]
        return a + 1j * b

    return log_coeff_fn


def add_noise_to_amplitudes(ground_state: NDArray[np.float64], eps: float):
    ground_state = np.asarray(ground_state, dtype=np.float64, order="C")
    assert ground_state.ndim == 1

    log_amplitudes = np.log(np.abs(ground_state))
    signs = np.sign(ground_state)

    noise = eps * 2 * (np.random.rand(log_amplitudes.size) - 0.5)
    noisy_ground_state = signs * np.exp(log_amplitudes + noise)
    noisy_ground_state /= np.linalg.norm(noisy_ground_state)
    return noisy_ground_state


def check_greedy_algorithm_quality():
    parser = argparse.ArgumentParser(
        description="Quality of the greedy optimization algorithm on small systems."
    )
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--hdf5", type=str)
    args = parser.parse_args()

    hamiltonian, ground_state = load_input_files(args)
    basis = hamiltonian.basis
    assert np.isclose(np.linalg.norm(ground_state), 1)
    exact_signs = sa.signs_to_bits(np.sign(ground_state))
    weights = ground_state**2

    log_coeff_fn = ground_state_to_log_coeff_fn(ground_state, basis)
    h = make_ising_model(basis.states, hamiltonian, log_psi_fn=log_coeff_fn)
    x = solve_ising_model(h, mode="greedy")
    sign_accuracy, sign_overlap = compute_accuracy_and_overlap(x, exact_signs, weights)
    print("{},{}".format(sign_accuracy, sign_overlap))


def analyze_influence_of_noise():
    parser = argparse.ArgumentParser(
        description="Influence of noise on greedy optimization (small systems)."
    )
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--hdf5", type=str)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-noise", type=float, default=1e-2)
    parser.add_argument("--max-noise", type=float, default=1e2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    if os.path.exists(args.output):
        logger.error(
            "Output file '{}' already exists: refusing to overwrite; "
            "delete it manually if this is what you really want",
            args.output,
        )
        return
    np.random.seed(args.seed)
    hamiltonian, ground_state = load_input_files(args)
    basis = hamiltonian.basis
    assert np.isclose(np.linalg.norm(ground_state), 1)
    exact_signs = sa.signs_to_bits(np.sign(ground_state))
    weights = ground_state**2

    noise_levels = np.linspace(np.log(args.min_noise), np.log(args.max_noise), args.steps)
    noise_levels = np.exp(noise_levels)

    for i, eps in enumerate(noise_levels):
        logger.info("[{}/{}] Testing with ɛ = {} ...", i + 1, args.steps, eps)

        # results = np.zeros((args.repetitions, 2), dtype=np.float64)
        with open(args.output, "a") as f:
            for k in range(args.repetitions):
                noisy_ground_state = add_noise_to_amplitudes(ground_state, eps=eps)
                assert np.isclose(np.linalg.norm(noisy_ground_state), 1)
                noisy_log_coeff_fn = ground_state_to_log_coeff_fn(noisy_ground_state, basis)
                amplitude_overlap = np.dot(np.abs(noisy_ground_state), np.abs(ground_state))
                h = make_ising_model(basis.states, hamiltonian, log_psi_fn=noisy_log_coeff_fn)
                x = solve_ising_model(h, mode="greedy")
                _, sign_overlap = compute_accuracy_and_overlap(x, exact_signs, weights)
                f.write("{},{},{}\n".format(eps, amplitude_overlap, sign_overlap))


def postprocess_influence_of_noise(csv_file: str):
    table = np.loadtxt(csv_file, delimiter=",")
    edges = np.linspace(0, 1, 101)
    x = 0.5 * (edges[1:] + edges[:-1])
    median = np.zeros(len(x), dtype=np.float64)
    upper = np.zeros(len(x), dtype=np.float64)
    lower = np.zeros(len(x), dtype=np.float64)
    amplitude_overlap = table[:, 1]
    sign_overlap = table[:, 2]
    for i in range(len(x)):
        mask = (edges[i] < amplitude_overlap) & (amplitude_overlap <= edges[i + 1])
        ys = sign_overlap[mask]
        if len(ys) > 0:
            assert np.all(ys <= 1)
            # mean[i] = np.mean(ys)
            m = np.percentile(ys, [25, 50, 75])
            lower[i], median[i], upper[i] = m
            # if mean[i] + std[i] > 1.0:
            #     counts, hist_edges = np.histogram(ys)
            #     np.savetxt(
            #         "error.dat", np.vstack([0.5 * (hist_edges[1:] + hist_edges[:-1]), counts]).T
            #     )
            #     assert False
        else:
            median[i] = float("nan")
            upper[i] = float("nan")
            lower[i] = float("nan")

    name = csv_file.replace(".csv", "_stats.csv")
    with open(name, "w") as f:
        f.write("amplitude_overlap,median,upper,lower\n")
        np.savetxt(f, np.vstack([x, median, upper, lower]).T, delimiter=",")


def analyze_coupling_distribution():
    parser = argparse.ArgumentParser(description="How are couplings distributed?")
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--hdf5", type=str)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    hamiltonian, ground_state = load_input_files(args)
    basis = hamiltonian.basis
    assert np.isclose(np.linalg.norm(ground_state), 1)
    max_coeff = np.max(np.abs(ground_state))
    logger.info("Max coeff: {}; max log coeff: {}", max_coeff, np.log(max_coeff))
    log_coeff_fn = ground_state_to_log_coeff_fn(ground_state, basis)
    model = make_ising_model(basis.states, hamiltonian, log_psi_fn=log_coeff_fn)

    matrix = model.ising_hamiltonian.exchange.tocoo()
    matrix.setdiag(0)
    matrix.eliminate_zeros()

    couplings = np.sort(np.abs(matrix.data))[::-1]
    np.savetxt(args.output, couplings)


def analyze_probability_of_frustration():
    parser = argparse.ArgumentParser(description="How often are couplings frustrated?")
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--hdf5", type=str)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    hamiltonian, ground_state = load_input_files(args)
    basis = hamiltonian.basis
    assert np.isclose(np.linalg.norm(ground_state), 1)
    log_coeff_fn = ground_state_to_log_coeff_fn(ground_state, basis)
    model = make_ising_model(basis.states, hamiltonian, log_psi_fn=log_coeff_fn)
    signs = sa.bits_to_signs(model.initial_signs, model.size)

    matrix = model.ising_hamiltonian.exchange.tocoo()
    matrix.setdiag(0)
    matrix.eliminate_zeros()

    is_frustrated = signs[matrix.row] * signs[matrix.col] * matrix.data > 0

    max_coupling = np.log(np.max(np.abs(matrix.data)))
    min_coupling = max(max_coupling - 20, np.log(np.min(np.abs(matrix.data))))
    logger.debug("min log coupling: {}; max log coupling: {}", min_coupling, max_coupling)

    frustrated = np.log(np.abs(matrix.data[is_frustrated]))
    frustrated = frustrated[(min_coupling <= frustrated) & (frustrated <= max_coupling)]
    normal = np.log(np.abs(matrix.data[np.invert(is_frustrated)]))
    normal = normal[(min_coupling <= normal) & (normal <= max_coupling)]

    bins = np.linspace(min_coupling, max_coupling, 50)

    frustrated_pdf, _ = np.histogram(frustrated, bins=bins, density=False)
    normal_pdf, _ = np.histogram(normal, bins=bins, density=False)
    # frustrated_pdf = frustrated.size * scipy.stats.gaussian_kde(frustrated, bw_method=bw_method)(x)
    # normal_pdf = normal.size * scipy.stats.gaussian_kde(normal, bw_method=bw_method)(x)
    y = normal_pdf / (normal_pdf + frustrated_pdf)
    y[normal_pdf + frustrated_pdf < 100] = float("nan")

    x = np.exp(0.5 * (bins[:-1] + bins[1:]))
    np.savetxt(args.output, np.vstack([x, y]).T, delimiter=",")


def analyze_smallest_amplitude_overlap():
    parser = argparse.ArgumentParser(description="How small can the amplitude overlap get?")
    parser.add_argument("--hdf5", type=str)
    parser.add_argument("--trials", default=100, type=int)
    parser.add_argument("--seed", default=12345, type=int)
    args = parser.parse_args()
    np.random.seed(args.seed)

    ground_state, _, _ = load_ground_state(args.hdf5)
    ground_state = np.abs(ground_state)
    assert np.isclose(np.linalg.norm(ground_state), 1)
    logger.info("max amplitude: {}", np.max(ground_state))
   
    overlaps = np.zeros(args.trials)
    for i in range(args.trials):
        noise = np.random.rand(len(ground_state))
        overlaps[i] = abs(np.dot(ground_state, noise)) / np.linalg.norm(noise)

    m = np.percentile(overlaps, [25, 50, 75])
    logger.info("mean: {}, median: {}, interquartile: {}", np.mean(overlaps), m[1], m[2] - m[0])

