import argparse
import collections
from typing import List, Optional, Tuple

import ising_glass_annealer as sa
import networkx
import numpy as np
import scipy.sparse
from loguru import logger
from scipy.sparse.csgraph import connected_components

from annealing_sign_problem import *


# def slice_coo_matrix(
#     matrix: scipy.sparse.coo_matrix, indices: np.ndarray
# ) -> scipy.sparse.coo_matrix:
#     indices = np.asarray(indices)
#     assert np.all(indices == np.sort(indices))
#
#     row_indices = np.searchsorted(indices, matrix.row)
#     row_indices[row_indices >= len(indices)] = len(indices) - 1
#     row_mask = indices[row_indices] == matrix.row
#     col_indices = np.searchsorted(indices, matrix.col)
#     col_indices[col_indices >= len(indices)] = len(indices) - 1
#     col_mask = indices[col_indices] == matrix.col
#     mask = row_mask & col_mask
#
#     new_row = row_indices[mask]
#     new_col = col_indices[mask]
#     new_data = matrix.data[mask]
#
#     return scipy.sparse.coo_matrix((new_data, (new_row, new_col)))


# def extract_local_hamiltonian(
#     mask: np.ndarray,
#     hamiltonian: sa.Hamiltonian,
#     spins: Tensor,
#     scale_field: float = 1,
# ) -> Tuple[sa.Hamiltonian, Tensor]:
#     (spin_indices,) = np.nonzero(mask)
#     spins = spins[spin_indices]
#     field = hamiltonian.field[spin_indices]
#     field *= scale_field
#     exchange = slice_coo_matrix(hamiltonian.exchange, spin_indices)
#     return sa.Hamiltonian(exchange, field), spins


# def is_frustrated(matrix: scipy.sparse.coo_matrix) -> bool:
#     def extract(mask):
#         return scipy.sparse.coo_matrix(
#             (matrix.data[mask], (matrix.row[mask], matrix.col[mask])), shape=matrix.shape
#         )
#
#     off_diagonal = matrix.row != matrix.col
#     matrix = extract(off_diagonal)
#
#     graph = networkx.convert_matrix.from_scipy_sparse_matrix(matrix)
#     assert networkx.is_connected(graph)
#     positive_graph = networkx.convert_matrix.from_scipy_sparse_matrix(extract(matrix.data > 0))
#     positive_coloring = networkx.coloring.greedy_color(
#         positive_graph, strategy="connected_sequential"
#     )
#     number_positive_colors = max(positive_coloring.values()) + 1
#     if number_positive_colors > 2:
#         logger.debug('"J > 0"-subgraph introduces frustration')
#         assert not networkx.algorithms.bipartite.is_bipartite(positive_graph)
#         return True
#     else:
#         logger.debug('"J > 0"-subgraph introduces no frustration')
#
#     # assert networkx.is_connected(positive_graph)
#     negative_graph = networkx.convert_matrix.from_scipy_sparse_matrix(extract(matrix.data < 0))
#     negative_coloring = networkx.coloring.greedy_color(
#         negative_graph, strategy="connected_sequential"
#     )
#     number_negative_colors = max(negative_coloring.values()) + 1
#     if number_negative_colors > 2:
#         logger.debug('"J < 0"-subgraph introduces frustration')
#         assert not networkx.algorithms.bipartite.is_bipartite(negative_graph)
#         return True
#     else:
#         logger.debug('"J < 0"-subgraph introduces no frustration')
#
#     if number_positive_colors < 2:
#         assert np.sum(matrix.data > 0) == 0
#         logger.debug("There are no positive couplings")
#         return False
#     if number_negative_colors < 2:
#         assert np.sum(matrix.data < 0) == 0
#         logger.debug("There are no negative couplings")
#         return False
#     return positive_coloring != negative_coloring


# def optimize_small_cluster(
#     spins: np.ndarray,
#     hamiltonian: ls.Operator,
#     log_coeff_fn: Callable[[np.ndarray], np.ndarray],
#     max_spins: int,
#     cutoff: float,
# ):
#     if spins.ndim > 1:
#         spins = spins[:, 0]
#     spins0 = spins
#     log_coeff0 = log_coeff_fn(spins0)
#     absolute_log_cutoff = np.log(cutoff) + np.max(log_coeff0.real)
#     logger.debug("Starting with a cluster of {} spins", spins.shape[0])
#     for tries in range(5):
#         extended_spins, extended_coeffs, extended_counts = hamiltonian.batched_apply(spins)
#         extended_spins = extended_spins[:, 0]
#         assert len(extended_counts) == spins.shape[0]
#         log_coeff = log_coeff_fn(spins)
#         extended_log_coeff = log_coeff_fn(extended_spins)
#
#         mask = np.empty(len(extended_coeffs), dtype=bool)
#         offset = 0
#         for i in range(len(extended_counts)):
#             log_couplings = (
#                 log_coeff[i].real
#                 + np.log(np.abs(extended_coeffs[offset : offset + extended_counts[i]]))
#                 + extended_log_coeff[offset : offset + extended_counts[i]].real
#             )
#             mask[offset : offset + extended_counts[i]] = log_couplings >= absolute_log_cutoff
#             offset += extended_counts[i]
#         logger.debug("{:.1f}% included", 100 * np.sum(mask) / len(mask))
#         extended_spins = extended_spins[mask]
#
#         spins = np.hstack((spins, extended_spins))
#         spins = np.unique(spins, axis=0)
#         logger.debug("Extended to {} spins", spins.shape[0])
#         if spins.shape[0] >= max_spins:
#             break
#
#     h, spins, x0, counts = extract_classical_ising_model(
#         spins,
#         hamiltonian,
#         log_coeff_fn,
#         # transformed_log_coeff_fn,
#         monte_carlo_weights=None,
#         scale_field=0,
#     )
#     spins = spins[:, 0]
#     number_components, component_labels = connected_components(h.exchange, directed=False)
#     assert number_components == 1
#
#     signs, e = sa.anneal(
#         h,
#         seed=None,
#         number_sweeps=10000,
#         repetitions=64,
#         only_best=True,
#     )
#     signs = extract_signs_from_bits(signs, number_spins=spins.shape[0])
#     local_indices = np.searchsorted(spins, spins0)
#     assert np.all(spins[local_indices] == spins0)
#
#     return spins0, signs[local_indices]


# def build_and_analyze_clusters(
#     spins: np.ndarray,
#     weights: Optional[np.ndarray],
#     ground_state: np.ndarray,
#     hamiltonian: ls.Operator,
#     log_coeff_fn: Callable[[np.ndarray], np.ndarray],
#     cutoff: float = 0,
# ):
#     if weights is None:
#         spins = np.unique(spins, axis=0)
#
#     def transformed_log_coeff_fn(x):
#         r = log_coeff_fn(x)
#         r.imag = 0
#         return r
#
#     h, spins, x0, counts = extract_classical_ising_model(
#         spins,
#         hamiltonian,
#         log_coeff_fn,
#         # transformed_log_coeff_fn,
#         monte_carlo_weights=weights,
#         scale_field=1,
#         cutoff=cutoff,
#     )
#     number_components, component_labels = connected_components(h.exchange, directed=False)
#     component_sizes = np.asarray([np.sum(component_labels == i) for i in range(number_components)])
#
#     results = []
#     count = 0
#     for i in range(number_components):
#         if component_sizes[i] < 10:
#             continue
#         count += 1
#         # if count > 2:
#         #     break
#         _, local_spins = extract_local_hamiltonian(component_labels == i, h, spins, scale_field=0)
#         local_spins, signs = optimize_small_cluster(
#             local_spins, hamiltonian, log_coeff_fn, 1000, cutoff=1e-3
#         )
#
#         # if component_sizes[i] < 10:
#         #     continue
#         # logger.debug("Processing cluster with {} elements...", component_sizes[i])
#         # local_hamiltonian_with_fields, _ = extract_local_hamiltonian(
#         #     component_labels == i, h, spins, scale_field=1
#         # )
#         # local_hamiltonian, local_spins = extract_local_hamiltonian(
#         #     component_labels == i, h, spins, scale_field=0
#         # )
#         # signs, e = sa.anneal(
#         #     local_hamiltonian,
#         #     seed=None,
#         #     number_sweeps=5000,
#         #     repetitions=32,
#         #     only_best=True,
#         # )
#         # signs = extract_signs_from_bits(signs, number_spins=local_spins.shape[0])
#         local_indices = hamiltonian.basis.batched_index(local_spins[:, 0])
#
#         true_signs = 2 * (ground_state[local_indices] >= 0) - 1
#         accuracy = np.sum(signs == true_signs) / local_spins.shape[0]
#         if accuracy < 0.5:
#             signs = -signs
#             accuracy = 1 - accuracy
#
#         v = ground_state[local_indices]
#         v /= np.linalg.norm(v)
#         overlap = abs(np.dot(v, np.abs(v) * signs))
#         results.append(
#             ComponentStats(size=local_spins.shape[0], accuracy=accuracy, overlap=overlap)
#         )
#         # results.append(ComponentStats(size=component_sizes[i], accuracy=accuracy, overlap=overlap))
#
#         # table = []
#         # for k in range(local_spins.shape[0]):
#         #     is_correct = signs[k] == true_signs[k]
#         #     field = abs(local_hamiltonian_with_fields.field[k])
#         #     mask = local_hamiltonian_with_fields.exchange.row == k
#         #     coupling = np.sum(np.abs(local_hamiltonian_with_fields.exchange.data[mask]))
#         #     table.append((int(is_correct), field, coupling))
#         # with open("correlation.dat", "a") as f:
#         #     for t in table:
#         #         f.write("{}\t{}\t{}\n".format(*t))
#
#     return results


# def compute_local_energy_one(
#     s0: np.ndarray,
#     # weights: Optional[np.ndarray],
#     # ground_state: np.ndarray,
#     hamiltonian: ls.Operator,
#     log_coeff_fn: Callable[[np.ndarray], np.ndarray],
# ):
#     if spins.ndim > 1:
#         spins = spins[:, 0]
#
#     initial_spins = spins
#
#     spins, _, _ = hamiltonian.batched_apply(spins)
#     spins = np.unique(spins, axis=0)
#     spins, signs = optimize_small_cluster(spins, hamiltonian, log_coeff_fn, 10000, cutoff=1e-3)
#     order = np.argsort(spins)
#     spins = spins[order]
#     signs = signs[order]
#
#     local_energies = []
#     for batch_index in range(initial_spins.shape[0]):
#         neighborhood_spins, neighborhood_coeffs = hamiltonian.apply(initial_spins[batch_index])
#         neighborhood_spins = neighborhood_spins[:, 0]
#         neighborhood_log_coeff = log_coeff_fn(neighborhood_spins).real
#         current_log_coeff = log_coeff_fn(initial_spins[batch_index : batch_index + 1]).real
#
#         neighborhood_indices = np.searchsorted(spins, neighborhood_spins)
#         assert np.all(spins[neighborhood_indices] == neighborhood_spins)
#         neighborhood_signs = signs[neighborhood_indices]
#
#         current_index = np.searchsorted(spins, initial_spins[batch_index])
#         assert spins[current_index] == initial_spins[batch_index]
#         current_sign = signs[current_index]
#
#         local_energy = np.sum(
#             neighborhood_coeffs
#             * np.exp(neighborhood_log_coeff - current_log_coeff)
#             * neighborhood_signs
#             / current_sign
#         )
#         local_energies.append(local_energy)
#     return np.asarray(local_energies)


def compute_local_energy_impl(
    s0: int,
    spins: np.ndarray,
    signs: np.ndarray,
    hamiltonian: ls.Operator,
    log_coeff_fn: Callable[[np.ndarray], np.ndarray],
):
    spins = np.asarray(spins, dtype=np.uint64)
    signs = np.asarray(signs)

    _initial_spin = np.zeros((1, 8), dtype=np.uint64)
    _initial_spin[0, 0] = s0
    log_coeff0 = log_coeff_fn(_initial_spin).real

    other_spins, other_coeffs = hamiltonian.apply(s0)
    other_log_coeff = log_coeff_fn(other_spins).real
    if other_spins.ndim > 1:
        other_spins = other_spins[:, 0]

    other_indices = np.searchsorted(spins, other_spins)
    assert np.all(spins[other_indices] == other_spins)
    other_signs = signs[other_indices]

    index0 = np.searchsorted(spins, s0)
    sign0 = signs[index0]

    e = np.sum(other_coeffs * np.exp(other_log_coeff - log_coeff0) * other_signs / sign0)
    return e


def compute_local_energy(
    spins: np.ndarray,
    hamiltonian: ls.Operator,
    log_coeff_fn: Callable[[np.ndarray], np.ndarray],
    extension_order: int,
    cutoff: float,
    number_sweeps: int,
    repetitions: int,
    cheat=None,
):
    if spins.ndim > 1:
        spins = spins[:, 0]
    local_energies = np.zeros(spins.shape[0], dtype=complex)
    for i in range(spins.shape[0]):
        cluster = create_cluster_for_local_energy(int(spins[i]), hamiltonian)
        if cheat is None:
            signs = optimize_connected_component(
                cluster,
                hamiltonian,
                log_coeff_fn,
                extension_order=extension_order,
                cutoff=cutoff,
                number_sweeps=number_sweeps,
                repetitions=repetitions,
            )
        else:
            (_basis, _ground_state) = cheat
            _xs = np.zeros((len(cluster), 8), dtype=np.uint64)
            _xs[:, 0] = cluster
            _rs, _, _ = _basis.batched_state_info(_xs)
            _indices = _basis.batched_index(_rs[:, 0])
            signs = np.sign(_ground_state[_indices])

        e = compute_local_energy_impl(spins[i], cluster, signs, hamiltonian, log_coeff_fn)
        local_energies[i] = e
    return local_energies


# def magically_compute_local_values(
#     spins: np.ndarray,
#     # weights: Optional[np.ndarray],
#     # ground_state: np.ndarray,
#     hamiltonian: ls.Operator,
#     log_coeff_fn: Callable[[np.ndarray], np.ndarray],
#     cutoff: float = 0,
# ):
#     # spins = np.unique(spins, axis=0)
#     h, spins, x0, counts = extract_classical_ising_model(
#         spins,
#         hamiltonian,
#         log_coeff_fn,
#         # transformed_log_coeff_fn,
#         monte_carlo_weights=np.ones(spins.shape[0]),  # weights,
#         scale_field=0,
#         cutoff=cutoff,
#     )
#     number_components, component_labels = connected_components(h.exchange, directed=False)
#     component_sizes = np.asarray([np.sum(component_labels == i) for i in range(number_components)])
#
#     local_energies = np.zeros(spins.shape[0], dtype=complex)
#     for i in range(number_components):
#         # if component_sizes[i] < 2:
#         #     local_energies[component_labels == i] = float("nan")
#         #     continue
#         local_spins = spins[component_labels == i]
#         e = compute_local_energy_one(local_spins, hamiltonian, log_coeff_fn)
#         local_energies[component_labels == i] = e
#         print(e)
#
#         # order = np.argsort(local_spins)
#         # local_spins = local_spins[order]
#         # signs = signs[order]
#         # assert np.all(np.sort(local_spins) == local_spins)
#         # # local_log_coeff = log_coeff_fn(local_spins)
#         # # local_log_coeff = local_log_coeff.real
#
#         # local_indices = hamiltonian.basis.batched_index(local_spins)
#         # true_signs = 2 * (ground_state[local_indices] >= 0) - 1
#         # accuracy = np.sum(signs == true_signs) / local_spins.shape[0]
#         # if accuracy < 0.5:
#         #     signs = -signs
#         #     accuracy = 1 - accuracy
#
#         # s = spins[batch_index]
#         # neighborhood_spins, neighborhood_coeffs = hamiltonian.apply(s)
#         # neighborhood_spins = neighborhood_spins[:, 0]
#         # neighborhood_log_coeff = log_coeff_fn(neighborhood_spins).real
#
#         # k = None
#         # for i in range(len(neighborhood_spins)):
#         #     if neighborhood_spins[i] == s:
#         #         k = i
#         #         break
#         # # k = np.searchsorted(local_spins, s)
#         # assert np.all(neighborhood_spins[k] == s)
#
#         # neighborhood_indices = np.searchsorted(local_spins, neighborhood_spins)
#         # assert np.all(local_spins[neighborhood_indices] == neighborhood_spins)
#         # neighborhood_signs = signs[neighborhood_indices]
#         # # current_sign = signs[k]
#
#         # local_energy = np.sum(
#         #     neighborhood_coeffs
#         #     * np.exp(neighborhood_log_coeff - neighborhood_log_coeff[k])
#         #     * neighborhood_signs
#         #     / neighborhood_signs[k]
#         # )
#         # print(accuracy, local_energy)
#         # local_energies.append(local_energy)
#     mask = ~np.isnan(local_energies)
#     local_energies = local_energies[mask]
#     counts = counts[mask]
#     print(np.dot(counts, local_energies) / np.sum(counts))
#     return local_energies


def create_cluster_for_local_energy(s0: int, hamiltonian: ls.Operator):
    s0 = int(s0)
    xs, _ = hamiltonian.apply(s0)
    if xs.ndim > 1:
        xs = xs[:, 0]
    xs = set((int(x) for x in xs))
    return sorted(list(xs))


def _make_hamiltonian_extension(
    spins: NDArray[np.uint64],
    ising_hamiltonian: sa.Hamiltonian,
    quantum_hamiltonian: ls.Operator,
    log_coeff_fn: Callable[[NDArray[np.uint64]], NDArray[np.float64]],
    reltol: float = 1e-2,
    log_coeff: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.uint64]:
    extended_spins, extended_coeffs, extended_counts = quantum_hamiltonian.batched_apply(spins)
    extended_spins = extended_spins[:, 0]
    assert len(extended_counts) == spins.shape[0]
    extended_log_coeff = log_coeff_fn(extended_spins)
    if log_coeff is None:
        log_coeff = log_coeff_fn(spins)

    ising_matrix = ising_hamiltonian.exchange.tocsr()
    strongest_log_couplings = np.log(get_strongest_couplings(ising_matrix))

    # log relative couplings
    extended_log_couplings = np.log(np.abs(extended_coeffs)) + extended_log_coeff.real
    offset = 0
    for i in range(len(extended_counts)):
        count = extended_counts[i]
        strongest = strongest_log_couplings[i]
        extended_log_couplings[offset : offset + count] += log_coeff[i].real - strongest
        offset += count
    mask = extended_log_couplings >= np.log(reltol)

    extended_spins = extended_spins[mask]
    spins = np.unique(np.hstack((spins, extended_spins)), axis=0)
    logger.debug(
        "{:.1f}% included; there are now {} spins in the cluster",
        100 * np.sum(mask) / len(mask),
        len(spins),
    )
    return spins


def optimize_connected_component(
    spins: np.ndarray,
    hamiltonian: ls.Operator,
    log_coeff_fn: Callable[[np.ndarray], np.ndarray],
    extension_order: int,
    cutoff: float,
    number_sweeps: int,
    repetitions: int,
):
    spins = np.asarray(spins, dtype=np.uint64)
    if spins.ndim > 1:
        spins = spins[:, 0]
    spins0 = spins
    log_coeff0 = log_coeff_fn(spins0)
    if cutoff == 0:
        absolute_log_cutoff = -np.inf
    else:
        absolute_log_cutoff = np.log(cutoff) + np.max(log_coeff0.real)
    logger.debug("Starting with a cluster of {} spins ...", len(spins))

    _temp_h, _, _, _ = extract_classical_ising_model(
        spins,
        hamiltonian,
        log_coeff_fn,
        monte_carlo_weights=None,
        scale_field=0,
    )

    for i in range(extension_order):
        extended_spins, extended_coeffs, extended_counts = hamiltonian.batched_apply(spins)
        extended_spins = extended_spins[:, 0]
        assert len(extended_counts) == spins.shape[0]
        log_coeff = log_coeff_fn(spins)
        extended_log_coeff = log_coeff_fn(extended_spins)

        mask = np.empty(len(extended_coeffs), dtype=bool)
        offset = 0
        for i in range(len(extended_counts)):
            log_couplings = (
                log_coeff[i].real
                + np.log(np.abs(extended_coeffs[offset : offset + extended_counts[i]]))
                + extended_log_coeff[offset : offset + extended_counts[i]].real
            )
            mask[offset : offset + extended_counts[i]] = log_couplings >= absolute_log_cutoff
            offset += extended_counts[i]
        extended_spins = extended_spins[mask]
        spins = np.hstack((spins, extended_spins))
        spins = np.unique(spins, axis=0)
        logger.debug(
            "{:.1f}% included; there are now {} spins in the cluster",
            100 * np.sum(mask) / len(mask),
            len(spins),
        )

    h, spins, x0, counts = extract_classical_ising_model(
        spins,
        hamiltonian,
        log_coeff_fn,
        monte_carlo_weights=None,
        scale_field=0,
    )
    order = np.argsort(np.abs(h.exchange.data))
    print(np.sum(h.exchange.data[h.exchange.data > 0]), np.sum(h.exchange.data > 0))
    print(np.sum(h.exchange.data[h.exchange.data < 0]), np.sum(h.exchange.data < 0))
    ferro = h.exchange.data < 0
    chosen = h.exchange.row[ferro]
    assert chosen.ndim == 1
    print(len(np.unique(chosen)))
    np.savetxt("couplings.dat", h.exchange.data[order])
    new_matrix = scipy.sparse.coo_matrix(
        (h.exchange.data[ferro], (h.exchange.row[ferro], h.exchange.col[ferro])),
        shape=h.exchange.shape,
    )
    print(new_matrix)
    number_components, _ = connected_components(h.exchange, directed=False)
    print(number_components)
    number_components, component_labels = connected_components(new_matrix, directed=False)
    total = 0
    for i in range(number_components):
        if np.sum(component_labels == i) > 1:
            total += 1
    print(total)
    assert False
    spins = spins[:, 0]
    # Sanity check
    number_components, _ = connected_components(h.exchange, directed=False)
    assert number_components == 1

    signs, e = sa.anneal(
        h,
        seed=None,
        number_sweeps=number_sweeps,
        repetitions=repetitions,
        only_best=True,
    )
    signs = extract_signs_from_bits(signs, number_spins=spins.shape[0])
    local_indices = np.searchsorted(spins, spins0)
    assert np.all(spins[local_indices] == spins0)
    signs = signs[local_indices]
    return signs


# def compute_accuracy_and_overlap(
#     spins: np.ndarray,
#     signs: np.ndarray,
#     hamiltonian: ls.Operator,
#     ground_state: np.ndarray,
# ):
#     spins = np.asarray(spins, dtype=np.uint64)
#     indices = hamiltonian.basis.batched_index(spins)
#     v = ground_state[indices]
#     v /= np.linalg.norm(v)
#     true_signs = 2 * (v >= 0) - 1
#     accuracy = np.sum(signs == true_signs) / len(spins)
#     if accuracy < 0.5:  # We do not care about the global sign flip
#         signs = -signs
#         accuracy = 1 - accuracy
#     overlap = abs(np.dot(v, np.abs(v) * signs))
#     return accuracy, overlap


# def sample_small_clusters(
#
# ) -> List[SmallCluster]:
#     spins, weights = monte_carlo_sampling(
#         ground_state,
#         hamiltonian,
#         number_samples=number_samples,
#         sampled_power=sampled_power,
#     )
#     pass

SETTINGS = {
    "kagome": {
        "seed": 783494,
        "yaml_path": "physical_systems/heisenberg_kagome_36.yaml",
        "hdf5_path": "physical_systems/heisenberg_kagome_36.h5",
    },
    "pyrochlore": {
        "seed": 674385,
        "yaml_path": "physical_systems/heisenberg_pyrochlore_2x2x2.yaml",
        "hdf5_path": "physical_systems/heisenberg_pyrochlore_2x2x2.h5",
    },
    "sk": {
        "seed": 783494,
        "yaml_path": "../physical_systems/sk_32_1.yaml",
        "hdf5_path": "../physical_systems/sk_32_1.h5",
    },
    "sampled_power": 0.1,
    "cutoff": 2,
    "order": 2,
    "number_samples": 5,
    "min_cluster_size": 100,
    "max_cluster_size": 5000,
    "keep_probability": 0.5,
    "number_sweeps": 10000,
    "repetitions": 64,
}


def random_cluster_size(min_size: float, max_size: float) -> int:
    u = np.random.random_sample()
    log_size = np.log(min_size) + (np.log(max_size) - np.log(min_size)) * u
    return int(round(np.exp(log_size)))


def generate_clusters(hamiltonian, ground_state, args) -> List[NDArray[np.uint64]]:
    logger.info("Monte Carlo sampling ...")
    sampling_result = monte_carlo_sampling(
        hamiltonian.basis.states,
        ground_state,
        number_samples=args.number_samples,
        sampled_power=args.sampled_power,
    )
    logger.info("Generating clusters ...")
    clusters = []
    for s in sampling_result.spins:
        size = random_cluster_size(args.min_cluster_size, args.max_cluster_size)
        cluster = create_small_cluster_around_point(
            s, hamiltonian, keep_probability=args.keep_probability, required_size=size
        )
        clusters.append(np.asarray(cluster, dtype=np.uint64))
    return clusters


def solve_and_test_model(h, frozen_spins, exact_signs, weights, annealing):
    x = solve_ising_model(h, mode="greedy", frozen_spins=frozen_spins)
    greedy_accuracy, greedy_overlap = compute_accuracy_and_overlap(x, exact_signs, weights)
    logger.info("Greedy: accuracy: {:.3f}; overlap: {:.3f}", greedy_accuracy, greedy_overlap)

    sa_accuracy = float("nan")
    sa_overlap = float("nan")
    if annealing:
        x = solve_ising_model(h, mode="sa", frozen_spins=frozen_spins)
        sa_accuracy, sa_overlap = compute_accuracy_and_overlap(x, exact_signs, weights)
        logger.info("SA:     accuracy: {:.3f}; overlap: {:.3f}", sa_accuracy, sa_overlap)
    return (h.size, greedy_accuracy, greedy_overlap, sa_accuracy, sa_overlap)


def process_cluster(cluster, hamiltonian, ground_state, log_coeff_fn, args):
    indices = hamiltonian.basis.batched_index(cluster)
    exact_psi = ground_state[indices]
    exact_signs = sa.signs_to_bits(np.sign(exact_psi))
    weights = exact_psi**2
    weights /= np.sum(weights)

    results = []

    h = make_ising_model(cluster, hamiltonian, log_psi_fn=log_coeff_fn)
    logger.debug("Extension №{}: there are {} spins in the cluster", 0, h.size)
    results.append(solve_and_test_model(h, cluster, exact_signs, weights, args.annealing))

    for i in range(args.order):
        h = make_hamiltonian_extension(h, log_coeff_fn)
        logger.debug("Extension №{}: there are now {} spins in the cluster", i + 1, h.size)
        # h = sparsify_using_local_cutoff(h, 2, cluster)
        h = sparsify_using_global_cutoff(h, args.global_cutoff, cluster)
        results.append(solve_and_test_model(h, cluster, exact_signs, weights, args.annealing))

    return results


def parse_command_line():
    parser = argparse.ArgumentParser(description="Test Simulated Annealing on larger systems.")
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--hdf5", type=str)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--annealing", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--global-cutoff", type=float, default=1e-4)
    parser.add_argument("--number-samples", type=int, default=5)
    parser.add_argument("--number-sweeps", type=int, default=5000)
    parser.add_argument("--repetitions", type=int, default=64)
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--max-cluster-size", type=int, default=1000)
    parser.add_argument("--sampled-power", type=float, default=0.1)
    parser.add_argument("--keep-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main():
    args = parse_command_line()
    np.random.seed(args.seed)
    if os.path.exists(args.output):
        logger.error(
            "Output file '{}' already exists: refusing to overwrite; "
            "delete it manually if this is what you really want",
            args.output,
        )
        return

    yaml_filename = args.yaml
    if args.hdf5 is not None:
        hdf5_filename = args.hdf5
    else:
        hdf5_filename = yaml_filename.replace(".yaml", ".h5")

    logger.info("Loading the ground state ...")
    hamiltonian = load_hamiltonian(yaml_filename)
    ground_state, ground_state_energy, _representatives = load_ground_state(hdf5_filename)
    hamiltonian.basis.build(_representatives)
    # actual_ground_state_energy = hamiltonian.expectation(ground_state)
    # if not np.isclose(ground_state_energy, actual_ground_state_energy, rtol=1e-12):
    #     raise AssertionError(
    #         "mismatch for ground state energy: {} != {}"
    #         "".format(ground_state_energy, actual_ground_state_energy)
    #     )
    log_coeff_fn = ground_state_to_log_coeff_fn(ground_state, hamiltonian.basis)

    print(args.annealing)
    clusters = generate_clusters(hamiltonian, ground_state, args)

    with open(args.output, "w") as f:
        f.write("# Generated by sampled_connected_components.py\n")
        f.write("# seed = {}\n".format(args.seed))
        f.write("# sampled_power = {}\n".format(args.sampled_power))
        f.write("# min_cluster_size = {}\n".format(args.min_cluster_size))
        f.write("# max_cluster_size = {}\n".format(args.max_cluster_size))
        f.write("# keep_probability = {}\n".format(args.keep_probability))
        f.write("# number_sweeps = {}\n".format(args.number_sweeps))
        f.write("# repetitions = {}\n".format(args.repetitions))
        f.write("# size,greedy_accuracy,greedy_overlap,sa_accuracy,sa_overlap\n")

    logger.info("Optimizing clusters ...")
    for cluster in clusters:
        columns = process_cluster(cluster, hamiltonian, ground_state, log_coeff_fn, args)
        s = ",".join(map(str, sum(map(list, columns), [])))
        with open(args.output, "a") as f:
            f.write(s + "\n")


if __name__ == "__main__":
    main()
