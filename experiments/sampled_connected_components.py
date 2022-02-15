from annealing_sign_problem import *
import collections
import ising_glass_annealer as sa
from loguru import logger
import networkx
import numpy as np
import torch
from torch import Tensor
from typing import Tuple, List, Optional
import unpack_bits
import nqs_playground as nqs
import scipy.sparse
from scipy.sparse.csgraph import connected_components


def monte_carlo_sampling(
    ground_state: np.ndarray,
    hamiltonian: ls.Operator,
    number_samples: int,
    sampled_power: float,
) -> Tuple[np.ndarray, np.ndarray]:
    p = np.abs(ground_state) ** sampled_power
    p /= np.sum(p)
    indices = np.random.choice(
        hamiltonian.basis.number_states, size=number_samples, replace=True, p=p
    )
    spins = hamiltonian.basis.states[indices]
    weights = p[indices]
    return spins, weights


def slice_coo_matrix(
    matrix: scipy.sparse.coo_matrix, indices: np.ndarray
) -> scipy.sparse.coo_matrix:
    indices = np.asarray(indices)
    assert np.all(indices == np.sort(indices))

    row_indices = np.searchsorted(indices, matrix.row)
    row_indices[row_indices >= len(indices)] = len(indices) - 1
    row_mask = indices[row_indices] == matrix.row
    col_indices = np.searchsorted(indices, matrix.col)
    col_indices[col_indices >= len(indices)] = len(indices) - 1
    col_mask = indices[col_indices] == matrix.col
    mask = row_mask & col_mask

    new_row = row_indices[mask]
    new_col = col_indices[mask]
    new_data = matrix.data[mask]

    return scipy.sparse.coo_matrix((new_data, (new_row, new_col)))


def extract_local_hamiltonian(
    mask: np.ndarray,
    hamiltonian: sa.Hamiltonian,
    spins: Tensor,
    scale_field: float = 1,
) -> Tuple[sa.Hamiltonian, Tensor]:
    (spin_indices,) = np.nonzero(mask)
    spins = spins[spin_indices]
    field = hamiltonian.field[spin_indices]
    field *= scale_field
    exchange = slice_coo_matrix(hamiltonian.exchange, spin_indices)
    return sa.Hamiltonian(exchange, field), spins


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


def optimize_small_cluster(
    spins: np.ndarray,
    hamiltonian: ls.Operator,
    log_coeff_fn: Callable[[np.ndarray], np.ndarray],
    max_spins: int,
    cutoff: float,
):
    if spins.ndim > 1:
        spins = spins[:, 0]
    spins0 = spins
    log_coeff0 = log_coeff_fn(spins0)
    absolute_log_cutoff = np.log(cutoff) + np.max(log_coeff0.real)
    logger.debug("Starting with a cluster of {} spins", spins.shape[0])
    for tries in range(5):
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
        logger.debug("{:.1f}% included", 100 * np.sum(mask) / len(mask))
        extended_spins = extended_spins[mask]

        spins = np.hstack((spins, extended_spins))
        spins = np.unique(spins, axis=0)
        logger.debug("Extended to {} spins", spins.shape[0])
        if spins.shape[0] >= max_spins:
            break

    h, spins, x0, counts = extract_classical_ising_model(
        spins,
        hamiltonian,
        log_coeff_fn,
        # transformed_log_coeff_fn,
        monte_carlo_weights=None,
        scale_field=0,
    )
    spins = spins[:, 0]
    number_components, component_labels = connected_components(h.exchange, directed=False)
    assert number_components == 1

    signs, e = sa.anneal(
        h,
        seed=None,
        number_sweeps=10000,
        repetitions=64,
        only_best=True,
    )
    signs = extract_signs_from_bits(signs, number_spins=spins.shape[0])
    local_indices = np.searchsorted(spins, spins0)
    assert np.all(spins[local_indices] == spins0)

    return spins0, signs[local_indices]


ComponentStats = collections.namedtuple("ComponentStats", ["size", "accuracy", "overlap"])


def build_and_analyze_clusters(
    spins: np.ndarray,
    weights: Optional[np.ndarray],
    ground_state: np.ndarray,
    hamiltonian: ls.Operator,
    log_coeff_fn: Callable[[np.ndarray], np.ndarray],
    cutoff: float = 0,
):
    if weights is None:
        spins = np.unique(spins, axis=0)

    def transformed_log_coeff_fn(x):
        r = log_coeff_fn(x)
        r.imag = 0
        return r

    h, spins, x0, counts = extract_classical_ising_model(
        spins,
        hamiltonian,
        log_coeff_fn,
        # transformed_log_coeff_fn,
        monte_carlo_weights=weights,
        scale_field=1,
        cutoff=cutoff,
    )
    number_components, component_labels = connected_components(h.exchange, directed=False)
    component_sizes = np.asarray([np.sum(component_labels == i) for i in range(number_components)])

    results = []
    count = 0
    for i in range(number_components):
        if component_sizes[i] < 10:
            continue
        count += 1
        # if count > 2:
        #     break
        _, local_spins = extract_local_hamiltonian(component_labels == i, h, spins, scale_field=0)
        local_spins, signs = optimize_small_cluster(
            local_spins, hamiltonian, log_coeff_fn, 1000, cutoff=1e-3
        )

        # if component_sizes[i] < 10:
        #     continue
        # logger.debug("Processing cluster with {} elements...", component_sizes[i])
        # local_hamiltonian_with_fields, _ = extract_local_hamiltonian(
        #     component_labels == i, h, spins, scale_field=1
        # )
        # local_hamiltonian, local_spins = extract_local_hamiltonian(
        #     component_labels == i, h, spins, scale_field=0
        # )
        # signs, e = sa.anneal(
        #     local_hamiltonian,
        #     seed=None,
        #     number_sweeps=5000,
        #     repetitions=32,
        #     only_best=True,
        # )
        # signs = extract_signs_from_bits(signs, number_spins=local_spins.shape[0])
        local_indices = hamiltonian.basis.batched_index(local_spins[:, 0])

        true_signs = 2 * (ground_state[local_indices] >= 0) - 1
        accuracy = np.sum(signs == true_signs) / local_spins.shape[0]
        if accuracy < 0.5:
            signs = -signs
            accuracy = 1 - accuracy

        v = ground_state[local_indices]
        v /= np.linalg.norm(v)
        overlap = abs(np.dot(v, np.abs(v) * signs))
        results.append(
            ComponentStats(size=local_spins.shape[0], accuracy=accuracy, overlap=overlap)
        )
        # results.append(ComponentStats(size=component_sizes[i], accuracy=accuracy, overlap=overlap))

        # table = []
        # for k in range(local_spins.shape[0]):
        #     is_correct = signs[k] == true_signs[k]
        #     field = abs(local_hamiltonian_with_fields.field[k])
        #     mask = local_hamiltonian_with_fields.exchange.row == k
        #     coupling = np.sum(np.abs(local_hamiltonian_with_fields.exchange.data[mask]))
        #     table.append((int(is_correct), field, coupling))
        # with open("correlation.dat", "a") as f:
        #     for t in table:
        #         f.write("{}\t{}\t{}\n".format(*t))

    return results


def compute_local_energy_one(
    spins: np.ndarray,
    # weights: Optional[np.ndarray],
    # ground_state: np.ndarray,
    hamiltonian: ls.Operator,
    log_coeff_fn: Callable[[np.ndarray], np.ndarray],
):
    if spins.ndim > 1:
        spins = spins[:, 0]

    initial_spins = spins

    spins, _, _ = hamiltonian.batched_apply(spins)
    spins = np.unique(spins, axis=0)
    spins, signs = optimize_small_cluster(spins, hamiltonian, log_coeff_fn, 10000, cutoff=1e-3)
    order = np.argsort(spins)
    spins = spins[order]
    signs = signs[order]

    local_energies = []
    for batch_index in range(initial_spins.shape[0]):
        neighborhood_spins, neighborhood_coeffs = hamiltonian.apply(initial_spins[batch_index])
        neighborhood_spins = neighborhood_spins[:, 0]
        neighborhood_log_coeff = log_coeff_fn(neighborhood_spins).real
        current_log_coeff = log_coeff_fn(initial_spins[batch_index:batch_index + 1]).real

        neighborhood_indices = np.searchsorted(spins, neighborhood_spins)
        assert np.all(spins[neighborhood_indices] == neighborhood_spins)
        neighborhood_signs = signs[neighborhood_indices]

        current_index = np.searchsorted(spins, initial_spins[batch_index])
        assert spins[current_index] == initial_spins[batch_index]
        current_sign = signs[current_index]

        local_energy = np.sum(
            neighborhood_coeffs
            * np.exp(neighborhood_log_coeff - current_log_coeff)
            * neighborhood_signs
            / current_sign
        )
        local_energies.append(local_energy)
    return np.asarray(local_energies)


def magically_compute_local_values(
    spins: np.ndarray,
    # weights: Optional[np.ndarray],
    # ground_state: np.ndarray,
    hamiltonian: ls.Operator,
    log_coeff_fn: Callable[[np.ndarray], np.ndarray],
    cutoff: float = 0,
):
    # spins = np.unique(spins, axis=0)
    h, spins, x0, counts = extract_classical_ising_model(
        spins,
        hamiltonian,
        log_coeff_fn,
        # transformed_log_coeff_fn,
        monte_carlo_weights=np.ones(spins.shape[0]), # weights,
        scale_field=0,
        cutoff=cutoff,
    )
    number_components, component_labels = connected_components(h.exchange, directed=False)
    component_sizes = np.asarray([np.sum(component_labels == i) for i in range(number_components)])

    local_energies = np.zeros(spins.shape[0], dtype=complex)
    for i in range(number_components):
        # if component_sizes[i] < 2:
        #     local_energies[component_labels == i] = float("nan")
        #     continue
        local_spins = spins[component_labels == i]
        e = compute_local_energy_one(local_spins, hamiltonian, log_coeff_fn)
        local_energies[component_labels == i] = e
        print(e)

        # order = np.argsort(local_spins)
        # local_spins = local_spins[order]
        # signs = signs[order]
        # assert np.all(np.sort(local_spins) == local_spins)
        # # local_log_coeff = log_coeff_fn(local_spins)
        # # local_log_coeff = local_log_coeff.real

        # local_indices = hamiltonian.basis.batched_index(local_spins)
        # true_signs = 2 * (ground_state[local_indices] >= 0) - 1
        # accuracy = np.sum(signs == true_signs) / local_spins.shape[0]
        # if accuracy < 0.5:
        #     signs = -signs
        #     accuracy = 1 - accuracy

        # s = spins[batch_index]
        # neighborhood_spins, neighborhood_coeffs = hamiltonian.apply(s)
        # neighborhood_spins = neighborhood_spins[:, 0]
        # neighborhood_log_coeff = log_coeff_fn(neighborhood_spins).real

        # k = None
        # for i in range(len(neighborhood_spins)):
        #     if neighborhood_spins[i] == s:
        #         k = i
        #         break
        # # k = np.searchsorted(local_spins, s)
        # assert np.all(neighborhood_spins[k] == s)

        # neighborhood_indices = np.searchsorted(local_spins, neighborhood_spins)
        # assert np.all(local_spins[neighborhood_indices] == neighborhood_spins)
        # neighborhood_signs = signs[neighborhood_indices]
        # # current_sign = signs[k]

        # local_energy = np.sum(
        #     neighborhood_coeffs
        #     * np.exp(neighborhood_log_coeff - neighborhood_log_coeff[k])
        #     * neighborhood_signs
        #     / neighborhood_signs[k]
        # )
        # print(accuracy, local_energy)
        # local_energies.append(local_energy)
    mask = ~ np.isnan(local_energies)
    local_energies = local_energies[mask]
    counts = counts[mask]
    print(np.dot(counts, local_energies) / np.sum(counts))
    return local_energies


def main():
    np.random.seed(12346)
    # yaml_path = "../data/symm/heisenberg_kagome_36.yaml"
    # hdf5_path = "../data/symm/heisenberg_kagome_36.h5"
    yaml_path = "../physical_systems/heisenberg_pyrochlore_2x2x2.yaml"
    hdf5_path = "../physical_systems/heisenberg_pyrochlore_2x2x2.h5"
    hamiltonian = load_hamiltonian(yaml_path)
    ground_state, ground_state_energy, _representatives = load_ground_state(hdf5_path)
    hamiltonian.basis.build(_representatives)
    logger.info("Ground state energy: {}", ground_state_energy)

    # mask = np.abs(ground_state) < 1e-10
    # ground_state[mask] = 1e-10
    # print(ground_state_energy)
    # print(hamiltonian.expectation(ground_state))
    # return

    log_coeff_fn = ground_state_to_log_coeff_fn(ground_state, hamiltonian.basis)

    stats = []
    sampled_power = 2
    cutoff = 0  # 2e-3
    number_repetitions = 1
    # filename = "stats_pyrochlore_2x2x2_p={}_cutoff={}.dat".format(sampled_power, cutoff)
    for number_samples in [1000]:  # [100 * i for i in range(10, 80, 2)]:
        for i in range(number_repetitions):
            spins, weights = monte_carlo_sampling(
                ground_state,
                hamiltonian,
                number_samples=number_samples,
                sampled_power=sampled_power,
            )
            r = magically_compute_local_values(
                spins, hamiltonian, log_coeff_fn, cutoff=cutoff
            )
            r = np.asarray(r)
            print(r / 32 / 4, np.mean(r) / 32 / 4, np.std(r) / 32 / 4)
            # r = build_and_analyze_clusters(
            #     spins, None, ground_state, hamiltonian, log_coeff_fn, cutoff=cutoff
            # )
            # stats += r
        # with open(filename, "w") as f:
        #     for s in sorted(stats, key=lambda s: s.size):
        #         f.write("{}\t{}\t{}\n".format(s.size, s.accuracy, s.overlap))
    print(ground_state_energy / 32 / 4)

    # model = load_unsymmetrized()
    # model = load_cnn()
    # device = torch.device("cpu")
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     device = torch.device("cuda:0")
    # for name, p in model.named_parameters():
    #     print(name, p.device)


if __name__ == "__main__":
    main()
