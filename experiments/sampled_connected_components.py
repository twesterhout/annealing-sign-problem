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
    for i in range(number_components):
        if component_sizes[i] < 10:
            continue
        logger.debug("Processing cluster with {} elements...", component_sizes[i])
        # local_hamiltonian_with_fields, _ = extract_local_hamiltonian(
        #     component_labels == i, h, spins, scale_field=1
        # )
        local_hamiltonian, local_spins = extract_local_hamiltonian(
            component_labels == i, h, spins, scale_field=0
        )
        signs, e = sa.anneal(
            local_hamiltonian,
            seed=None,
            number_sweeps=5000,
            repetitions=32,
            only_best=True,
        )
        signs = extract_signs_from_bits(signs, number_spins=local_spins.shape[0])
        local_indices = hamiltonian.basis.batched_index(local_spins[:, 0])

        true_signs = 2 * (ground_state[local_indices] >= 0) - 1
        accuracy = np.sum(signs == true_signs) / local_spins.shape[0]
        if accuracy < 0.5:
            signs = -signs
            accuracy = 1 - accuracy

        v = ground_state[local_indices]
        v /= np.linalg.norm(v)
        overlap = abs(np.dot(v, np.abs(v) * signs))
        results.append(ComponentStats(size=component_sizes[i], accuracy=accuracy, overlap=overlap))

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


def main():
    np.random.seed(12345)
    yaml_path = "../data/symm/heisenberg_kagome_36.yaml"
    hdf5_path = "../data/symm/heisenberg_kagome_36.h5"
    # yaml_path = "../physical_systems/heisenberg_pyrochlore_2x2x2.yaml"
    # hdf5_path = "../physical_systems/heisenberg_pyrochlore_2x2x2.h5"
    hamiltonian = load_hamiltonian(yaml_path)
    ground_state, ground_state_energy, _representatives = load_ground_state(hdf5_path)
    hamiltonian.basis.build(_representatives)
    log_coeff_fn = ground_state_to_log_coeff_fn(ground_state, hamiltonian.basis)

    stats = []
    sampled_power = 1.5
    cutoff = 4e-4
    number_repetitions = 5
    filename = "stats_kagome_36_p={}_cutoff={}.dat".format(sampled_power, cutoff)
    for number_samples in [100 * i for i in range(100, 500, 25)]:
        for i in range(number_repetitions):
            spins, weights = monte_carlo_sampling(
                ground_state,
                hamiltonian,
                number_samples=number_samples,
                sampled_power=sampled_power,
            )
            r = build_and_analyze_clusters(
                spins, None, ground_state, hamiltonian, log_coeff_fn, cutoff=cutoff
            )
            stats += r
        with open(filename, "w") as f:
            for s in sorted(stats, key=lambda s: s.size):
                f.write("{}\t{}\t{}\n".format(s.size, s.accuracy, s.overlap))

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
