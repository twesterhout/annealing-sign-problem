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
from loguru import logger
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

import _build_matrix
from _build_matrix import ffi

# import unpack_bits
# import torch
# from torch import Tensor
# from torch.utils.tensorboard import SummaryWriter
# from torch.nn.parameter import Parameter  # , UninitializedParameter

# from ctypes import CDLL, POINTER, c_int64, c_uint32, c_uint64, c_double


# def project_dir():
#     return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# _lib = CDLL(os.path.join(project_dir(), "annealing_sign_problem", "libbuild_matrix.so"))


# def __preprocess_library():
#     # fmt: off
#     info = [
#         ("build_matrix", [c_uint64, POINTER(c_uint64), POINTER(c_int64), POINTER(c_double),
#             POINTER(c_uint64), POINTER(c_double), POINTER(c_int64), POINTER(c_double),
#             POINTER(c_uint32), POINTER(c_uint32), POINTER(c_double), POINTER(c_double)], c_uint64),
#         ("extract_signs", [c_uint64, POINTER(c_double), POINTER(c_uint64)], None),
#     ]
#     # fmt: on
#     for (name, argtypes, restype) in info:
#         f = getattr(_lib, name)
#         f.argtypes = argtypes
#         f.restype = restype


# __preprocess_library()


# def _int_to_ls_bits512(x: int, n: int = 8) -> np.ndarray:
#     assert n > 0
#     if n == 1:
#         return x
#     x = int(x)
#     bits = np.empty(n, dtype=np.uint64)
#     for i in range(n):
#         bits[i] = x & 0xFFFFFFFFFFFFFFFF
#         x >>= 64
#     return bits


# def _ls_bits512_to_int(bits) -> int:
#     n = len(bits)
#     if n == 0:
#         return 0
#     x = int(bits[n - 1])
#     for i in range(n - 2, -1, -1):
#         x <<= 64
#         x |= int(bits[i])
#     return x


# def batched_apply(spins, hamiltonian):
#     spins = np.asarray(spins, dtype=np.uint64, order="C")
#     if spins.ndim < 2:
#         spins = spins.reshape(-1, 1)
#     out_spins = []
#     out_coeffs = []
#     out_counts = []
#     for σ in spins:
#         out_counts.append(0)
#         for (other_σ, coeff) in zip(*hamiltonian.apply(_ls_bits512_to_int(σ))):
#             assert coeff.imag == 0
#             out_spins.append(other_σ)
#             out_coeffs.append(coeff.real)
#             out_counts[-1] += 1
#     out_spins = np.stack(out_spins)
#     out_coeffs = np.asarray(out_coeffs, dtype=np.float64)
#     out_counts = np.asarray(out_counts, dtype=np.int64)
#     logger.debug(
#         "max_buffer_size = {}, used = {}, max = {}",
#         hamiltonian.max_buffer_size,
#         out_spins.shape[0] / spins.shape[0],
#         max(out_counts),
#     )
#     return out_spins, out_coeffs, out_counts


# def get_device(obj) -> Optional[torch.device]:
#     r = _get_a_var(obj)
#     return r.device if r is not None else None


# def get_dtype(obj) -> Optional[torch.dtype]:
#     r = _get_a_var(obj)
#     return r.dtype if r is not None else None


# def _get_a_var(obj):
#     if isinstance(obj, Tensor):
#         return obj
#     if isinstance(obj, torch.nn.Module):
#         for result in obj.parameters():
#             if isinstance(result, Tensor):
#                 return result
#     if isinstance(obj, list) or isinstance(obj, tuple):
#         for result in map(get_a_var, obj):
#             if isinstance(result, Tensor):
#                 return result
#     if isinstance(obj, dict):
#         for result in map(get_a_var, obj.items()):
#             if isinstance(result, Tensor):
#                 return result
#     return None


# def split_into_batches(xs: Tensor, batch_size: int, device=None):
#     r"""Iterate over `xs` in batches of size `batch_size`. If `device` is not `None`, batches are
#     moved to `device`.
#     """
#     batch_size = int(batch_size)
#     if batch_size <= 0:
#         raise ValueError("invalid batch_size: {}; expected a positive integer".format(batch_size))
#
#     expanded = False
#     if isinstance(xs, (np.ndarray, Tensor)):
#         xs = (xs,)
#         expanded = True
#     else:
#         assert isinstance(xs, (tuple, list))
#     n = xs[0].shape[0]
#     if any(filter(lambda x: x.shape[0] != n, xs)):
#         raise ValueError("tensors 'xs' must all have the same batch dimension")
#     if n == 0:
#         return None
#
#     i = 0
#     while i + batch_size <= n:
#         chunks = tuple(x[i : i + batch_size] for x in xs)
#         if device is not None:
#             chunks = tuple(chunk.to(device) for chunk in chunks)
#         if expanded:
#             chunks = chunks[0]
#         yield chunks
#         i += batch_size
#     if i != n:  # Remaining part
#         chunks = tuple(x[i:] for x in xs)
#         if device is not None:
#             chunks = tuple(chunk.to(device) for chunk in chunks)
#         if expanded:
#             chunks = chunks[0]
#         yield chunks


# def forward_with_batches(f, xs, batch_size: int, device=None) -> Tensor:
#     r"""Applies ``f`` to all ``xs`` propagating no more than ``batch_size``
#     samples at a time. ``xs`` is split into batches along the first dimension
#     (i.e. dim=0). ``f`` must return a torch.Tensor.
#     """
#     if xs.shape[0] == 0:
#         raise ValueError("invalid xs: {}; input should not be empty".format(xs))
#     out = []
#     for chunk in split_into_batches(xs, batch_size, device):
#         out.append(f(chunk))
#     return torch.cat(out, dim=0)

# class Unpack(torch.nn.Module):
#     r"""Unpacks spin configurations represented as bits (`uint64_t` or `ls_bits512`) into a
#     2D-tensor of `float32`.
#     """
#     __constants__ = ["number_spins"]
#     number_spins: int
#
#     def __init__(self, number_spins: int):
#         super().__init__()
#         self.number_spins = number_spins
#
#     def forward(self, x: Tensor) -> Tensor:
#         if x.dim() == 1:
#             x = x.unsqueeze(dim=1)
#         return unpack_bits.unpack(x, self.number_spins)
#
#     def extra_repr(self) -> str:
#         return "number_spins={}".format(self.number_spins)


# class TensorIterableDataset(torch.utils.data.IterableDataset):
#     def __init__(self, *tensors, batch_size=1, shuffle=False):
#         assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
#         assert all(tensors[0].device == tensor.device for tensor in tensors)
#         self.tensors = tensors
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#
#     @property
#     def device(self):
#         return self.tensors[0].device
#
#     def __len__(self):
#         return self.tensors[0].size(0)
#
#     def __iter__(self):
#         if self.shuffle:
#             indices = torch.randperm(self.tensors[0].size(0), device=self.device)
#             tensors = tuple(tensor[indices] for tensor in self.tensors)
#         else:
#             tensors = self.tensors
#         return zip(*(torch.split(tensor, self.batch_size) for tensor in tensors))


# def supervised_loop_once(
#     dataset: Iterable[Tuple[Tensor, Tensor, Tensor]],
#     model: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
#     scheduler: Optional[Any],
#     swa_model=None,
#     swa_scheduler=None,
# ) -> Dict[str, float]:
#     tick = time.time()
#     model.train()
#     total_loss: float = 0.0
#     total_count: int = 0
#     for batch in dataset:
#         (x, y, w) = batch
#         w = w / torch.sum(w)
#         optimizer.zero_grad()
#         ŷ = model(x)
#         loss = loss_fn(ŷ, y, w)
#         loss.backward()
#         optimizer.step()
#         total_loss += x.size(0) * loss.item()
#         total_count += x.size(0)
#     if scheduler is not None:
#         scheduler.step()
#     if swa_model is not None:
#         assert scheduler is None
#         assert swa_scheduler is not None
#         swa_model.update_parameters(model)
#         swa_scheduler.step()
#     tock = time.time()
#     return {"loss": total_loss / total_count, "time": tock - tick}


# @torch.no_grad()
# def compute_average_loss(dataset, model, loss_fn, accuracy_fn):
#     tick = time.time()
#     model.eval()
#     total_loss = 0
#     total_sum = 0
#     total_count = 0
#     for batch in dataset:
#         (x, y, w) = batch
#         w = w / torch.sum(w)
#         ŷ = model(x)
#         loss = loss_fn(ŷ, y, w)
#         accuracy = accuracy_fn(ŷ, y, w)
#         total_loss += x.size(0) * loss.item()
#         total_sum += x.size(0) * accuracy.item()
#         total_count += x.size(0)
#     tock = time.time()
#     return {
#         "loss": total_loss / total_count,
#         "accuracy": total_sum / total_count,
#         "time": tock - tick,
#     }


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
    if spins.ndim == 1:
        spins = np.hstack([spins.reshape(-1, 1), np.zeros((spins.shape[0], 7), dtype=np.uint64)])
    elif spins.ndim == 2:
        if spins.shape[1] != 8:
            raise ValueError("'spins' has wrong shape: {}; expected (?, 8)".format(x.shape))
        spins = np.ascontiguousarray(spins)
    else:
        raise ValueError("'spins' has wrong shape: {}; expected a 2D array".format(x.shape))
    return spins


def make_ising_model(
    spins: NDArray[np.uint64],
    quantum_hamiltonian: ls.Operator,
    log_psi: Optional[NDArray[np.float64]] = None,
    log_psi_fn: Optional[Callable[[NDArray[np.uint64]], NDArray[np.float64]]] = None,
    external_field: bool = False,
    debug: bool = False,
):
    if debug:
        ref_h, ref_spins, ref_x0, ref_counts = extract_classical_ising_model(
            spins, quantum_hamiltonian, log_psi_fn
        )

    start_time = time.time()
    if log_psi is None and log_psi_fn is None:
        raise ValueError("at least one of log_psi or log_psi_fn should be specified")
    if external_field and log_psi_fn is None:
        raise ValueError("log_psi_fn should be specified when external_field=True")

    spins = _normalize_spins(spins)
    spins, indices, counts = np.unique(spins, return_index=True, return_counts=True, axis=0)
    if np.any(counts != 1):
        logger.warning("'spins' were not unique, are you sure this is what you want?")
        spins = spins[indices]
        if log_psi is not None:
            log_psi = log_psi[indices]
    if log_psi is None:
        log_psi = log_psi_fn(spins)
    n = spins.shape[0]
    other_spins, other_coeffs, other_counts = quantum_hamiltonian.batched_apply(spins)
    if not np.allclose(other_coeffs.imag, 0, atol=1e-6):
        raise ValueError("expected all Hamiltonian matrix elements to be real")
    other_coeffs = other_coeffs.real

    assert quantum_hamiltonian.basis.number_spins <= 64, "TODO: only works with up to 64 bits"
    spins = spins[:, 0]
    other_spins = other_spins[:, 0]
    other_indices = np.clip(np.searchsorted(spins, other_spins), 0, n - 1)
    belong_to_spins = other_spins == spins[other_indices]

    psi = np.exp(log_psi, dtype=np.complex128)
    if not np.allclose(psi.imag, 0, atol=1e-6):
        raise ValueError("expected all wavefunction coefficients to be real")
    psi = np.ascontiguousarray(psi.real)
    psi /= np.linalg.norm(psi)
    other_psi = np.where(belong_to_spins, psi[other_indices], 0)

    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(other_counts)
    elements = other_coeffs * np.abs(other_psi)

    # TODO: speed me up
    tick = time.time()
    for i in range(n):
        elements[offsets[i] : offsets[i + 1]] *= np.abs(psi[i])
    tock = time.time()
    # logger.debug("Python for loop took {:.2} seconds", tock - tick)

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
    x0 = np.empty((n + 63) // 64, dtype=np.uint64)
    _build_matrix.lib.extract_signs(
        n,
        ffi.from_buffer("double const[]", psi),
        ffi.from_buffer("uint64_t const[]", x0),
    )
    end_time = time.time()
    # logger.debug("Building the Ising model took {:.2} seconds", end_time - start_time)

    if debug:
        assert np.all(ref_spins[:, 0] == spins)
        ref_exchange = ref_h.exchange.tocoo()
        exchange = ising_hamiltonian.exchange.tocoo()
        assert np.all(ref_exchange.row == exchange.row)
        assert np.all(ref_exchange.col == exchange.col)
        if not np.allclose(ref_exchange.data, exchange.data):
            print(ref_exchange.data)
            print(exchange.data)
            assert False
        if external_field:
            assert np.allclose(ref_h.field, ising_hamiltonian.field)
        assert np.all(ref_x0 == x0)
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

    predicted_signs = extract_signs_from_bits(predicted, number_spins)
    exact_signs = extract_signs_from_bits(exact, number_spins)
    accuracy = np.mean(exact_signs == predicted_signs)
    accuracy = max(accuracy, 1 - accuracy)
    overlap = abs(np.dot(exact_signs * predicted_signs, weights / np.sum(weights)))
    return accuracy, overlap


def solve_ising_model(
    model: IsingModel,
    seed: int = 12345,
    number_sweeps: int = 5120,
    repetitions: int = 64,
    only_best: bool = True,
) -> NDArray[np.uint64]:
    x, e = sa.anneal(
        model.ising_hamiltonian,
        seed=seed,
        number_sweeps=number_sweeps,
        repetitions=repetitions,
        only_best=only_best,
    )
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


def color_via_spanning_tree(spins, quantum_hamiltonian, ground_state, frozen_spins):
    log_psi_fn = ground_state_to_log_coeff_fn(ground_state, quantum_hamiltonian.basis)
    ising = make_ising_model(spins, quantum_hamiltonian, log_psi_fn=log_psi_fn)

    matrix = ising.ising_hamiltonian.exchange.tocoo()
    matrix.setdiag(np.zeros(spins.size))
    matrix.eliminate_zeros()

    graph = networkx.from_scipy_sparse_matrix(matrix)
    assert networkx.is_weighted(graph)
    tree = networkx.minimum_spanning_tree(graph)
    colors = networkx.greedy_color(tree, strategy="connected_sequential_dfs")
    assert max(colors.values()) == 1

    indices = binary_search(ising.spins, frozen_spins)
    frozen_signs = np.array([2 * colors[s] - 1 for s in indices])
    return sa.signs_to_bits(frozen_signs)


def strongest_coupling_greedy_color(
    spins, quantum_hamiltonian, ground_state, frozen_spins, number_largest=1
):
    log_psi_fn = ground_state_to_log_coeff_fn(ground_state, quantum_hamiltonian.basis)
    ising = make_ising_model(spins, quantum_hamiltonian, log_psi_fn=log_psi_fn)
    infinite_graph = AlmostInfiniteGraph(quantum_hamiltonian, ground_state)

    matrix = ising.ising_hamiltonian.exchange.tocoo()
    matrix.setdiag(np.zeros(spins.size))
    matrix.eliminate_zeros()

    number_components, _ = connected_components(matrix, directed=False)
    assert number_components == 1

    def ising_edges():
        order = np.argsort(np.abs(matrix.data))[::-1]
        for k in order:
            s1 = matrix.row[k]
            s2 = matrix.col[k]
            if s1 < s2:
                c = float(matrix.data[k])
                yield (s1, s2, c)

    @dataclass
    class Cluster:
        spins: Set[int]
        signs: Dict[int, float]

    csr_matrix = matrix.tocsr()

    def merge_energy(cluster1, cluster2):
        if len(cluster1.spins) > len(cluster2.spins):
            return merge_energy(cluster2, cluster1)
        energy = 0
        for (i1, sign1) in zip(cluster1.spins, cluster1.signs):
            for k in range(csr_matrix.indptr[i1], csr_matrix.indptr[i1 + 1]):
                i2 = csr_matrix.indices[k]
                if i2 in cluster2.spins:
                    coupling = csr_matrix.data[k]
                    sign2 = cluster2.signs[i2]
                    energy += sign1 * sign2 * coupling
        return energy

    # signs = dict()
    # np.zeros(spins.size, dtype=np.float64)
    number_clusters = 0
    # next_cluster_index = 0
    # cluster_indices = dict()
    clusters = dict()
    # np.full(spins.size, -1, dtype=np.int32)

    for (s1, s2, coupling) in ising_edges():
        # Both spins already have colors
        if (s1 in clusters) and (s2 in clusters):
            cluster1 = clusters[s1]
            cluster2 = clusters[s2]
            if cluster1 == cluster2:
                # Spins belong to the same cluster. There is no reason
                # to flip spins because all previous couplings were stronger
                # than our current one
                pass
            else:
                # Check whether we need to flip one of the clusters
                # should_flip = merge_energy(cluster1, cluster2) > 0
                is_frustrated = cluster1.signs[s1] * cluster2.signs[s2] * coupling > 0
                # if is_frustrated != should_flip:
                #     logger.debug("should_flip and is_frustrated do not agree")
                should_flip = is_frustrated
                keys = list(clusters.keys())
                for key in keys:
                    if clusters[key] == cluster2:
                        sign = clusters[key].signs[key]
                        if should_flip:
                            sign *= -1

                        clusters[key] = cluster1
                        cluster1.spins.add(key)
                        cluster1.signs[key] = sign
                number_clusters -= 1
        elif s1 in clusters:
            cluster1 = clusters[s1]
            cluster2 = Cluster({s2}, {s2: 1})
            if merge_energy(cluster2, cluster1) > 0:
                cluster2.signs[s2] *= -1

            clusters[s2] = cluster1
            cluster1.spins.add(s2)
            cluster1.signs[s2] = cluster2.signs[s2]

        elif s2 in clusters:
            cluster2 = clusters[s2]
            cluster1 = Cluster({s1}, {s1: 1})
            if merge_energy(cluster1, cluster2) > 0:
                cluster1.signs[s1] *= -1

            clusters[s1] = cluster2
            cluster2.spins.add(s1)
            cluster2.signs[s1] = cluster1.signs[s1]

        else:
            # Neither of the spins has a color
            sign = -int(np.sign(coupling))
            cluster = Cluster({s1, s2}, {s1: 1, s2: sign})
            clusters[s1] = cluster
            clusters[s2] = cluster
            number_clusters += 1

    # print(number_clusters)
    # print(cluster_indices)

    assert number_clusters == 1
    assert len(clusters) == spins.size

    mega_cluster = next(iter(clusters.values()))
    for cluster in clusters.values():
        assert mega_cluster == cluster

    signs = mega_cluster.signs

    count = 0
    while True:
        changed = False
        count += 1
        for s1 in signs.keys():
            e = 0
            for k in range(csr_matrix.indptr[s1], csr_matrix.indptr[s1 + 1]):
                s2 = csr_matrix.indices[k]
                coupling = csr_matrix.data[k]
                e += signs[s2] * coupling
            e *= signs[s1]
            if e > 0:
                # logger.debug("{} is locally non-optimal", ising.spins[s1])
                changed = True
                signs[s1] *= -1
        if not changed:
            break
    print(count)

    # graph = networkx.Graph()
    # graph.add_nodes_from(signs.keys())

    # edges = []
    # for (s1, s2, coupling) in ising_edges():
    #     if s1 != s2:
    #         is_frustrated = signs[s1] * signs[s2] * coupling > 0
    #         if is_frustrated:
    #             edges.append((s1, s2, coupling))
    #             graph.add_edge(s1, s2, weight=coupling)
    # edges = sorted(edges, reverse=True, key=lambda t: abs(t[2]))
    # components = networkx.connected_components(graph)
    # print(sorted(map(len, components)))

    # def iteration_strategy(G, colors):
    #     order = np.argsort(np.abs(matrix.data))[::-1]
    #     visited = set()
    #     for k in order:
    #         s1 = int(ising.spins[matrix.row[k]])
    #         s2 = int(ising.spins[matrix.col[k]])
    #         assert s1 != s2
    #         if s1 not in visited:
    #             visited.add(s1)
    #             assert s1 in G
    #             yield s1
    #         if s2 not in visited:
    #             visited.add(s2)
    #             assert s2 in G
    #             yield s2

    # g = networkx.Graph()
    # g.add_nodes_from(spins.tolist())
    # colors = dict()
    # for s1 in iteration_strategy(g, colors):
    #     assert s1 not in colors
    #     neighbor_colors = []
    #     for s2, c in infinite_graph.neighbours(s1):
    #         if s2 != s1 and s2 in g and s2 in colors:
    #             neighbor_colors.append((colors[s2], c))

    #     if len(neighbor_colors) == 0:
    #         colors[s1] = 1
    #     else:
    #         if sum((s2 * c for s2, c in neighbor_colors)) > 0:
    #             colors[s1] = -1
    #         else:
    #             colors[s1] = 1

    # for s1 in spins:
    #     count = 0
    #     for s2, c in infinite_graph.neighbours(s1):
    #         if s2 != s1 and s2 in g:
    #             # g.add_node(s2)
    #             g.add_edge(s1, s2, weight=c)
    #             g.add_edge(s2, s1, weight=c)
    #             count += 1
    #             if count >= number_largest:
    #                 break

    # g = networkx.Graph()
    # g.add_nodes_from(spins.tolist())
    # for k in range(matrix.nnz):
    #     s1 = int(ising.spins[matrix.row[k]])
    #     s2 = int(ising.spins[matrix.col[k]])
    #     g.add_edge(s1, s2)
    #     g.add_edge(s2, s1)

    # g = networkx.convert_matrix.from_scipy_sparse_matrix(matrix)
    # colors = networkx.greedy_color(g, strategy=iteration_strategy, interchange=True)

    # for c in set(colors.values()):
    #     print(c, sum(color == c for color in colors.values()))
    # print(colors)

    # signs = np.array([2 * colors[s] - 1 for s in frozen_spins])
    frozen_indices = binary_search(ising.spins, frozen_spins)
    frozen_signs = np.array([signs[s] for s in frozen_indices])
    return sa.signs_to_bits(frozen_signs)


def strongest_coupling_model(spins, quantum_hamiltonian, ground_state, number_largest: int = 2):
    infinite_graph = AlmostInfiniteGraph(quantum_hamiltonian, ground_state)
    basis = quantum_hamiltonian.basis
    indices = basis.batched_index(spins)
    signs = np.sign(ground_state)  # [indices])

    g = networkx.Graph()
    for s1 in spins:
        g.add_node(s1)
        count = 0
        for s2, c in infinite_graph.neighbours(s1):
            if s2 != s1:
                g.add_node(s2)
                g.add_edge(s1, s2, weight=c)
                g.add_edge(s2, s1, weight=c)
                count += 1
                if count >= number_largest:
                    break

    components = list(networkx.connected_components(g))

    cycles = networkx.cycle_basis(g)
    cycle_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            s1 = int(cycle[i])
            s2 = int(cycle[(i + 1) % len(cycle)])
            cycle_edges.add((s1, s2))
            cycle_edges.add((s2, s1))

    is_largest_frustrated = np.zeros(spins.size, dtype=np.uint8)
    is_largest_within = np.zeros(spins.size, dtype=np.uint8)
    for k in range(spins.size):
        s1 = int(spins[k])
        neighbours = infinite_graph.neighbours(s1)
        count = 0
        for (s2, c) in neighbours:
            if s2 != s1:
                # print(s1, signs[basis.index(s1)], s2, signs[basis.index(s2)], c)
                if signs[basis.index(s1)] * signs[basis.index(s2)] * c > 0:
                    is_largest_frustrated[k] = 1
                    if (int(s1), int(s2)) in cycle_edges:
                        is_largest_within[k] = 1
                count += 1
                if count >= number_largest:
                    break

    logger.info(
        "Stats: spins={}, components={}, cycles={}, largest_frustrated={}, largest_in_cycle={}",
        spins.size,
        len(components),
        len(cycles),
        np.mean(is_largest_frustrated),
        np.sum(is_largest_within) / np.sum(is_largest_frustrated),
    )


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
    spins, _, _ = model.quantum_hamiltonian.batched_apply(model.spins)
    spins = np.unique(spins, axis=0)
    return make_ising_model(spins, model.quantum_hamiltonian, log_psi_fn=log_psi_fn)


def get_strongest_couplings(matrix: scipy.sparse.spmatrix) -> NDArray[Any]:
    matrix = matrix.tocsr()
    largest_couplings = matrix.max(axis=1).toarray().squeeze(axis=1)
    smallest_couplings = matrix.min(axis=1).toarray().squeeze(axis=1)
    print(largest_couplings.shape, matrix.shape)
    assert len(largest_couplings) == matrix.shape[0]
    assert len(smallest_couplings) == matrix.shape[0]
    return np.maximum(np.abs(smallest_couplings), largest_couplings)


def binary_search(haystack, needles):
    assert np.all(np.sort(haystack) == haystack)
    indices = np.searchsorted(haystack, needles)
    assert np.all(haystack[indices] == needles)
    return indices


def strategy_largest_coupling_first(G, colors):
    visited = [False for _ in len(G)]

    pass


def solve_by_colouring(model: IsingModel) -> IsingModel:
    pass


def sparsify_based_on_cutoff(model: IsingModel, cutoff: float, frozen_spins) -> IsingModel:
    frozen_spin_indices = binary_search(model.spins, frozen_spins)
    is_spin_frozen = np.ones(model.spins.shape, dtype=np.uint8)
    is_spin_frozen[frozen_spin_indices] = 1

    matrix = model.ising_hamiltonian.exchange.tocsr().copy()
    print(matrix.nnz, np.abs(matrix.data).min(), np.abs(matrix.data).max())
    strongest = get_strongest_couplings(matrix)
    print(strongest)

    for i in range(matrix.shape[0]):
        # if i == 0:
        #     print(matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]])
        c = cutoff * strongest[i]
        for k in range(matrix.indptr[i], matrix.indptr[i + 1]):
            j = matrix.indices[k]
            if np.abs(matrix.data[k]) < c and (is_spin_frozen[i] & is_spin_frozen[j]) != 0:
                matrix.data[k] = 0
    matrix.eliminate_zeros()
    matrix = 0.5 * (matrix + matrix.transpose())
    print(matrix.nnz)

    _, component_indices = connected_components(matrix, directed=False)
    magic_index = component_indices[frozen_spin_indices[0]]
    print(component_indices[frozen_spin_indices])
    assert np.all(component_indices[frozen_spin_indices] == magic_index)
    mask = component_indices == magic_index
    print(np.sum(mask))

    spins = model.spins[mask]
    signs = sa.bits_to_signs(model.initial_signs, model.size)
    signs = signs[mask]
    signs = sa.signs_to_bits(signs)

    matrix = matrix[mask][:, mask]
    print(matrix.shape)
    field = model.ising_hamiltonian.field[mask]

    # logger.debug("Constructing sa.Hamiltonian")
    ising_hamiltonian = sa.Hamiltonian(matrix, field)
    # logger.debug("Done constructing sa.Hamiltonian")
    new_model = IsingModel(
        spins,
        model.quantum_hamiltonian,
        ising_hamiltonian,
        model.initial_signs,
    )
    # logger.debug("Built new model")
    return new_model


def optimize_signs_on_cluster(
    cluster: SamplingResult,
    quantum_hamiltonian: ls.Operator,
    log_psi_fn: Callable[[NDArray[np.uint64]], NDArray[np.float64]],
    extension_order: int,
    cutoff: float,
    number_sweeps: int = 5192,
    repetitions: int = 64,
):
    h = make_ising_model(cluster.spins, quantum_hamiltonian, log_psi_fn=log_psi_fn)
    x = solve_ising_model(h, seed=None, number_sweeps=number_sweeps, repetitions=repetitions)
    results = [x]
    logger.debug("Starting with a cluster of {} spins", h.size)

    h_coo = h.ising_hamiltonian.exchange.tocoo()
    frozen_spins = cluster.spins

    for i in range(extension_order):
        h = make_hamiltonian_extension(h, log_psi_fn)

        # graph = networkx.convert_matrix.from_scipy_sparse_matrix(h.ising_hamiltonian.exchange)
        # print(list(graph.edges())[:100])
        # print(networkx.is_weighted(graph))
        # weights = networkx.get_edge_attributes(graph, "weight")
        # print(list(weights.items())[:100])

        logger.debug("Extension №{}: there are now {} spins in the cluster", i + 1, h.size)
        h = sparsify_based_on_cutoff(h, cutoff=cutoff, frozen_spins=frozen_spins)
        logger.debug("After sparsifying: {} spins in the cluster", h.size)
        number_components, component_indices = connected_components(
            h.ising_hamiltonian.exchange, directed=False
        )
        component_sizes = [np.sum(component_indices == k) for k in range(number_components)]
        # print(sorted(component_sizes))
        assert number_components == 1

        # with h5py.File("test_cluster_{}.h5".format(i), "w") as f:
        #     f["data"] = h.ising_hamiltonian.exchange.data
        #     f["indices"] = h.ising_hamiltonian.exchange.indices
        #     f["indptr"] = h.ising_hamiltonian.exchange.indptr
        #     f["field"] = h.ising_hamiltonian.field

        x = solve_ising_model(h, seed=None, number_sweeps=number_sweeps, repetitions=repetitions)
        signs = extract_signs_from_bits(x, number_spins=h.size)

        indices = np.searchsorted(h.spins, cluster.spins)
        print(indices.shape)
        assert np.all(h.spins[indices] == cluster.spins)
        print(signs.shape)
        part = extract_bits_from_signs(signs[indices])
        results.append(part)

    return results


def strongest_edge_statistics(cluster: IsingModel, full_signs, log_psi_fn):
    assert len(full_signs) == cluster.quantum_hamiltonian.basis.number_states
    extension = make_hamiltonian_extension(cluster, log_psi_fn)

    # signs = np.si

    pass

    aligned = exact_signs[matrix.row] == exact_signs[matrix.col]
    frustrated_mask = ((matrix.data > 0) & aligned) | ((matrix.data < 0) & ~aligned)
    normal_mask = ((matrix.data > 0) & ~aligned) | ((matrix.data < 0) & aligned)


def signs_to_bits(signs: NDArray[Any]) -> NDArray[np.uint64]:
    signs = np.sign(signs)
    mask = signs == 1
    assert np.all(mask | (signs == -1))
    bits = np.packbits(mask, axis=-1, bitorder="little")
    rem = len(bits) % 8
    if rem != 0:
        bits = np.pad(bits, ((0, 8 - rem),))
    return bits.view(np.uint64)


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
        out["signs"] = signs_to_bits(ground_state)


def extract_classical_ising_model(
    spins: np.ndarray,
    hamiltonian: ls.Operator,
    log_coeff_fn: Callable[[np.ndarray], np.ndarray],
    monte_carlo_weights: Optional[np.ndarray] = None,
    # sampled_power: Optional[int] = None,
    # device: Optional[torch.device] = None,
    scale_field: float = 1,
    cutoff: float = 0,
):
    r"""Map quantum Hamiltonian to classical Ising model where wavefunction coefficients are now
    considered spin degrees of freedom.

    Parameters
    ----------
    spins: numpy.ndarray
        An array of spin configurations with which classical spins will be
        associated. If `monte_carlo_weights` argument is `None`, `spins` should
        contain no duplicate elements.
    hamiltonian: lattice_symmetries.Operator
        Quantum Hamiltonian.
    log_coeff_fn: Callable
        A function which given a batch of spin configurations, computes
        `log(ψ(s))` for each spin configuration `s`. This function should
        return a `numpy.ndarray` of `numpy.complex128`.
    monte_carlo_weights: numpy.ndarray, optional
        Specifies weights of Monte Carlo samples, if `spins` was obtained by
        Monte Carlo sampling.
    scale_field: float
        How to scale external fields, set it to `0` if you wish to ignore
        external fields.
    """
    spins = _normalize_spins(spins)
    # If spins come from Monte Carlo sampling, there might be duplicates.
    is_from_monte_carlo = monte_carlo_weights is not None
    spins, indices, counts = np.unique(spins, return_index=True, return_counts=True, axis=0)
    if not is_from_monte_carlo and np.any(counts != 1):
        raise ValueError("'spins' contains duplicate spin configurations")
    if is_from_monte_carlo:
        assert False
        monte_carlo_weights = monte_carlo_weights[indices]
        monte_carlo_weights /= np.sum(monte_carlo_weights)

    def forward(x: NDArray[np.uint64]) -> NDArray[np.complex128]:
        assert isinstance(x, np.ndarray) and x.dtype == np.uint64
        r = log_coeff_fn(x)
        assert r.shape == (x.shape[0],) and r.dtype == np.complex128
        return r

    log_ψs = forward(spins)
    other_spins, other_coeffs, other_counts = hamiltonian.batched_apply(spins)
    assert np.all(other_counts > 0)
    if not np.allclose(other_coeffs.imag, 0):
        raise ValueError("expected all matrix elements to be real")
    other_coeffs = np.ascontiguousarray(other_coeffs.real)
    other_log_ψs = forward(other_spins)

    # Safer exponentiation
    scale = max(np.max(other_log_ψs.real), np.max(log_ψs.real))
    other_log_ψs.real -= scale
    other_ψs = np.exp(other_log_ψs, dtype=np.complex128)
    if not np.allclose(other_ψs.imag, 0, atol=1e-6):
        raise ValueError("expected all wavefunction coefficients to be real")
    other_ψs = np.ascontiguousarray(other_ψs.real)
    other_log_ψs = None

    log_ψs.real -= scale
    ψs = np.exp(log_ψs, dtype=np.complex128)
    if not np.allclose(ψs.imag, 0, atol=1e-6):
        raise ValueError("expected all wavefunction coefficients to be real")
    ψs = np.ascontiguousarray(ψs.real)
    log_ψs = None

    if is_from_monte_carlo:
        assert np.all(monte_carlo_weights > 0)
        # normalization = 1 / np.sum(monte_carlo_weights)
        # monte_carlo_weights *= np.exp(-scale)
        # normalization = 1 / np.sum(monte_carlo_weights)
        # weights = monte_carlo_weights / ψs ** 2
        # normalization = np.exp(-2 * scale) / np.sum(weights)
        normalization = 1 / np.sqrt(np.dot(counts, np.abs(ψs) ** 2 / monte_carlo_weights))
        ψs /= monte_carlo_weights
    else:
        normalization = 1 / np.linalg.norm(ψs)
    ψs *= normalization
    other_ψs *= normalization

    n = spins.shape[0]
    field = np.zeros(n, dtype=np.float64)
    row_indices = np.empty(other_spins.shape[0], dtype=np.uint32)
    col_indices = np.empty(other_spins.shape[0], dtype=np.uint32)
    elements = np.empty(other_spins.shape[0], dtype=np.float64)
    written = _build_matrix.lib.build_matrix(
        n,
        ffi.from_buffer("ls_bits512 const[]", spins),
        ffi.from_buffer("int64_t const[]", counts),
        ffi.from_buffer("double const[]", ψs),
        ffi.from_buffer("ls_bits512 const[]", other_spins),
        ffi.from_buffer("double const[]", other_coeffs),
        ffi.from_buffer("int64_t const[]", other_counts),
        ffi.from_buffer("double const[]", other_ψs),
        ffi.from_buffer("uint32_t[]", row_indices),
        ffi.from_buffer("uint32_t[]", col_indices),
        ffi.from_buffer("double[]", elements),
        ffi.from_buffer("double[]", field),
    )
    row_indices = row_indices[:written]
    col_indices = col_indices[:written]
    elements = elements[:written]
    matrix = scipy.sparse.coo_matrix((elements, (row_indices, col_indices)), shape=(n, n))
    # Convert COO matrix to CSR to ensure that duplicate elements are summed
    # together. Duplicate elements can arise when working in symmetry-adapted
    # bases.
    matrix = matrix.tocsr()
    # Symmetrize the matrix if spins come from Monte Carlo sampling. When
    # matrix elements do not come from Monte Carlo sampling, symmetrizing does
    # not hurt and may fix some numerical differences.
    matrix = 0.5 * (matrix + matrix.T)
    matrix = matrix.tocoo()
    mask = np.abs(matrix.data) >= cutoff * np.max(np.abs(matrix.data))
    row_indices = matrix.row[mask]
    col_indices = matrix.col[mask]
    elements = matrix.data[mask]
    matrix = scipy.sparse.coo_matrix((elements, (row_indices, col_indices)), shape=(n, n))
    field *= scale_field
    h = sa.Hamiltonian(matrix, field)

    x0 = np.empty((spins.shape[0] + 63) // 64, dtype=np.uint64)
    _build_matrix.lib.extract_signs(
        n,
        ffi.from_buffer("double const[]", ψs),
        ffi.from_buffer("uint64_t const[]", x0),
    )

    logger.debug(
        "The classical Hamiltonian has dimension {} and contains {} non-zero elements.",
        n,
        written,
    )
    logger.debug("Jₘᵢₙ = {}, Jₘₐₓ = {}", np.abs(matrix.data).min(), np.abs(matrix.data).max())
    logger.debug("Bₘᵢₙ = {}, Bₘₐₓ = {}", np.abs(field).min(), np.abs(field).max())
    return h, spins, x0, counts


def extract_bits_from_signs(signs: NDArray[np.float64]) -> NDArray[np.uint64]:
    return sa.signs_to_bits(signs)


def extract_signs_from_bits(bits: np.ndarray, number_spins: int) -> np.ndarray:
    assert bits.dtype == np.uint64 and bits.ndim == 1
    i = np.arange(number_spins, dtype=np.uint64)
    signs = (bits[i // 64] >> (i % 64)) & 1
    signs = 2 * signs.astype(np.float64) - 1
    assert np.all((signs == 1) | (signs == -1))
    return signs


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
