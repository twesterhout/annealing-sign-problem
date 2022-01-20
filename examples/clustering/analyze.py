from models import load_rbm, Phase
import collections
from unpack_bits import unpack
import nqs_playground as nqs
import math
from loguru import logger
import torch
from torch import Tensor
from typing import Any, List, Tuple
import numpy as np
import networkx
import ising_glass_annealer as sa
import lattice_symmetries as ls
import scipy.sparse
import scipy.sparse.csgraph
from annealing_sign_problem import extract_classical_ising_model

DEFAULT_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

System = collections.namedtuple("System", ["hamiltonian", "log_amplitude", "sign"])


def heisenberg1d_model(number_spins: int = 40, transform_signs: bool = True):
    basis = ls.SpinBasis(ls.Group([]), number_spins=number_spins, hamming_weight=number_spins // 2)
    c = -1 if transform_signs else 1
    # fmt: off
    matrix = np.array([[1,     0,     0, 0],
                       [0,    -1, 2 * c, 0],
                       [0, 2 * c,    -1, 0],
                       [0,     0,     0, 1]])
    # fmt: on
    edges = [(i, (i + 1) % number_spins) for i in range(number_spins)]
    hamiltonian = ls.Operator(basis, [ls.Interaction(matrix, edges)])
    return hamiltonian


def checkerboard(shape: Tuple[int, int]) -> Tensor:
    assert shape[0] * shape[1] % 2 == 0
    sites = np.arange(shape[0] * shape[1]).reshape(*shape)
    sites = sites % shape[1] + sites // shape[1]
    sites = sites % 2
    return torch.from_numpy(sites)


class MarshallSignRule(torch.nn.Module):
    def __init__(self, shape: Tuple[int, int], dtype: torch.dtype):
        super().__init__()
        self.register_buffer("mask", checkerboard(shape).view(-1).to(dtype))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        # if x.dtype == torch.int64:
        n = self.mask.size(0)
        x = unpack(x, n)
        mask = self.mask.to(device=x.device)
        # 0 in bias means + and 1 means -
        bias = ((n - (x * mask).sum(dim=1)) // 2) % 2
        # return math.pi * bias.view(-1, 1)
        probability = torch.stack([1 - bias, bias], dim=1)
        return probability


@torch.no_grad()
def load_rbm_ansatz(
    filename: str = "../scaling/Nqs/Ground/Heisenberg1d_40_1_4.wf",
    device: torch.device = DEFAULT_DEVICE,
) -> System:
    log_amplitude_fn = load_rbm(filename).to(device)
    sign_fn = MarshallSignRule((1, log_amplitude_fn.in_features), get_dtype(log_amplitude_fn))
    hamiltonian = heisenberg1d_model(log_amplitude_fn.in_features, transform_signs=False)
    return System(hamiltonian, log_amplitude_fn, sign_fn)


class ExactLogAmplitude(torch.nn.Module):
    def __init__(self, basis: ls.SpinBasis, ground_state: Tensor):
        super().__init__()
        self.basis = basis
        self.register_buffer("log_amplitudes", ground_state.detach().abs().log_())

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() > 1:
            x = x[:, 0]
        cpu_spins = x.cpu().numpy().view(np.uint64)
        cpu_indices = self.basis.batched_index(cpu_spins).view(np.int64)
        indices = torch.from_numpy(cpu_indices).to(self.log_amplitudes.device)
        return self.log_amplitudes[indices].view([-1, 1])


def _make_sign_probabilities(ground_state: Tensor) -> Tensor:
    signs = (torch.sign(ground_state).to(torch.int64) + 1) // 2
    assert torch.all((signs == 0) | (signs == 1))
    return torch.stack([1 - signs, signs], dim=1)


class ExactSign(torch.nn.Module):
    def __init__(self, basis: ls.SpinBasis, ground_state: Tensor):
        super().__init__()
        self.basis = basis
        self.register_buffer("probabilities", _make_sign_probabilities(ground_state))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() > 1:
            x = x[:, 0]
        cpu_spins = x.cpu().numpy().view(np.uint64)
        cpu_indices = self.basis.batched_index(cpu_spins).view(np.int64)
        indices = torch.from_numpy(cpu_indices).to(self.probabilities.device)
        return self.probabilities[indices]


@torch.no_grad()
def load_ground_state(
    yaml_filename: str = "../../data/symm/heisenberg_kagome_36.yaml",
    hdf5_filename: str = "../../data/symm/heisenberg_kagome_36.h5",
    device: torch.device = DEFAULT_DEVICE,
) -> System:
    hamiltonian = nqs.load_hamiltonian(yaml_filename)
    ground_state, energy, representatives = nqs.load_ground_state(hdf5_filename)
    hamiltonian.basis.build(representatives)
    log_amplitude_fn = ExactLogAmplitude(hamiltonian.basis, ground_state).to(device)
    sign_fn = ExactSign(hamiltonian.basis, ground_state).to(device)
    # print(log_amplitude_fn.log_amplitudes.device)
    # print(sign_fn.probabilities.device)
    return System(hamiltonian, log_amplitude_fn, sign_fn)


@torch.no_grad()
def monte_carlo_sample(
    system: System,
    sampled_power: float,
    number_samples: int = 5000 // 64,
    number_chains: int = 64,
    sweep_size: int = 4,
    number_discarded: int = 500,
    mode: str = "zanella",
) -> Tensor:
    basis = system.hamiltonian.basis
    sampling_options = nqs.SamplingOptions(
        number_samples=number_samples,
        number_chains=number_chains,
        number_discarded=number_discarded,
        sweep_size=sweep_size,
        mode=mode,
        device=nqs.get_device(system.log_amplitude),
    )
    log_prob_fn = lambda x: sampled_power * system.log_amplitude(x)
    spins, _, weights, info = nqs.sample_some(
        log_prob_fn, basis, sampling_options, is_log_prob_fn=True
    )
    # Ignore different Markov chains
    spins = spins.view(-1, spins.size(-1))
    weights = weights.view(-1)
    log_coeff_fn = nqs.combine_amplitude_and_sign(system.log_amplitude, system.sign, use_jit=False)
    local_energies = nqs.local_values(spins, system.hamiltonian, log_coeff_fn, batch_size=16384)
    local_energies = local_energies.real
    mean_energy = torch.dot(weights, local_energies)
    energy_variance = torch.dot(weights, (local_energies - mean_energy) ** 2)
    logger.info("Energy: {}", mean_energy.item())
    logger.info("Energy variance: {}", energy_variance.item())
    return spins


@torch.no_grad()
def construct_classical_hamiltonian(
    system: System, spins: Tensor, sampled_power: float, scale_field: float = 0
) -> sa.Hamiltonian:
    cpu_spins0 = spins.cpu().numpy().view(np.uint64)
    log_coeff_fn = nqs.combine_amplitude_and_sign(system.log_amplitude, system.sign, use_jit=False)
    device = nqs.get_device(log_coeff_fn)
    h, cpu_spins, x0, cpu_counts = extract_classical_ising_model(
        cpu_spins0,
        system.hamiltonian,
        log_coeff_fn,
        sampled_power=sampled_power,
        device=device,
        scale_field=scale_field,
    )
    spins = torch.from_numpy(cpu_spins.view(np.int64)).to(device)
    counts = torch.from_numpy(cpu_counts).to(device)

    return h, spins, counts


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

    # row = []
    # col = []
    # data = []
    # for r, c, J in zip(matrix.row, matrix.col, matrix.data):
    #     if r in indices and c in indices:
    #         r_new = np.searchsorted(indices, r)
    #         c_new = np.searchsorted(indices, c)
    #         row.append(r_new)
    #         col.append(c_new)
    #         data.append(J)
    # assert np.all(row == new_row)
    # assert np.all(col == new_col)
    # assert np.all(data == new_data)
    return scipy.sparse.coo_matrix((new_data, (new_row, new_col)))


def extract_local_hamiltonian(
    mask: np.ndarray, hamiltonian: sa.Hamiltonian, spins: Tensor
) -> Tuple[sa.Hamiltonian, Tensor]:
    (spin_indices,) = np.nonzero(mask)
    spins = spins[spin_indices]
    field = hamiltonian.field[spin_indices]
    exchange = slice_coo_matrix(hamiltonian.exchange, spin_indices)
    return sa.Hamiltonian(exchange, field), spins


def is_frustrated(matrix: scipy.sparse.coo_matrix) -> bool:
    def extract(mask):
        return scipy.sparse.coo_matrix(
            (matrix.data[mask], (matrix.row[mask], matrix.col[mask])), shape=matrix.shape
        )

    off_diagonal = matrix.row != matrix.col
    matrix = extract(off_diagonal)

    graph = networkx.convert_matrix.from_scipy_sparse_matrix(matrix)
    assert networkx.is_connected(graph)
    positive_graph = networkx.convert_matrix.from_scipy_sparse_matrix(extract(matrix.data > 0))
    positive_coloring = networkx.coloring.greedy_color(
        positive_graph, strategy="connected_sequential"
    )
    number_positive_colors = max(positive_coloring.values()) + 1
    if number_positive_colors > 2:
        logger.debug('"J > 0"-subgraph introduces frustration')
        assert not networkx.algorithms.bipartite.is_bipartite(positive_graph)
        return True
    else:
        logger.debug('"J > 0"-subgraph introduces no frustration')

    # assert networkx.is_connected(positive_graph)
    negative_graph = networkx.convert_matrix.from_scipy_sparse_matrix(extract(matrix.data < 0))
    negative_coloring = networkx.coloring.greedy_color(
        negative_graph, strategy="connected_sequential"
    )
    number_negative_colors = max(negative_coloring.values()) + 1
    if number_negative_colors > 2:
        logger.debug('"J < 0"-subgraph introduces frustration')
        assert not networkx.algorithms.bipartite.is_bipartite(negative_graph)
        return True
    else:
        logger.debug('"J < 0"-subgraph introduces no frustration')

    if number_positive_colors < 2:
        assert np.sum(matrix.data > 0) == 0
        logger.debug("There are no positive couplings")
        return False
    if number_negative_colors < 2:
        assert np.sum(matrix.data < 0) == 0
        logger.debug("There are no negative couplings")
        return False
    return positive_coloring != negative_coloring


def anneal_with_retries(hamiltonian, number_sweeps: int = 5000, retry: int = 10):
    def try_once():
        x, _, e = sa.anneal(
            hamiltonian,
            x0=None,
            seed=np.random.randint(1 << 31),
            number_sweeps=number_sweeps,
            beta0=None,
            beta1=None,
        )
        return e[-1], x

    _, x = min((try_once() for _ in range(retry)))
    return x


@torch.no_grad()
def optimize_connected_components(
    system: System, hamiltonian: sa.Hamiltonian, spins: Tensor, how_many: int = 5
):
    number_components, component_labels = scipy.sparse.csgraph.connected_components(
        hamiltonian.exchange, directed=False
    )
    component_sizes = np.asarray([np.sum(component_labels == i) for i in range(number_components)])
    order = np.argsort(component_sizes)

    how_many = min(how_many, number_components)
    component_indices = order[-how_many:]

    for k, i in enumerate(component_indices):
        local_hamiltonian, local_spins = extract_local_hamiltonian(
            component_labels == i, hamiltonian, spins
        )

        logger.debug(
            "Is local Hamiltonian frustrated? {}", is_frustrated(local_hamiltonian.exchange)
        )
        # print(local_hamiltonian.exchange)
        # print(local_hamiltonian.field)
        x = anneal_with_retries(local_hamiltonian, number_sweeps=20000, retry=5)

        def extract_signs(bits):
            i = np.arange(local_hamiltonian.shape[0], dtype=np.uint64)
            signs = (bits[i // 64] >> (i % 64)) & 1
            signs = 1 - signs
            signs = torch.from_numpy(signs.view(np.int64)).to(local_spins.device)
            return signs

        predicted_signs = extract_signs(x)
        # print(predicted_signs)
        expected_signs = system.sign(local_spins).argmax(dim=1)
        # print(expected_signs)
        weights = 2 * system.log_amplitude(local_spins)
        if weights.numel() > 1:
            weights.squeeze_(dim=1)
        weights -= torch.max(weights)
        weights = torch.exp_(weights)
        weights /= torch.sum(weights)
        are_same = (predicted_signs == expected_signs).to(weights.dtype)
        overlap = torch.abs(torch.dot(weights, 2 * are_same - 1))
        unweighted_accuracy = torch.mean(are_same)

        logger.debug(
            "№{} largest component contains {} elements", how_many - k, local_hamiltonian.shape[0]
        )
        logger.debug(" Overlap: {}", overlap)
        logger.debug("Accuracy: {}  (unweighted)", unweighted_accuracy)


def main():
    # system = load_rbm_ansatz()
    system = load_ground_state()
    sampled_power = 1.5
    number_samples = 20000 // 64
    spins = monte_carlo_sample(system, number_samples=number_samples, sampled_power=sampled_power)
    h, spins, counts = construct_classical_hamiltonian(system, spins, sampled_power=sampled_power)
    optimize_connected_components(system, h, spins, how_many=5)
    return

    filename = "../scaling/Nqs/Ground/Heisenberg1d_40_1_2.wf"
    logger.debug("Loading RBM weights from '{}' ...", filename)
    log_amplitude_fn = load_rbm(filename)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug("Running on '{}' ...", device)
    log_amplitude_fn.to(device)
    log_coeff_fn = nqs.combine_amplitude_and_phase(
        log_amplitude_fn, MarshallSignRule((1, log_amplitude_fn.in_features), torch.float64)
    )

    # h_flipped = heisenberg1d_model(log_amplitude_fn.in_features, transform_signs=True)
    h_true = heisenberg1d_model(log_amplitude_fn.in_features, transform_signs=False)

    basis = h_true.basis
    sampling_options = nqs.SamplingOptions(
        number_samples=800000,
        number_chains=2,
        number_discarded=10000,
        sweep_size=1,
        mode="zanella",
        device=device,
    )
    spins, _, _, info = nqs.sample_some(log_amplitude_fn, basis, sampling_options)
    spins = spins.view(-1, spins.size(-1))
    # logger.info("Info from the sampler: {}", info)
    local_energies = nqs.local_values(spins, h_true, log_coeff_fn, batch_size=8192)
    local_energies = local_energies.real
    logger.info("Energy: {}", local_energies.mean(dim=0).cpu())
    logger.info("Energy variance: {}", local_energies.var(dim=0).cpu())

    cpu_spins0 = spins.cpu().numpy().view(np.uint64)
    h_classical, cpu_spins, x0, cpu_counts = extract_classical_ising_model(
        cpu_spins0,
        h_true,
        log_coeff_fn,
        sampled_power=2,
        device=device,
        scale_field=0.0,
    )
    x, _, e = sa.anneal(
        h_classical,
        x0=None,
        seed=np.random.randint(1 << 31),
        number_sweeps=5000,
        beta0=None,
        beta1=None,
    )
    spins = torch.from_numpy(cpu_spins.view(np.int64)).to(device)

    msr_fn = MarshallSignRule((1, log_amplitude_fn.in_features), dtype=torch.float64).to(device)
    true_signs = msr_fn(spins).squeeze(dim=1)
    # assert true_signs.dtype == torch.float32
    # print((true_signs / math.pi).to(dtype=torch.int64))
    true_signs = true_signs / math.pi
    true_signs = true_signs.to(dtype=torch.int64)

    def extract_signs(bits):
        i = np.arange(cpu_spins.shape[0], dtype=np.uint64)
        signs = (bits[i // 64] >> (i % 64)) & 1
        signs = 1 - signs
        signs = torch.from_numpy(signs.view(np.int64)).to(device)
        return signs

    signs = extract_signs(x)

    # print(true_signs)
    # print(signs)
    signs0 = extract_signs(x0)
    # # if False:  # NOTE: disabling for now
    mask = (signs == signs0).double()
    accuracy = torch.sum(mask).item() / signs.shape[0]
    logger.debug("Unweighted accuracy: {}", accuracy)
    # cpu_indices = hamiltonian.basis.batched_index(cpu_spins[:, 0])
    # indices = torch.from_numpy(cpu_indices.view(np.int64)).to(spins.device)
    true_weights = 2 * nqs.forward_with_batches(log_amplitude_fn, spins, 16384).squeeze(dim=1)
    true_weights -= torch.max(true_weights)
    true_weights = torch.exp(true_weights)
    true_weights /= torch.sum(true_weights)
    true_overlap = abs(torch.dot(2 * mask - 1, true_weights).item())
    logger.debug("Overlap: {}", true_overlap)

    weights = torch.from_numpy(cpu_counts).to(device=device, dtype=torch.float64)
    weights /= torch.sum(weights)
    overlap = abs(torch.dot(2 * mask - 1, weights).item())
    logger.debug("Overlap: {}", overlap)

    # if overlap < 0:
    #     logger.warning("Applying global sign flip...")
    #     signs = 1 - signs


if __name__ == "__main__":
    main()
