import annealing_sign_problem
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

from sampled_connected_components import magically_compute_local_values


### Code from Physical Review X, 11(4), 041021
# {{{
class Net_2x2x2_dense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dense = torch.nn.Linear(32, 128, bias=True)

    def forward(self, x):
        x = self._dense(x)
        x = torch.log(torch.cosh(x))
        return torch.sum(x, axis=1).view(x.shape[0], 1)


def combine_amplitude_and_phase(*modules) -> torch.nn.Module:
    class CombiningState(torch.nn.Module):
        def __init__(self, amplitude, phase, kx=0, ky=0, kz=0, nx=0, ny=0, nz=0):
            super().__init__()
            self.amplitude = amplitude
            self.phase = phase
            self.amplitude_correction = 1

        def forward(self, x):
            ampl = self.amplitude(x)
            phase = self.phase(x)
            return torch.cat([ampl * self.amplitude_correction, phase], dim=1)

    return CombiningState(*modules)


def logmeanexp(inputs_logampls, inputs_phases, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs_logampls = inputs_logampls.view(-1)
        inputs_phases = inputs_phases.view(-1)
        dim = 0
    # print(inputs_logampls.size(), inputs_phases.size())

    s, _ = torch.max(inputs_logampls, dim=dim, keepdim=True)
    # print(inputs_logampls, s)
    inputs_logampls -= s
    real = torch.sum(torch.exp(inputs_logampls) * torch.cos(inputs_phases), axis=1).unsqueeze(1)
    imag = torch.sum(torch.exp(inputs_logampls) * torch.sin(inputs_phases), axis=1).unsqueeze(1)
    # print('realimag', real.size(), imag.size())
    logampls = (
        torch.log(real ** 2 + imag ** 2) / 2.0
        + s
        - torch.log(torch.ones(real.size(0), device=s.device) * inputs_logampls.size(1)).unsqueeze(
            1
        )
    )
    phases = torch.atan2(imag, real)

    # print('after', logampls.size(), phases.size())
    return logampls, phases


class Net_nonsymmetric_2l_2x2x2(torch.nn.Module):
    def __init__(self, nx: int, ny: int, nz: int):
        super().__init__()
        self.mask_size = (2, 2, 2)
        self._conv1 = torch.nn.Conv3d(
            4, 16, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._conv2 = torch.nn.Conv3d(
            16, 32, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._dense = torch.nn.Linear(32, 1, bias=False)

        self.nx = nx
        self.ny = ny
        self.nz = nz

    def pad_circular(
        self, x
    ):  # x[Nbatch, 1, W, H] -> x[Nbatch, 1, W + 2 pad, H + 2 pad] (pariodic padding)
        if self.nx > 1:
            x = torch.cat([x, x[:, :, 0 : self.mask_size[0] - 1, :, :]], dim=2)
        if self.ny > 1:
            x = torch.cat([x, x[:, :, :, 0 : self.mask_size[1] - 1, :]], dim=3)
        if self.nz > 1:
            x = torch.cat([x, x[:, :, :, :, 0 : self.mask_size[2] - 1]], dim=4)
        return x

    def forward(self, x):
        x = x.view((x.shape[0], 4, self.nx, self.ny, self.nz))
        x = self.pad_circular(x)
        x = self._conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pad_circular(x)
        x = self._conv2(x)
        x = torch.nn.functional.relu(x)
        x = x.view(x.shape[0], 32, -1).mean(dim=2)
        x = self._dense(x)
        return x


class Net_nonsymmetric_1l_2x2x2_narrowing_simplephase(torch.nn.Module):
    def __init__(self, nx: int, ny: int, nz: int):
        super().__init__()
        self.mask_size = (2, 2, 2)
        self._conv = torch.nn.Conv3d(
            4, 32, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._dense = torch.nn.Linear(32, 1, bias=False)

        self.nx = nx
        self.ny = ny
        self.nz = nz

    def pad_circular(
        self, x
    ):  # x[Nbatch, 1, W, H] -> x[Nbatch, 1, W + 2 pad, H + 2 pad] (pariodic padding)
        if self.nx > 1:
            x = torch.cat([x, x[:, :, 0 : self.mask_size[0] - 1, :, :]], dim=2)
        if self.ny > 1:
            x = torch.cat([x, x[:, :, :, 0 : self.mask_size[1] - 1, :]], dim=3)
        if self.nz > 1:
            x = torch.cat([x, x[:, :, :, :, 0 : self.mask_size[2] - 1]], dim=4)

        return x

    def forward(self, x):
        x = x.view((x.shape[0], 4, self.nx, self.ny, self.nz))
        x = self.pad_circular(x)
        x = self._conv(x)
        x = torch.nn.functional.elu(x)
        x = x.view(x.shape[0], 32, -1).mean(dim=2)

        x = self._dense(x)

        return x


class Net_nonsymmetric_3l_2x2x2_narrowing(torch.nn.Module):
    def __init__(self, nx: int, ny: int, nz: int):
        super().__init__()
        self.mask_size = (2, 2, 2)
        self._conv1 = torch.nn.Conv3d(
            4, 16, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._conv2 = torch.nn.Conv3d(
            16, 12, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._conv3 = torch.nn.Conv3d(
            12, 8, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._dense1 = torch.nn.Linear(8, 8, bias=True)
        self._dense2 = torch.nn.Linear(8, 1, bias=True)

        self.nx = nx
        self.ny = ny
        self.nz = nz

    def pad_circular(
        self, x
    ):  # x[Nbatch, 1, W, H] -> x[Nbatch, 1, W + 2 pad, H + 2 pad] (pariodic padding)
        if self.nx > 1:
            x = torch.cat([x, x[:, :, 0 : self.mask_size[0] - 1, :, :]], dim=2)
        if self.ny > 1:
            x = torch.cat([x, x[:, :, :, 0 : self.mask_size[1] - 1, :]], dim=3)
        if self.nz > 1:
            x = torch.cat([x, x[:, :, :, :, 0 : self.mask_size[2] - 1]], dim=4)

        return x

    def forward(self, x):
        x = x.view((x.shape[0], 4, self.nx, self.ny, self.nz))
        x = self.pad_circular(x)
        x = self._conv1(x)
        x = torch.nn.functional.elu(x)
        x = self.pad_circular(x)
        x = self._conv2(x)
        x = torch.nn.functional.elu(x)
        x = self.pad_circular(x)
        x = self._conv3(x)
        x = torch.nn.functional.elu(x)

        x = x.view(x.shape[0], 8, -1).mean(dim=2)

        x = self._dense1(x)
        x = torch.nn.functional.elu(x)
        x = self._dense2(x)

        return x


def combine_amplitude_and_phase_all_2x2x2(*modules) -> torch.nn.Module:
    class CombiningState(torch.nn.Module):
        def __init__(self, amplitude, phase):
            super().__init__()
            self.amplitude = amplitude
            self.phase = phase
            # fmt: off
            M = np.array([24, 25, 27, 26, 29, 28, 30, 31,  8,  9, 11, 10, 13, 12, 14, 15, 16, 17, 19, 18, 21, 20, 22, 23,  0,  1,  3,  2,  5,  4,  6,  7])
            R = np.array([ 0,  2,  4,  6,  1,  3,  5,  7, 24, 26, 28, 30, 25, 27, 29, 31,  8, 10, 12, 14,  9, 11, 13, 15, 16, 18, 20, 22, 17, 19, 21, 23])
            I = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 12, 13, 14, 15,  8,  9, 10, 11, 18, 19, 16, 17, 22, 23, 20, 21, 25, 24, 27, 26, 29, 28, 31, 30])
            xy = np.array([ 0,  2,  1,  3,  4,  6,  5,  7,  8, 10,  9, 11, 12, 14, 13, 15, 24, 26, 25, 27, 28, 30, 29, 31, 16, 18, 17, 19, 20, 22, 21, 23])
            tr_x = np.array([ 4,  5,  6,  7,  0,  1,  2,  3, 12, 13, 14, 15,  8,  9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19, 28, 29, 30, 31, 24, 25, 26, 27])
            tr_y = np.array([ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9, 14, 15, 12, 13, 18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 30, 31, 28, 29])
            tr_z = np.array([ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30])
            # fmt: on

            self.symms = [np.arange(32)]
            self.symms = self.symms + [symm[tr_x] for symm in self.symms]
            self.symms = self.symms + [symm[tr_y] for symm in self.symms]
            self.symms = self.symms + [symm[tr_z] for symm in self.symms]

            self.symms = self.symms + [symm[M] for symm in self.symms]
            self.symms = self.symms + [symm[I] for symm in self.symms]
            self.symms = self.symms + [symm[xy] for symm in self.symms]
            self.symms = (
                self.symms + [symm[R] for symm in self.symms] + [symm[R[R]] for symm in self.symms]
            )

        def forward(self, x):
            ampl_log, phase = logmeanexp(
                torch.cat(
                    [self.amplitude(x[..., symm]) for symm in self.symms]
                    + [self.amplitude(-x[..., symm]) for symm in self.symms],
                    dim=1,
                ),
                torch.cat(
                    [self.phase(x[..., symm]) for symm in self.symms]
                    + [self.phase(-x[..., symm]) + np.pi for symm in self.symms],
                    dim=1,
                ),
                dim=1,
            )
            return torch.cat([ampl_log, phase], dim=1).double()

    return CombiningState(*modules)


# }}}


@torch.no_grad()
def load_unsymmetrized(path="dense_largeMC_symm_0.050_1.000_onlyamplitude.data"):
    model = combine_amplitude_and_phase(Net_2x2x2_dense(), Net_2x2x2_dense())
    # model = combine_amplitude_and_phase_all_2x2x2(Net_2x2x2_dense(), Net_2x2x2_dense())
    model = model.double()
    model.load_state_dict(torch.load(path))
    for name, p in model.named_parameters():
        assert not torch.any(torch.isnan(p))
    return model


@torch.no_grad()
def load_cnn(path="3f_simplephase_0.000_1.000_onlyamplitude.data_200"):
    nx = 2
    ny = 2
    nz = 2
    model = combine_amplitude_and_phase(
        Net_nonsymmetric_3l_2x2x2_narrowing(nx, ny, nz),
        Net_nonsymmetric_1l_2x2x2_narrowing_simplephase(nx, ny, nz),
    )
    # model = combine_amplitude_and_phase_all_2x2x2(Net_2x2x2_dense(), Net_2x2x2_dense())
    model = model.double()
    model.load_state_dict(torch.load(path))
    for name, p in model.named_parameters():
        assert not torch.any(torch.isnan(p))
    return model


@torch.no_grad()
def monte_carlo_sampling(model, hamiltonian):
    @torch.no_grad()
    def log_prob_fn(x):
        # Networks were trained in double precision
        x = unpack_bits.unpack(x, 32).double()
        r = model.amplitude_correction * model.amplitude(x)
        # Not all networks were trained successfully, and some networks give weird outputs...
        min_bound = -500
        max_bound = 500
        if torch.any((r < min_bound) | (r > max_bound)):
            for i in x.size(0):
                if r[i] < min_bound or r[i] > max_bound:
                    raise ValueError("model({}) = {}".format(x[i].cpu().long(), r[i]))
        return r

    @torch.no_grad()
    def log_coeff_fn(x):
        x = unpack_bits.unpack(x, 32).double()
        r = model(x)
        return torch.complex(r[:, 0], r[:, 1])

    def log_coeff_casting(x):
        z = log_coeff_fn(x)
        # r = z.real np.abs(z)
        ϕ = z.imag / np.pi
        ϕ = torch.round(ϕ)
        ϕ = ϕ * np.pi
        z.imag = ϕ
        return z

    spins, log_probs, weights, info = nqs.sample_some(
        log_prob_fn,
        hamiltonian.basis,
        nqs.SamplingOptions(
            number_samples=10,
            number_chains=2,
            sweep_size=30,
            number_discarded=10,
            mode="zanella",
            device=nqs.get_device(model),
        ),
        is_log_prob_fn=True,
    )
    shape = spins.size()
    spins = spins.view(-1, spins.size(-1)).cpu().numpy()
    spins = np.unique(spins, axis=0)
    spins = torch.from_numpy(spins).to(nqs.get_device(model)).view(shape)

    original_local_energies = nqs.local_values(spins, hamiltonian, log_coeff_fn)
    original_local_energies_2 = nqs.local_values(spins, hamiltonian, log_coeff_casting)
    print("with projections: ", torch.mean(original_local_energies_2) / 32 / 4)
    print(original_local_energies_2.view(-1).cpu().real)

    spins = spins.view(-1, spins.size(-1)).cpu()
    weights = weights.view(-1).cpu()
    original_local_energies = original_local_energies.view(-1).cpu()
    return spins, weights, original_local_energies


# @torch.no_grad()
# def optimize_connected_components(
#     system: System, hamiltonian: sa.Hamiltonian, spins: Tensor, how_many: int = 5
# ):
#     number_components, component_labels = scipy.sparse.csgraph.connected_components(
#         hamiltonian.exchange, directed=False
#     )
#     component_sizes = np.asarray([np.sum(component_labels == i) for i in range(number_components)])
#     order = np.argsort(component_sizes)
#
#     how_many = min(how_many, number_components)
#     component_indices = order[-how_many:]
#
#     for k, i in enumerate(component_indices):
#         local_hamiltonian, local_spins = extract_local_hamiltonian(
#             component_labels == i, hamiltonian, spins
#         )
#
#         logger.debug(
#             "Is local Hamiltonian frustrated? {}", is_frustrated(local_hamiltonian.exchange)
#         )
#         # print(local_hamiltonian.exchange)
#         # print(local_hamiltonian.field)
#         x = anneal_with_retries(local_hamiltonian, number_sweeps=20000, retry=5)
#
#         def extract_signs(bits):
#             i = np.arange(local_hamiltonian.shape[0], dtype=np.uint64)
#             signs = (bits[i // 64] >> (i % 64)) & 1
#             signs = 1 - signs
#             signs = torch.from_numpy(signs.view(np.int64)).to(local_spins.device)
#             return signs
#
#         predicted_signs = extract_signs(x)
#         # print(predicted_signs)
#         expected_signs = system.sign(local_spins).argmax(dim=1)
#         # print(expected_signs)
#         weights = 2 * system.log_amplitude(local_spins)
#         if weights.numel() > 1:
#             weights.squeeze_(dim=1)
#         weights -= torch.max(weights)
#         weights = torch.exp_(weights)
#         weights /= torch.sum(weights)
#         are_same = (predicted_signs == expected_signs).to(weights.dtype)
#         overlap = torch.abs(torch.dot(weights, 2 * are_same - 1))
#         unweighted_accuracy = torch.mean(are_same)
#
#         logger.debug(
#             "№{} largest component contains {} elements", how_many - k, local_hamiltonian.shape[0]
#         )
#         logger.debug(" Overlap: {}", overlap)
#         logger.debug("Accuracy: {}  (unweighted)", unweighted_accuracy)


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


def _reference_log_apply_one(spin, operator, log_psi, device):
    spins, coeffs = operator.apply(spin)
    spins = torch.from_numpy(spins.view(np.int64)).to(device)
    output = log_psi(spins)
    coeffs = torch.from_numpy(coeffs).to(device=device)
    output = output.to(dtype=coeffs.dtype)
    if output.dim() > 1:
        output.squeeze_(dim=1)
    scale = torch.max(output.real)
    output.real -= scale
    torch.exp_(output)
    # coeffs = torch.from_numpy(coeffs).to(device=device, dtype=output.dtype)
    return scale + torch.log(torch.dot(coeffs, output))


def project_to_real_fn(log_coeff_fn):

    @torch.no_grad()
    def log_coeff_projecting(x):
        z = log_coeff_fn(x)
        ϕ = z.imag / np.pi
        if isinstance(ϕ, Tensor):
            ϕ = torch.round(ϕ)
        else:
            ϕ = np.round(ϕ)
        ϕ = ϕ * np.pi
        z.imag = ϕ
        return z

    return log_coeff_projecting

def build_clusters_naive(spins, hamiltonian, log_coeff_fn):
    spins = np.asarray(spins, dtype=np.uint64, order="C")
    if spins.ndim == 1:
        spins = np.hstack([spins.reshape(-1, 1), np.zeros((spins.shape[0], 7), dtype=np.uint64)])
    elif spins.ndim == 2:
        if spins.shape[1] != 8:
            raise ValueError("'spins' has wrong shape: {}; expected (?, 8)".format(spins.shape))
        spins = np.ascontiguousarray(spins)
    else:
        raise ValueError("'spins' has wrong shape: {}; expected a 2D array".format(spins.shape))
    # If spins come from Monte Carlo sampling, there might be duplicates.
    spins, counts = np.unique(spins, return_counts=True, axis=0)
    other_spins, other_coeffs, other_counts = hamiltonian.batched_apply(spins)

    combined_spins = np.vstack((spins, other_spins))
    combined_spins = np.unique(combined_spins, return_counts=False, axis=0)

    def log_coeff_casting(x):
        z = log_coeff_fn(x)
        # r = z.real np.abs(z)
        ϕ = z.imag / np.pi
        ϕ = np.round(ϕ)
        ϕ = ϕ * np.pi
        z.imag = ϕ
        return z

    h, spins0, x0, counts = annealing_sign_problem.extract_classical_ising_model(
        combined_spins, hamiltonian, log_coeff_casting, monte_carlo_weights=None, scale_field=0
    )
    number_components, component_labels = connected_components(h.exchange, directed=False)
    component_sizes = np.asarray([np.sum(component_labels == i) for i in range(number_components)])
    print(sorted(component_sizes))

    for i in range(1):
        s = spins[i, 0]
        spin_index = np.searchsorted(combined_spins[:, 0], s)
        assert np.all(spins[i] == combined_spins[spin_index])
        component_index = component_labels[spin_index]
        local_hamiltonian, local_spins = extract_local_hamiltonian(
            component_labels == component_index, h, combined_spins
        )

        logger.debug(
            "Is local Hamiltonian frustrated? {}", is_frustrated(local_hamiltonian.exchange)
        )

        x, e = sa.anneal(
            local_hamiltonian,
            seed=None,
            number_sweeps=5000,
            repetitions=20,
            only_best=True,
        )
        print(e)
        predicted_signs = annealing_sign_problem.extract_signs_from_bits(
            x, number_spins=local_spins.shape[0]
        )

        current_log_psi = log_coeff_fn(spins[i].reshape(1, -1))
        k = np.searchsorted(local_spins[:, 0], s)
        assert np.all(local_spins[k] == spins[i])
        current_sign = predicted_signs[k]

        neighborhood_spins, neighborhood_coeffs = hamiltonian.apply(s)
        neighborhood_log_psi = log_coeff_fn(neighborhood_spins)

        neighborhood_signs = []
        for σ in neighborhood_spins:
            k = np.searchsorted(local_spins[:, 0], σ[0])
            assert np.all(local_spins[k] == σ)
            neighborhood_signs.append(predicted_signs[k])
        neighborhood_signs = np.asarray(neighborhood_signs)

        local_energy = np.sum(np.exp(np.abs(neighborhood_log_psi) - np.abs(current_log_psi)) * neighborhood_signs / current_sign)
        print(local_energy)
        # def extract_signs(bits):
        #     i = np.arange(local_hamiltonian.shape[0], dtype=np.uint64)
        #     signs = (bits[i // 64] >> (i % 64)) & 1
        #     signs = 1 - signs
        #     signs = torch.from_numpy(signs.view(np.int64)).to(local_spins.device)
        #     return signs

        # predicted_signs = extract_signs(x)
        # # print(predicted_signs)
        # expected_signs = system.sign(local_spins).argmax(dim=1)
        # # print(expected_signs)
        # weights = 2 * system.log_amplitude(local_spins)
        # if weights.numel() > 1:
        #     weights.squeeze_(dim=1)
        # weights -= torch.max(weights)
        # weights = torch.exp_(weights)
        # weights /= torch.sum(weights)
        # are_same = (predicted_signs == expected_signs).to(weights.dtype)
        # overlap = torch.abs(torch.dot(weights, 2 * are_same - 1))
        # unweighted_accuracy = torch.mean(are_same)

        # logger.debug(
        #     "№{} largest component contains {} elements", how_many - k, local_hamiltonian.shape[0]
        # )
        # logger.debug(" Overlap: {}", overlap)
        # logger.debug("Accuracy: {}  (unweighted)", unweighted_accuracy)


@torch.no_grad()
def establish_baseline():
    yaml_path = "../physical_systems/heisenberg_pyrochlore_2x2x2_no_symmetries.yaml"
    hamiltonian = nqs.load_hamiltonian(yaml_path)
    # model = load_unsymmetrized()
    model = load_cnn()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda:0")
    for name, p in model.named_parameters():
        print(name, p.device)

    # def log_amplitude_fn(x):
    #     x = unpack_bits.unpack(x, 32).double()
    #     r = model(x)[:, 0]
    #     # if not torch.all(r < 100):
    #     #     print(r)
    #     #     assert False
    #     # if not torch.all(r > -100):
    #     #     print(r)
    #     #     assert False
    #     # assert torch.all(r < 100)
    #     # assert torch.all(r > -100)
    #     return r

    spins, weights, original_local_energies = monte_carlo_sampling(model, hamiltonian)
    print(torch.mean(original_local_energies).item() / 32 / 4)

    # hamiltonian.basis.build()
    # spins, log_probs, weights, info = nqs.sample_some(
    #     log_amplitude_fn,
    #     hamiltonian.basis,
    #     nqs.SamplingOptions(
    #         number_samples=200,
    #         number_chains=2,
    #         sweep_size=5,
    #         number_discarded=10,
    #         mode="zanella",
    #         device=device,
    #     ),
    # )
    # (np.sin(0.025) * np.array([[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]])).tolist()

    @torch.no_grad()
    def log_coeff_fn(x):
        should_cast = False
        if not isinstance(x, Tensor):
            should_cast = True
            if x.dtype == np.uint64:
                x = x.astype(np.int64)
            x = torch.from_numpy(x).to(device)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        def f(y):
            y = unpack_bits.unpack(y, 32).double()
            r = model(y)
            r = torch.complex(r[:, 0], r[:, 1])
            return r.cpu()

        r = nqs.forward_with_batches(f, x, batch_size=10000)
        # x = unpack_bits.unpack(x, 32).double()
        # r = model(x)
        # r = torch.complex(r[:, 0], r[:, 1])
        if should_cast:
            r = r.cpu().numpy()
        return r

    def log_coeff_casting_fn(x):
        z = log_coeff_fn(x)
        # r = z.real np.abs(z)
        ϕ = z.imag / np.pi
        if isinstance(ϕ, Tensor):
            ϕ = torch.round(ϕ)
        else:
            ϕ = np.round(ϕ)
        ϕ = ϕ * np.pi
        z.imag = ϕ
        return z

    e = magically_compute_local_values(spins, hamiltonian, log_coeff_casting_fn, cutoff=1e-3)
    print(e)
    print(np.mean(e) / 4 / 32)
    print(np.std(e) / 4 / 32)
    # build_clusters_naive(spins, hamiltonian, log_coeff_fn)
    # local_energies = nqs.local_values(spins, hamiltonian, log_coeff_fn)
    # print(local_energies)
    # print(torch.mean(local_energies) / 4 / 32)
    # print(torch.std(local_energies) / 4 / 32)


if __name__ == "__main__":
    establish_baseline()
    # print(load_unsymmetrized())
