import lattice_symmetries as ls
import ising_glass_annealer as sa
import unpack_bits
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter  # , UninitializedParameter
import numpy as np
import scipy.sparse
import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from loguru import logger

# from ctypes import CDLL, POINTER, c_int64, c_uint32, c_uint64, c_double
import yaml
import h5py
import os
import _build_matrix
from _build_matrix import ffi


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


def get_device(obj) -> Optional[torch.device]:
    r = _get_a_var(obj)
    return r.device if r is not None else None


def get_dtype(obj) -> Optional[torch.dtype]:
    r = _get_a_var(obj)
    return r.dtype if r is not None else None


def _get_a_var(obj):
    if isinstance(obj, Tensor):
        return obj
    if isinstance(obj, torch.nn.Module):
        for result in obj.parameters():
            if isinstance(result, Tensor):
                return result
    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, Tensor):
                return result
    return None


def split_into_batches(xs: Tensor, batch_size: int, device=None):
    r"""Iterate over `xs` in batches of size `batch_size`. If `device` is not `None`, batches are
    moved to `device`.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("invalid batch_size: {}; expected a positive integer".format(batch_size))

    expanded = False
    if isinstance(xs, (np.ndarray, Tensor)):
        xs = (xs,)
        expanded = True
    else:
        assert isinstance(xs, (tuple, list))
    n = xs[0].shape[0]
    if any(filter(lambda x: x.shape[0] != n, xs)):
        raise ValueError("tensors 'xs' must all have the same batch dimension")
    if n == 0:
        return None

    i = 0
    while i + batch_size <= n:
        chunks = tuple(x[i : i + batch_size] for x in xs)
        if device is not None:
            chunks = tuple(chunk.to(device) for chunk in chunks)
        if expanded:
            chunks = chunks[0]
        yield chunks
        i += batch_size
    if i != n:  # Remaining part
        chunks = tuple(x[i:] for x in xs)
        if device is not None:
            chunks = tuple(chunk.to(device) for chunk in chunks)
        if expanded:
            chunks = chunks[0]
        yield chunks


def forward_with_batches(f, xs, batch_size: int, device=None) -> Tensor:
    r"""Applies ``f`` to all ``xs`` propagating no more than ``batch_size``
    samples at a time. ``xs`` is split into batches along the first dimension
    (i.e. dim=0). ``f`` must return a torch.Tensor.
    """
    if xs.shape[0] == 0:
        raise ValueError("invalid xs: {}; input should not be empty".format(xs))
    out = []
    for chunk in split_into_batches(xs, batch_size, device):
        out.append(f(chunk))
    return torch.cat(out, dim=0)

class Unpack(torch.nn.Module):
    r"""Unpacks spin configurations represented as bits (`uint64_t` or `ls_bits512`) into a
    2D-tensor of `float32`.
    """
    __constants__ = ["number_spins"]
    number_spins: int

    def __init__(self, number_spins: int):
        super().__init__()
        self.number_spins = number_spins

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(dim=1)
        return unpack_bits.unpack(x, self.number_spins)

    def extra_repr(self) -> str:
        return "number_spins={}".format(self.number_spins)


class TensorIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, *tensors, batch_size=1, shuffle=False):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        assert all(tensors[0].device == tensor.device for tensor in tensors)
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle

    @property
    def device(self):
        return self.tensors[0].device

    def __len__(self):
        return self.tensors[0].size(0)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.tensors[0].size(0), device=self.device)
            tensors = tuple(tensor[indices] for tensor in self.tensors)
        else:
            tensors = self.tensors
        return zip(*(torch.split(tensor, self.batch_size) for tensor in tensors))


def supervised_loop_once(
    dataset: Iterable[Tuple[Tensor, Tensor, Tensor]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    scheduler: Optional[Any],
    swa_model=None,
    swa_scheduler=None,
) -> Dict[str, float]:
    tick = time.time()
    model.train()
    total_loss: float = 0.0
    total_count: int = 0
    for batch in dataset:
        (x, y, w) = batch
        w = w / torch.sum(w)
        optimizer.zero_grad()
        ŷ = model(x)
        loss = loss_fn(ŷ, y, w)
        loss.backward()
        optimizer.step()
        total_loss += x.size(0) * loss.item()
        total_count += x.size(0)
    if scheduler is not None:
        scheduler.step()
    if swa_model is not None:
        assert scheduler is None
        assert swa_scheduler is not None
        swa_model.update_parameters(model)
        swa_scheduler.step()
    tock = time.time()
    return {"loss": total_loss / total_count, "time": tock - tick}


@torch.no_grad()
def compute_average_loss(dataset, model, loss_fn, accuracy_fn):
    tick = time.time()
    model.eval()
    total_loss = 0
    total_sum = 0
    total_count = 0
    for batch in dataset:
        (x, y, w) = batch
        w = w / torch.sum(w)
        ŷ = model(x)
        loss = loss_fn(ŷ, y, w)
        accuracy = accuracy_fn(ŷ, y, w)
        total_loss += x.size(0) * loss.item()
        total_sum += x.size(0) * accuracy.item()
        total_count += x.size(0)
    tock = time.time()
    return {
        "loss": total_loss / total_count,
        "accuracy": total_sum / total_count,
        "time": tock - tick,
    }


def extract_classical_ising_model(
    spins,
    hamiltonian,
    log_ψ,
    sampled_power: Optional[int] = None,
    device: Optional[torch.device] = None,
    scale_field: Optional[float] = None,
):
    r"""Map quantum Hamiltonian to classical ising model where wavefunction coefficients are now
    considered spin degrees of freedom.
    """
    logger.debug("Constructing classical Ising model...")
    spins = np.asarray(spins, dtype=np.uint64, order="C")
    if spins.ndim == 1:
        spins = np.hstack([spins.reshape(-1, 1), np.zeros((spins.shape[0], 7), dtype=np.uint64)])
    elif spins.ndim == 2:
        if spins.shape[1] != 8:
            raise ValueError("'x' has wrong shape: {}; expected (?, 8)".format(x.shape))
        spins = np.ascontiguousarray(spins)
    else:
        raise ValueError("'x' has wrong shape: {}; expected a 2D array".format(x.shape))
    # If spins come from Monte Carlo sampling, it might contains duplicates.
    spins, counts = np.unique(spins, return_counts=True, axis=0)
    if sampled_power is None and np.any(counts != 1):
        raise ValueError("'spins' contains duplicate spin configurations, but sampled_power=None")

    @torch.no_grad()
    def forward(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.view(np.int64))
            if device is not None:
                x = x.to(device)
        r = forward_with_batches(log_ψ, x, batch_size=10240)
        if r.numel() > 1:
            r.squeeze_(dim=1)
        return r.cpu().numpy()

    ψs = forward(spins)
    other_spins, other_coeffs, other_counts = hamiltonian.batched_apply(spins)
    assert np.all(other_counts > 0)
    if not np.allclose(other_coeffs.imag, 0):
        raise ValueError("expected all matrix elements to be real")
    other_coeffs = np.ascontiguousarray(other_coeffs.real)
    other_ψs = forward(other_spins)

    scale = np.max(other_ψs.real)
    other_ψs.real -= scale
    other_ψs = np.exp(other_ψs, dtype=np.complex128)
    if not np.allclose(other_ψs.imag, 0, atol=1e-6):
        raise ValueError("expected all wavefunction coefficients to be real")
    other_ψs = np.ascontiguousarray(other_ψs.real)

    ψs.real -= scale
    ψs = np.exp(ψs, dtype=np.complex128)
    if not np.allclose(ψs.imag, 0, atol=1e-6):
        raise ValueError("expected all wavefunction coefficients to be real")
    ψs = np.ascontiguousarray(ψs.real)

    if sampled_power is None:
        normalization = 1 / np.linalg.norm(ψs)
    else:
        normalization = 1 / np.sqrt(np.dot(counts, np.abs(ψs) ** (2 - sampled_power)))
        ψs = np.sign(ψs) * np.abs(ψs) ** (1 - sampled_power)
    ψs *= normalization
    other_ψs *= normalization

    field = np.zeros(spins.shape[0], dtype=np.float64)
    row_indices = np.empty(other_spins.shape[0], dtype=np.uint32)
    col_indices = np.empty(other_spins.shape[0], dtype=np.uint32)
    elements = np.empty(other_spins.shape[0], dtype=np.float64)
    # uint64_t build_matrix(
    #     uint64_t num_spins, ls_bits512 const spins[],
    #     int64_t const * counts, double const * psi,
    #     ls_bits512 const * other_spins, double const * other_coeffs,
    #     int64_t const * other_counts, double const * other_psi,
    #     uint32_t * row_indices, uint32_t * col_indices,
    #     double * elements, double * field);
    written = _build_matrix.lib.build_matrix(
        spins.shape[0],
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
        ffi.from_buffer("double[]", field)
        # spins.ctypes.data_as(POINTER(c_uint64)),
        # counts.ctypes.data_as(POINTER(c_int64)),
        # ψs.ctypes.data_as(POINTER(c_double)),
        # other_spins.ctypes.data_as(POINTER(c_uint64)),
        # other_coeffs.ctypes.data_as(POINTER(c_double)),
        # other_counts.ctypes.data_as(POINTER(c_int64)),
        # other_ψs.ctypes.data_as(POINTER(c_double)),
        # row_indices.ctypes.data_as(POINTER(c_uint32)),
        # col_indices.ctypes.data_as(POINTER(c_uint32)),
        # elements.ctypes.data_as(POINTER(c_double)),
        # field.ctypes.data_as(POINTER(c_double)),
    )
    row_indices = row_indices[:written]
    col_indices = col_indices[:written]
    elements = elements[:written]

    matrix = scipy.sparse.coo_matrix(
        (elements, (row_indices, col_indices)), shape=(spins.shape[0], spins.shape[0]),
    )
    # Convert COO matrix to CSR to ensure that duplicate elements are summed
    # together. Duplicate elements can arise when working in symmetry-adapted
    # bases.
    matrix = matrix.tocsr()
    # Symmetrize the matrix if spins come from Monte Carlo sampling. When
    # matrix elements do not come from Monte Carlo sampling, symmetrizing does
    # not hurt and may fix some numerical differences.
    matrix = 0.5 * (matrix + matrix.T)
    matrix = matrix.tocoo()

    logger.debug("Denseness: {}", np.sum(np.abs(matrix.data)) / spins.shape[0])

    if scale_field is not None:
        field *= scale_field
    h = sa.Hamiltonian(matrix, field)

    # print("Max field", field.max(), np.abs(field).max())
    # print("Min field", field.min(), np.abs(field).min())
    # print("Max coupling", matrix.data.max(), np.abs(matrix.data).max())
    # print("Min coupling", matrix.data.min(), np.abs(matrix.data).min())

    x0 = np.empty((spins.shape[0] + 63) // 64, dtype=np.uint64)
    # void extract_signs(
    #     uint64_t num_spins, double const * psi, uint64_t * signs);
    _build_matrix.lib.extract_signs(
        spins.shape[0],
        ffi.from_buffer("double const[]", ψs),
        ffi.from_buffer("uint64_t const[]", x0),
        # ψs.ctypes.data_as(POINTER(c_double)),
        # x0.ctypes.data_as(POINTER(c_uint64)),
    )

    logger.debug(
        "Done! The Hamiltonian has dimension {} and contains {} non-zero elements.",
        spins.shape[0],
        written,
    )
    logger.debug("Jₘᵢₙ = {}, Jₘₐₓ = {}", np.abs(matrix.data).min(), np.abs(matrix.data).max())
    logger.debug("Bₘᵢₙ = {}, Bₘₐₓ = {}", np.abs(field).min(), np.abs(field).max())

    return h, spins, x0, counts


def load_ground_state(filename: str):
    with h5py.File(filename, "r") as f:
        ground_state = f["/hamiltonian/eigenvectors"][:]
        ground_state = ground_state.squeeze()
        if ground_state.ndim > 1:
            ground_state = ground_state[0, :]
        energy = f["/hamiltonian/eigenvalues"][0]
        basis_representatives = f["/basis/representatives"][:]
    return torch.from_numpy(ground_state), energy, basis_representatives


# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     # scale = torch.diag(mx.sum(dim=1).reciprocal()).to_sparse()
#     # return torch.mm(scale, mx)
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.0
#     r_mat_inv = scipy.sparse.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx


# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)


# def load_graph(filename: str):
#     with open(filename, "r") as f:
#         config = yaml.load(f, Loader=yaml.SafeLoader)
#     n = config["basis"]["number_spins"]
#     edges = np.array(config["hamiltonian"]["terms"][0]["sites"])
#     edges = torch.from_numpy(edges)
#     edges = torch.cat([edges, torch.stack([edges[:, 1], edges[:, 0]], dim=1)], dim=0)
#     edges = edges.t().contiguous()
#     return edges
#     # adj = torch.diag()
#     # print(adj)
#     # adj = torch.zeros((n, n))
#     edges = []
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 edges.append((i, j))
#         # x = i % 6
#         # y = i // 6
#         # edges.append((i, (x - 2 + 6) % 6 + ((y + 2) % 6) * 6))
#         # edges.append((i, (x - 1 + 6) % 6 + ((y + 2) % 6) * 6))
#         # edges.append((i, (x - 0 + 6) % 6 + ((y + 2) % 6) * 6))
# 
#         # edges.append((i, (x - 2 + 6) % 6 + ((y + 1) % 6) * 6))
#         # edges.append((i, (x - 1 + 6) % 6 + ((y + 1) % 6) * 6))
#         # edges.append((i, (x - 0 + 6) % 6 + ((y + 1) % 6) * 6))
#         # edges.append((i, (x + 1 + 6) % 6 + ((y + 1) % 6) * 6))
# 
#         # edges.append((i, (x - 2 + 6) % 6 + ((y + 0) % 6) * 6))
#         # edges.append((i, (x - 1 + 6) % 6 + ((y + 0) % 6) * 6))
#         # # edges.append((i, (x - 0 + 6) % 6 + ((y + 0) % 6) * 6))
#         # edges.append((i, (x + 1 + 6) % 6 + ((y + 0) % 6) * 6))
#         # edges.append((i, (x + 2 + 6) % 6 + ((y + 0) % 6) * 6))
# 
#         # edges.append((i, (x - 1 + 6) % 6 + ((y - 1 + 6) % 6) * 6))
#         # edges.append((i, (x - 0 + 6) % 6 + ((y - 1 + 6) % 6) * 6))
#         # edges.append((i, (x + 1 + 6) % 6 + ((y - 1 + 6) % 6) * 6))
#         # edges.append((i, (x + 2 + 6) % 6 + ((y - 1 + 6) % 6) * 6))
# 
#         # edges.append((i, (x - 0 + 6) % 6 + ((y - 2 + 6) % 6) * 6))
#         # edges.append((i, (x + 1 + 6) % 6 + ((y - 2 + 6) % 6) * 6))
#         # edges.append((i, (x + 2 + 6) % 6 + ((y - 2 + 6) % 6) * 6))
# 
#     adj = torch.zeros((n, n))
#     for i in range(len(edges)):
#         adj[edges[i][0], edges[i][1]] = 1
#     # adj = adj + adj.t() + torch.eye(n)
#     print(adj.sum(dim=1))
#     assert torch.all(adj == adj.t())
#     return adj
#     # edges = torch.cat([edges, torch.stack([edges[:, 1], edges[:, 0]], dim=1)], dim=0).to(torch.long)
#     # return edges.t().contiguous()
#     # return torch.from_numpy(edges.T).contiguous().to(torch.long)
#     adj = scipy.sparse.coo_matrix(
#         (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n)
#     )
#     adj = adj + adj.T + scipy.sparse.eye(n)
#     adj = normalize(adj).tocoo().astype(np.float32)
#     return torch.sparse_coo_tensor(
#         torch.from_numpy(np.vstack((adj.row, adj.col))), torch.from_numpy(adj.data), (n, n)
#     ).coalesce()


def load_basis_and_hamiltonian(filename: str):
    with open(filename, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    basis = ls.SpinBasis.load_from_yaml(config["basis"])
    hamiltonian = ls.Operator.load_from_yaml(config["hamiltonian"], basis)
    return basis, hamiltonian


def make_log_coeff_fn(ground_state: np.ndarray, basis):
    log_amplitudes = ground_state.abs().log_().unsqueeze(dim=1)
    phases = torch.where(
        ground_state >= 0,
        torch.scalar_tensor(0.0, dtype=ground_state.dtype),
        torch.scalar_tensor(np.pi, dtype=ground_state.dtype),
    ).unsqueeze(dim=1)

    @torch.no_grad()
    def log_coeff_fn(spin: Tensor) -> Tensor:
        if not isinstance(spin, np.ndarray):
            spin = spin.numpy().view(np.uint64)
        if spin.ndim > 1:
            spin = spin[:, 0]
        indices = torch.from_numpy(ls.batched_index(basis, spin).view(np.int64))
        a = log_amplitudes[indices]
        b = phases[indices]
        return torch.complex(a, b)

    return log_coeff_fn


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
