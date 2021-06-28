from .common import *
import ctypes
from collections import namedtuple
from typing import Tuple
import lattice_symmetries as ls
import nqs_playground as nqs
from nqs_playground.core import _get_dtype, _get_device
import torch
from torch import Tensor
from typing import Optional
import time
from loguru import logger
import sys
import numpy as np


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

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.tensors[0].size(0), device=self.device)
            tensors = tuple(tensor[indices] for tensor in self.tensors)
        else:
            tensors = self.tensors
        return zip(*(torch.split(tensor, self.batch_size) for tensor in tensors))


def supervised_loop_once(dataset, model, optimizer, scheduler, loss_fn):
    tick = time.time()
    model.train()
    total_loss = 0
    total_count = 0
    for batch in dataset:
        (x, y, w) = batch
        w = w / torch.sum(w)
        optimizer.zero_grad()
        ŷ = model(x)
        loss = loss_fn(ŷ, y, w)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += x.size(0) * loss.item()
        total_count += x.size(0)
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


def prepare_datasets(
    spins,
    target_signs,
    weights,
    train_batch_size: int,
    inference_batch_size: int = 16384,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    spins = spins.to(device)
    target_signs = target_signs.to(device)
    # target_signs = torch.where(
    #     target_signs > 0,
    #     torch.scalar_tensor(0, dtype=torch.int64, device=device),
    #     torch.scalar_tensor(1, dtype=torch.int64, device=device),
    # )
    if weights is None:
        weights = torch.ones_like(target_signs, device=device, dtype=dtype)
    data = (spins, target_signs, weights)
    train_dataset = TensorIterableDataset(*data, batch_size=train_batch_size, shuffle=True)
    test_dataset = TensorIterableDataset(*data, batch_size=inference_batch_size, shuffle=False)
    return train_dataset, test_dataset


def tune_neural_network(
    model: torch.nn.Module,
    spins: Tensor,
    target_signs: Tensor,
    weights: Optional[Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    batch_size: int,
):
    logger.info("Starting supervised training...")
    dtype = _get_dtype(model)
    device = _get_device(model)
    train_dataset, test_dataset = prepare_datasets(
        spins, target_signs, weights, train_batch_size=batch_size, device=device, dtype=dtype
    )
    logger.info("Training dataset contains {} samples", spins.size(0))

    def loss_fn(ŷ, y, w):
        """Weighted Cross Entropy"""
        r = torch.nn.functional.cross_entropy(ŷ, y, reduction="none")
        return torch.dot(r, w)

    @torch.no_grad()
    def accuracy_fn(ŷ, y, w):
        r = (ŷ.argmax(dim=1) == y).to(w.dtype)
        return torch.dot(r, w)

    info = compute_average_loss(test_dataset, model, loss_fn, accuracy_fn)
    logger.debug("[0/{}]: loss = {}, accuracy = {}", epochs, info["loss"], info["accuracy"])
    # if info["accuracy"] < 0.2:
    #     target_signs = 1 - target_signs
    #     train_dataset, test_dataset = prepare_datasets(
    #         spins, target_signs, weights, train_batch_size=batch_size, device=device, dtype=dtype
    #     )
    #     info = compute_average_loss(test_dataset, model, loss_fn, accuracy_fn)
    #     logger.debug("[0/{}]: loss = {}, accuracy = {}", epochs, info["loss"], info["accuracy"])
    for epoch in range(epochs):
        info = supervised_loop_once(train_dataset, model, optimizer, scheduler, loss_fn)
        if epoch < epochs - 1:
            logger.debug("[{}/{}]: loss = {}", epoch + 1, epochs, info["loss"])
    info = compute_average_loss(test_dataset, model, loss_fn, accuracy_fn)
    logger.debug(
        "[{}/{}]: loss = {}, accuracy = {}", epochs, epochs, info["loss"], info["accuracy"]
    )


def tune_sign_structure(
    spins: np.ndarray,
    hamiltonian: ls.Operator,
    log_psi: torch.nn.Module,
    number_sweeps: int,
    seed: Optional[int] = None,
    beta0: Optional[int] = None,
    beta1: Optional[int] = None,
    sampled_power: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    h, spins, x0, counts = extract_classical_ising_model(
        spins, hamiltonian, log_psi, sampled_power=sampled_power, device=device
    )
    beta0 = 6.0 # TODO: Fix me!!
    beta1 = 10000.0
    x, _, _ = sa.anneal(h, x0, seed=seed, number_sweeps=number_sweeps, beta0=beta0, beta1=beta1)
    # x2, _, e2 = sa.anneal(h, ~x0, seed=None, number_sweeps=number_sweeps, beta0=beta0, beta1=beta1)
    # logger.info("{} vs. {}, {} vs. {}", e1[-1], e2[-1], x0 & x1, x0 & x2)
    # x = x1
    # _, _, _ = sa.anneal(h, x, seed=None, number_sweeps=number_sweeps, beta0=beta0, beta1=beta1)
    # _, _, _ = sa.anneal(h, ~x, seed=None, number_sweeps=number_sweeps, beta0=beta0, beta1=beta1)
    i = np.arange(spins.shape[0], dtype=np.uint64)
    signs = (x[i // 64] >> (i % 64)) & 1
    signs = 1 - signs
    signs = torch.from_numpy(signs.view(np.int64))
    return spins, signs, counts


Config = namedtuple(
    "Config",
    [
        "model",
        "ground_state",
        "hamiltonian",
        "number_sa_sweeps",
        "number_supervised_epochs",
        "number_monte_carlo_samples",
        "number_outer_iterations",
        "train_batch_size",
        "optimizer",
        "scheduler",
        "device",
    ],
)


def _make_log_coeff_fn(amplitude, sign, basis):
    assert isinstance(amplitude, Tensor)
    assert isinstance(sign, torch.nn.Module)

    dtype = _get_dtype(sign)
    device = _get_device(sign)
    log_amplitude = amplitude.log().unsqueeze(dim=1)
    log_amplitude = log_amplitude.to(device=device, dtype=dtype)

    @torch.no_grad()
    def log_coeff_fn(spin: Tensor) -> Tensor:
        b = np.pi * sign(spin).argmax(dim=1, keepdim=True).to(dtype)
        spin = spin.cpu().numpy().view(np.uint64)
        if spin.ndim > 1:
            spin = spin[:, 0]
        indices = torch.from_numpy(ls.batched_index(basis, spin).view(np.int64)).to(device)
        a = log_amplitude[indices]
        return torch.complex(a, b)

    return log_coeff_fn


def find_ground_state(config):
    hamiltonian = config.hamiltonian
    basis = hamiltonian.basis
    # all_spins = torch.from_numpy(basis.states.view(np.int64))
    # correct_sign_structure = torch.where(
    #     ground_state > 0.0,
    #     torch.scalar_tensor(1.0, dtype=ground_state.dtype),
    #     torch.scalar_tensor(-1.0, dtype=ground_state.dtype),
    # )

    log_psi = _make_log_coeff_fn(config.ground_state.abs(), config.model, basis)
    sampled_power = 2  # True
    with torch.no_grad():
        p = config.ground_state.abs() ** sampled_power
        p = p.numpy()
        p /= np.sum(p)
    dtype = _get_dtype(config.model)
    device = _get_device(config.model)
    for i in range(config.number_outer_iterations):
        logger.info("Starting outer iteration {}...", i + 1)
        if sampled_power is not None:
            batch_indices = np.random.choice(
                basis.number_states, size=config.number_monte_carlo_samples, replace=True, p=p
            )
        else:
            batch_indices = np.random.choice(
                basis.number_states, size=config.number_monte_carlo_samples, replace=False, p=None
            )

        spins = basis.states[batch_indices]
        spins, signs, counts = tune_sign_structure(
            spins,
            hamiltonian,
            log_psi,
            number_sweeps=config.number_sa_sweeps,
            sampled_power=sampled_power,
            device=device,
        )
        with torch.no_grad():
            if sampled_power is not None:
                weights = torch.from_numpy(counts).to(dtype=dtype, device=device)
                weights /= torch.sum(weights)
            else:
                assert np.all(counts == 1)
                weights = torch.from_numpy(p[batch_indices]).to(dtype=dtype, device=device)
                weights /= torch.sum(weights)
        spins = torch.from_numpy(spins.view(np.int64))
        tune_neural_network(
            config.model,
            spins,
            signs,
            weights,
            optimizer=config.optimizer(config.model),
            scheduler=config.scheduler(config.model),
            epochs=config.number_supervised_epochs,
            batch_size=config.train_batch_size,
        )

    # return 1 - np.abs(get_overlap()), best_energy


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.tanh(torch.nn.functional.softplus(input))


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=0,
            ),
            Mish(),
            # torch.nn.MaxPool1d(kernel_size=2),
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        k = self.kernel_size
        x = torch.cat([x[:, :, -k // 2 :, :], x, x[:, :, : k // 2, :]], dim=2)
        x = torch.cat([x[:, :, :, -k // 2 :], x, x[:, :, :, : k // 2]], dim=3)
        return self.layer(x)


class Phase(torch.nn.Module):
    def __init__(self, shape, number_channels, kernel_size=3):
        super().__init__()
        number_blocks = len(number_channels)
        layers = [ConvBlock(in_channels=1, out_channels=number_channels[0], kernel_size=kernel_size)]
        for i in range(1, len(number_channels)):
            layers.append(
                ConvBlock(
                    in_channels=number_channels[i - 1],
                    out_channels=number_channels[i],
                    kernel_size=kernel_size,
                )
            )
        self.shape = shape
        self.layers = torch.nn.Sequential(*layers)
        self.tail = torch.nn.Linear(number_channels[-1], 2, bias=False)

    def forward(self, x):
        number_spins = self.shape[0] * self.shape[1]
        x = nqs.unpack(x, number_spins)
        x = x.view(x.size(0), 1, *self.shape)
        x = self.layers(x)
        x = x.view(*x.size()[:2], -1).sum(dim=2)
        return self.tail(x)


def main():
    ground_state, E, representatives = load_ground_state(
        os.path.join(project_dir(), "data/symm/j1j2_square_6x6.h5")
    )
    basis, hamiltonian = load_basis_and_hamiltonian(
        os.path.join(project_dir(), "data/symm/j1j2_square_6x6.yaml")
    )
    basis.build(representatives)
    representatives = None

    torch.manual_seed(123)
    np.random.seed(127)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Phase((6, 6), number_channels=[28, 28, 20], kernel_size=5).to(device)
    # model = torch.nn.Sequential(
    #     nqs.Unpack(basis.number_spins),
    #     torch.nn.Linear(basis.number_spins, 64),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(64, 64),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(64, 2, bias=False),
    # )

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = Config(
        model=model,
        ground_state=ground_state,
        hamiltonian=hamiltonian,
        number_sa_sweeps=10000,
        number_supervised_epochs=50,
        number_monte_carlo_samples=40000,
        number_outer_iterations=200,
        train_batch_size=256,
        optimizer=lambda m: torch.optim.Adam(model.parameters(), lr=1.5233e-4),
        scheduler=lambda m: None,
        device=device,
    )
    find_ground_state(config)
