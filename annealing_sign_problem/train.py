import argparse
from .common import *
from .models import *
import ctypes
from collections import namedtuple
from typing import Tuple
import lattice_symmetries as ls

# import nqs_playground as nqs
# from nqs_playground.core import _get_dtype, _get_device
import math
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter  # , UninitializedParameter

# from torch_geometric.nn import DenseGCNConv, GCNConv, ChebConv
from typing import Optional
import time
from loguru import logger
import sys
import numpy as np
import concurrent.futures


def prepare_datasets(
    spins: Tensor,
    target_signs: Tensor,
    weights: Optional[Tensor],
    device: torch.device,
    dtype: torch.dtype,
    train_batch_size: int,
    inference_batch_size: int = 16384,
):
    spins = spins.to(device)
    target_signs = target_signs.to(device)
    if weights is None:
        weights = torch.ones_like(target_signs, device=device, dtype=dtype)
    data = (spins, target_signs, weights)
    train_dataset = TensorIterableDataset(*data, batch_size=train_batch_size, shuffle=True)
    test_dataset = TensorIterableDataset(*data, batch_size=inference_batch_size, shuffle=False)
    return train_dataset, test_dataset


def default_on_epoch_end(epoch, epochs, loss, accuracy=None):
    if epoch % 10 == 0:
        args = (epoch, epochs, loss)
        s = "[{}/{}]: loss = {}"
        if accuracy is not None:
            s += ", accuracy = {}"
            args = args + (accuracy,)
        logger.debug(s, *args)


def tune_neural_network(
    model: torch.nn.Module,
    train_data: Tuple[Tensor, Tensor, Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    batch_size: int,
    swa_lr: Optional[float] = None,
    swa_epochs: int = 0,
    on_epoch_end=default_on_epoch_end,
) -> None:
    logger.debug("Starting supervised training...")
    dtype = get_dtype(model)
    device = get_device(model)
    train_dataset = TensorIterableDataset(*train_data, batch_size=batch_size, shuffle=True)
    test_dataset = TensorIterableDataset(*train_data, batch_size=16384, shuffle=False)
    logger.debug("Training dataset contains {} samples", train_data[0].size(0))

    def loss_fn(ŷ: Tensor, y: Tensor, w: Tensor) -> Tensor:
        """Weighted Cross Entropy"""
        r = torch.nn.functional.cross_entropy(ŷ, y, reduction="none")
        return torch.dot(r, w)

    @torch.no_grad()
    def accuracy_fn(ŷ: Tensor, y: Tensor, w: Tensor) -> Tensor:
        r = (ŷ.argmax(dim=1) == y).to(w.dtype)
        return torch.dot(r, w)

    if swa_epochs > 0:
        assert swa_lr is not None
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)

    info = compute_average_loss(test_dataset, model, loss_fn, accuracy_fn)
    on_epoch_end(0, epochs, info["loss"], info["accuracy"])
    # Normal training
    for epoch in range(epochs - swa_epochs):
        info = supervised_loop_once(
            train_dataset,
            model,
            optimizer,
            loss_fn,
            scheduler=scheduler,
            swa_model=None,
            swa_scheduler=None,
        )
        if epoch != epochs - swa_epochs - 1:
            on_epoch_end(epoch + 1, epochs, info["loss"])
    # Stochastic weight averaging to improve generalization
    for epoch in range(epochs - swa_epochs, epochs):
        info = supervised_loop_once(
            train_dataset,
            model,
            optimizer,
            loss_fn,
            scheduler=None,
            swa_model=swa_model,
            swa_scheduler=swa_scheduler,
        )
        if epoch != epochs - 1:
            on_epoch_end(epoch + 1, epochs, info["loss"])
    if swa_epochs > 0:
        torch.optim.swa_utils.update_bn(train_dataset, swa_model)
        model.load_state_dict(swa_model.module.state_dict())

    info = compute_average_loss(test_dataset, model, loss_fn, accuracy_fn)
    on_epoch_end(epochs, epochs, info["loss"], info["accuracy"])


@torch.no_grad()
def make_sampler(basis: ls.SpinBasis, ground_state: Tensor) -> Callable[[], Tuple[Tensor, Tensor]]:
    weights = ground_state.to(torch.float64).abs() ** 2
    weights /= torch.sum(weights)
    spins = torch.from_numpy(basis.states.view(np.int64)).to(weights.device)
    assert weights.size() == spins.size()

    class Sampler(torch.nn.Module):
        spins: Tensor
        weights: Tensor

        def __init__(self, spins: Tensor, weights: Tensor) -> None:
            super().__init__()
            self.spins = spins
            self.weights = weights

        @torch.no_grad()
        def forward(self, number_samples: int) -> Tuple[Tensor, Tensor]:
            if self.weights.numel() > 2 ** 24:
                cpu_indices = np.random.choice(
                    self.weights.numel(),
                    size=number_samples,
                    replace=True,
                    p=self.weights.cpu().numpy(),
                )
                indices = torch.from_numpy(cpu_indices).to(self.weights.device)
            else:
                indices = torch.multinomial(
                    self.weights, num_samples=number_samples, replacement=True
                )
            return self.spins[indices], self.weights[indices]

    return Sampler(spins, weights)


def _extract_classical_model_with_exact_fields(
    spins, hamiltonian, ground_state, sampled_power, device=None, scale_field=None
):
    if device is None:
        device = ground_state.device
    basis = hamiltonian.basis
    log_amplitude = ground_state.abs().log().unsqueeze(dim=1)
    log_sign = torch.where(
        ground_state >= 0,
        torch.scalar_tensor(0, device=device, dtype=ground_state.dtype),
        torch.scalar_tensor(np.pi, device=device, dtype=ground_state.dtype),
    ).unsqueeze(dim=1)

    @torch.no_grad()
    def log_coeff_fn(spin: Tensor) -> Tensor:
        spin = spin.cpu().numpy().view(np.uint64)
        if spin.ndim > 1:
            spin = spin[:, 0]
        indices = torch.from_numpy(ls.batched_index(basis, spin).view(np.int64)).to(device)
        a = log_amplitude[indices]
        b = log_sign[indices]
        return torch.complex(a, b)

    return extract_classical_ising_model(
        spins,
        hamiltonian,
        log_coeff_fn,
        sampled_power=sampled_power,
        device=device,
        scale_field=scale_field,
    )


def optimize_sign_structure(
    spins: Tensor,
    weights: Tensor,
    hamiltonian: ls.Operator,
    log_coeff_fn: torch.nn.Module,
    ground_state: Tensor,
    number_sweeps: int = 10000,
    # seed: Optional[int] = None,
    beta0: Optional[int] = None,
    beta1: Optional[int] = None,
    # sampled_power: Optional[int] = None,
    scale_field: Optional[float] = None,
    cheat: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    if cheat:
        cpu_spins = spins.cpu().numpy().view(np.uint64)
        # If spins come from Monte Carlo sampling, they might contains duplicates.
        cpu_spins, cpu_counts = np.unique(cpu_spins, return_counts=True, axis=0)
        assert hamiltonian.basis.number_spins <= 64
        _cpu_spins = cpu_spins[:, 0] if cpu_spins.ndim > 1 else cpu_spins
        cpu_indices = hamiltonian.basis.batched_index(_cpu_spins)
        indices = torch.from_numpy(cpu_indices.view(np.int64)).to(spins.device)
        signs = torch.where(
            ground_state[indices] >= 0,
            torch.scalar_tensor(0, dtype=torch.int64, device=spins.device),
            torch.scalar_tensor(1, dtype=torch.int64, device=spins.device),
        )
    else:
        device = spins.device
        cpu_spins0 = spins.cpu().numpy().view(np.uint64)
        h, cpu_spins, x0, cpu_counts = extract_classical_ising_model(
            cpu_spins0,
            hamiltonian,
            log_coeff_fn,
            sampled_power=2,
            device=device,
            scale_field=scale_field,
        )
        x, _, e = sa.anneal(
            h,
            x0,
            seed=np.random.randint(1 << 31),
            number_sweeps=number_sweeps,
            beta0=beta0,
            beta1=beta1,
        )

        def extract_signs(bits):
            i = np.arange(cpu_spins.shape[0], dtype=np.uint64)
            signs = (bits[i // 64] >> (i % 64)) & 1
            signs = 1 - signs
            signs = torch.from_numpy(signs.view(np.int64)).to(device)
            return signs

        signs = extract_signs(x)
        signs0 = extract_signs(x0)
        if False:  # NOTE: disabling for now
            if torch.sum(signs == signs0).float() / spins.shape[0] < 0.5:
                logger.warning("Applying global sign flip...")
                signs = 1 - signs

    spins = torch.from_numpy(cpu_spins.view(np.int64)).to(spins.device)
    counts = torch.from_numpy(cpu_counts).to(spins.device)
    return spins, signs, counts


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
    scale_field=None,
    ground_state=None,
):
    spins0 = spins
    h, spins, x0, counts = extract_classical_ising_model(
        spins0,
        hamiltonian,
        log_psi,
        sampled_power=sampled_power,
        device=device,
        scale_field=scale_field,
    )
    x, _, e = sa.anneal(h, x0, seed=seed, number_sweeps=number_sweeps, beta0=beta0, beta1=beta1)
    if ground_state is not None:
        h_exact, _, x0_exact, _ = _extract_classical_model_with_exact_fields(
            spins0,
            hamiltonian,
            ground_state,
            sampled_power=sampled_power,
            device=device,
        )
        # x_with_exact, _, e_with_exact = sa.anneal(
        #     h_exact, x0, seed=seed, number_sweeps=number_sweeps, beta0=beta0, beta1=beta1
        # )

    def extract_signs(bits):
        i = np.arange(spins.shape[0], dtype=np.uint64)
        signs = (bits[i // 64] >> (i % 64)) & 1
        signs = 1 - signs
        signs = torch.from_numpy(signs.view(np.int64))
        return signs

    signs = extract_signs(x)
    signs0 = extract_signs(x0)
    if ground_state is not None:
        signs0_exact = extract_signs(x0_exact)
        # signs_with_exact = extract_signs(x_with_exact)
        # accuracy_with_exact = torch.sum(signs0_exact == signs_with_exact).float() / spins.shape[0]
        accuracy_normal = torch.sum(signs0_exact == signs).float() / spins.shape[0]
        # logger.debug("SA accuracy with exact fields: {}", accuracy_with_exact)
        logger.debug("SA accuracy with approximate fields: {}", accuracy_normal)

    if False:  # NOTE: disabling for now
        if torch.sum(signs == signs0).float() / spins.shape[0] < 0.5:
            logger.warning("Applying global sign flip...")
            signs = 1 - signs
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
        "output",
    ],
)


def _make_log_coeff_fn(amplitude, sign, basis, dtype=None, device=None):
    assert isinstance(amplitude, Tensor)
    assert isinstance(sign, torch.nn.Module)
    if dtype is None:
        dtype = get_dtype(sign)
    if device is None:
        device = get_device(sign)
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


def _make_log_amplitude_fn(amplitude, basis, device, dtype):
    assert isinstance(amplitude, Tensor)
    log_amplitude = amplitude.log().unsqueeze(dim=1)
    log_amplitude = log_amplitude.to(device=device, dtype=dtype)

    @torch.no_grad()
    def log_coeff_fn(spin: Tensor) -> Tensor:
        spin = spin.cpu().numpy().view(np.uint64)
        if spin.ndim > 1:
            spin = spin[:, 0]
        indices = torch.from_numpy(ls.batched_index(basis, spin).view(np.int64)).to(device)
        return log_amplitude[indices]

    return log_coeff_fn


def test_simulated_annealing_on_patches(
    hamiltonian,
    ground_state,
    sampled_power,
    number_sweeps,
    number_monte_carlo_samples,
    number_outer_iterations,
):
    basis = hamiltonian.basis

    with torch.no_grad():
        p = ground_state.abs() ** sampled_power
        p = p.numpy()
        p /= np.sum(p)

    logger.info("Using {} Monte Carlo samples...", number_monte_carlo_samples)
    for i in range(number_outer_iterations):
        logger.info("Experiment #{}...", i + 1)
        batch_indices = np.random.choice(
            basis.number_states, size=number_monte_carlo_samples, replace=True, p=p
        )

        spins0 = basis.states[batch_indices]
        h_exact, spins, x_exact, counts = _extract_classical_model_with_exact_fields(
            spins0, hamiltonian, ground_state, sampled_power=sampled_power
        )
        h, _, _, _ = _extract_classical_model_with_exact_fields(
            spins0, hamiltonian, ground_state, sampled_power=sampled_power, scale_field=0
        )
        x_from_exact, _, e_from_exact = sa.anneal(h_exact, x0=None, number_sweeps=number_sweeps)
        x, _, e = sa.anneal(h, x0=None, number_sweeps=number_sweeps)

        def extract_signs(bits):
            i = np.arange(spins.shape[0], dtype=np.uint64)
            signs = (bits[i // 64] >> (i % 64)) & 1
            signs = 1 - signs
            signs = torch.from_numpy(signs.view(np.int64))
            return signs

        signs = extract_signs(x)
        signs_exact = extract_signs(x_exact)
        signs_from_exact = extract_signs(x_from_exact)
        counts = torch.from_numpy(counts)

        accuracy = torch.sum(signs_exact == signs).float() / spins.shape[0]
        weighted_accuracy = torch.dot((signs_exact == signs).float(), counts.float()) / torch.sum(
            counts
        )
        if accuracy < 0.5:
            accuracy = 1 - accuracy
            weighted_accuracy = 1 - weighted_accuracy

        accuracy_from_exact = torch.sum(signs_exact == signs_from_exact).float() / spins.shape[0]
        weighted_accuracy_from_exact = torch.dot(
            (signs_exact == signs_from_exact).float(), counts.float()
        ) / torch.sum(counts)

        logger.info(
            "SA accuracy with exact fields: unweighted={:.4f}, weighted={:.4f}",
            accuracy_from_exact,
            weighted_accuracy_from_exact,
        )
        logger.info(
            "SA accuracy with zero  fields: unweighted={:.4f}, weighted={:.4f}",
            accuracy,
            weighted_accuracy,
        )


def find_ground_state(config):
    hamiltonian = config.hamiltonian
    basis = hamiltonian.basis
    # all_spins = torch.from_numpy(basis.states.view(np.int64))
    try:
        dtype = _get_dtype(config.model)
    except AttributeError:
        dtype = torch.float32
    try:
        device = _get_device(config.model)
    except AttributeError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_psi = _make_log_coeff_fn(config.ground_state.abs(), config.model, basis, dtype, device)
    sampled_power = 2.0  # True
    with torch.no_grad():
        p = config.ground_state.abs() ** sampled_power
        p = p.numpy()
        p /= np.sum(p)

    ground_state = config.ground_state.to(device)
    correct_sign_structure = torch.where(
        ground_state >= 0.0,
        torch.scalar_tensor(0, dtype=torch.int64, device=device),
        torch.scalar_tensor(1, dtype=torch.int64, device=device),
    )

    def compute_metrics():
        with torch.no_grad():
            config.model.eval()
            all_spins = torch.from_numpy(basis.states.view(np.int64)).to(device)
            predicted_sign_structure = forward_with_batches(config.model, all_spins, 16384)
            predicted_sign_structure = predicted_sign_structure.argmax(dim=1)
            mask = correct_sign_structure == predicted_sign_structure
            accuracy = torch.sum(mask, dim=0).item() / all_spins.size(0)
            overlap = torch.dot(2 * mask.to(ground_state.dtype) - 1, ground_state ** 2)
        return accuracy, overlap

    accuracy, overlap = compute_metrics()
    logger.info("Accuracy = {}, overlap = {}", accuracy, overlap)

    # scale_field = [0, 0] + [None for _ in range(config.number_outer_iterations - 2)]
    scale_field = [0] + [None for _ in range(config.number_outer_iterations)]
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
            scale_field=scale_field[i],
            ground_state=ground_state,
        )
        with torch.no_grad():
            if sampled_power is not None:
                weights = None
                # weights = torch.from_numpy(counts).to(dtype=dtype, device=device)
                # weights /= torch.sum(weights)
            else:
                assert np.all(counts == 1)
                weights = torch.from_numpy(p[batch_indices]).to(dtype=dtype, device=device)
                weights /= torch.sum(weights)
        spins = torch.from_numpy(spins.view(np.int64))

        optimizer = config.optimizer(config.model)
        scheduler = config.scheduler(optimizer)
        tune_neural_network(
            config.model,
            spins,
            signs,
            weights,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=config.number_supervised_epochs,
            batch_size=config.train_batch_size,
        )
        torch.save(
            config.model.state_dict(), os.path.join(config.output, "model_{}.pt".format(i + 1))
        )
        accuracy, overlap = compute_metrics()
        logger.info("Accuracy = {}, overlap = {}", accuracy, overlap)

    # return 1 - np.abs(get_overlap()), best_energy


def supervised_learning_test(config):
    hamiltonian = config.hamiltonian
    basis = hamiltonian.basis
    try:
        dtype = _get_dtype(config.model)
    except AttributeError:
        dtype = torch.float32
    try:
        device = _get_device(config.model)
    except AttributeError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ground_state = config.ground_state.to(device)
    correct_sign_structure = torch.where(
        ground_state >= 0.0,
        torch.scalar_tensor(0, dtype=torch.int64, device=device),
        torch.scalar_tensor(1, dtype=torch.int64, device=device),
    )

    sampled_power = 2
    dataset_filename = os.path.join(config.output, "training_dataset.h5")
    if os.path.exists(dataset_filename):
        with h5py.File(dataset_filename, "r") as f:
            spins = torch.from_numpy(np.asarray(f["/spins"])).to(device)
            signs = torch.from_numpy(np.asarray(f["/signs"])).to(device)
            counts = torch.from_numpy(np.asarray(f["/counts"])).to(device)
    else:
        config.model.eval()
        log_psi = _make_log_coeff_fn(config.ground_state.abs(), config.model, basis, dtype, device)
        with torch.no_grad():
            p = config.ground_state.abs() ** sampled_power
            p = p.numpy()
            p /= np.sum(p)
        if sampled_power is not None:
            batch_indices = np.random.choice(
                basis.number_states, size=config.number_monte_carlo_samples, replace=True, p=p
            )
        else:
            batch_indices = np.random.choice(
                basis.number_states, size=config.number_monte_carlo_samples, replace=False, p=None
            )
        logger.debug("[IGNORE ME] Sampled {} of the whole space...", np.sum(p[batch_indices]))
        spins = basis.states[batch_indices]
        # if True:
        #     spins, _, _ = hamiltonian.batched_apply(spins)
        #     spins, _, _ = hamiltonian.batched_apply(spins)
        #     spins, _, _ = hamiltonian.batched_apply(spins)
        #     spins = np.unique(spins, return_counts=False, axis=0)
        #     sampled_power = None

        spins, signs, counts = tune_sign_structure(
            spins,
            hamiltonian,
            log_psi,
            number_sweeps=config.number_sa_sweeps,
            sampled_power=sampled_power,
            device=device,
        )
        batch_indices = basis.batched_index(spins[:, 0]).astype(np.int64)
        logger.debug("[IGNORE ME] Sampled {} of the whole space...", np.sum(p[batch_indices]))
        probabilities = torch.from_numpy(p[batch_indices]).to(device)
        probabilities /= torch.sum(probabilities)
        batch_indices = torch.from_numpy(batch_indices).to(device)
        mask = signs.to(device) == correct_sign_structure[batch_indices]
        logger.debug("[IGNORE ME] Accuracy: {}", mask.sum() / batch_indices.size(0))
        logger.debug("[IGNORE ME] Weighted accuracy: {}", torch.dot(mask.double(), probabilities))
        with h5py.File(dataset_filename, "w") as f:
            f["/spins"] = spins.view(np.int64)
            f["/signs"] = signs.numpy()
            f["/counts"] = counts
        spins = torch.from_numpy(spins.view(np.int64))
        counts = torch.from_numpy(counts)

    weights = None
    # with torch.no_grad():
    #     if sampled_power is not None:
    #         weights = None
    #         # weights = torch.from_numpy(counts).to(dtype=dtype, device=device)
    #         # weights /= torch.sum(weights)
    #     else:
    #         assert np.all(counts == 1)
    #         weights = torch.from_numpy(p[batch_indices]).to(dtype=dtype, device=device)
    #         weights /= torch.sum(weights)

    logger.info("{}", torch.sum(1 - 2 * correct_sign_structure))
    logger.info("{}", torch.dot(1 - 2 * correct_sign_structure.double(), ground_state ** 2))

    def compute_metrics():
        with torch.no_grad():
            config.model.eval()
            all_spins = torch.from_numpy(basis.states.view(np.int64)).to(device)
            predicted_sign_structure = forward_with_batches(config.model, all_spins, 16384)
            predicted_sign_structure = predicted_sign_structure.argmax(dim=1)
            mask = correct_sign_structure == predicted_sign_structure
            accuracy = torch.sum(mask, dim=0).item() / all_spins.size(0)
            overlap = torch.dot(2 * mask.to(ground_state.dtype) - 1, ground_state ** 2)
        return accuracy, overlap

    tb_writer = SummaryWriter(log_dir=config.output)

    def on_epoch_end(epoch, epochs, loss, accuracy=None):
        if epoch % 100 == 0:
            accuracy, overlap = compute_metrics()
            tb_writer.add_scalar("loss", loss, epoch)
            tb_writer.add_scalar("accuracy", accuracy, epoch)
            tb_writer.add_scalar("overlap", overlap, epoch)
            logger.debug(
                "[{}/{}]: loss = {}, accuracy = {}, overlap = {}",
                epoch,
                epochs,
                loss,
                accuracy,
                overlap,
            )
        else:
            if accuracy is not None:
                logger.debug("[{}/{}]: loss = {}, accuracy = {}", epoch, epochs, loss, accuracy)
            tb_writer.add_scalar("loss", loss, epoch)

    accuracy, overlap = compute_metrics()
    logger.debug("Accuracy: {}; overlap: {}", accuracy, overlap)
    optimizer = config.optimizer(config.model)
    scheduler = config.scheduler(optimizer)
    tune_neural_network(
        config.model,
        spins,
        signs,
        weights,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.number_supervised_epochs,
        batch_size=config.train_batch_size,
        on_epoch_end=on_epoch_end,
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
            # Mish(),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(out_channels),
            # torch.nn.MaxPool1d(kernel_size=2),
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        k = self.kernel_size
        x = torch.cat([x[:, :, -((k - 1) // 2) :, :], x, x[:, :, : k // 2, :]], dim=2)
        x = torch.cat([x[:, :, :, -((k - 1) // 2) :], x, x[:, :, :, : k // 2]], dim=3)
        return self.layer(x)


class ConvModel(torch.nn.Module):
    def __init__(self, shape, number_channels, kernel_size=3):
        super().__init__()
        number_blocks = len(number_channels)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in number_channels]
        else:
            assert isinstance(kernel_size, (list, tuple))
            assert len(kernel_size) == number_blocks

        layers = [
            ConvBlock(in_channels=1, out_channels=number_channels[0], kernel_size=kernel_size[0])
        ]
        for i in range(1, len(number_channels)):
            layers.append(
                ConvBlock(
                    in_channels=number_channels[i - 1],
                    out_channels=number_channels[i],
                    kernel_size=kernel_size[i],
                )
            )

        self.shape = shape
        self.layers = torch.nn.Sequential(*layers)
        self.tail = torch.nn.Linear(shape[0] * shape[1] * number_channels[-1], 2)

    def forward(self, x):
        number_spins = self.shape[0] * self.shape[1]
        x = nqs.unpack(x, number_spins)
        x = self.layers(x.view(-1, 1, *self.shape))
        x = self.tail(x.view(x.size(0), -1))
        return x


class DenseModel(torch.nn.Module):
    def __init__(self, shape, number_features, use_batchnorm=True, dropout=None):
        super().__init__()
        number_features = number_features + [2]
        layers = [torch.nn.Linear(shape[0] * shape[1], number_features[0])]
        for i in range(1, len(number_features)):
            layers.append(torch.nn.ReLU(inplace=True))
            if use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(number_features[i - 1]))
            if dropout is not None:
                layers.append(torch.nn.Dropout(p=dropout, inplace=True))
            layers.append(torch.nn.Linear(number_features[i - 1], number_features[i]))

        self.shape = shape
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        number_spins = self.shape[0] * self.shape[1]
        x = nqs.unpack(x, number_spins)
        return self.layers(x)


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class GraphModel(torch.nn.Module):
    def __init__(self, graph, number_features):
        super().__init__()
        self.adj = graph
        assert torch.all(self.adj >= 0) and torch.all(self.adj < 36)
        self.gc1 = ChebConv(1, number_features, 4)
        # self.dense1 = torch.nn.Linear(36, number_features * 36)
        self.gc2 = ChebConv(number_features, number_features, 4)
        # self.gc3 = GCNConv(number_features, number_features)
        # self.gc4 = GCNConv(number_features, number_features)
        self.tail = torch.nn.Linear(number_features * 36, 2)

    def forward(self, x):
        assert x.dim() == 1 or x.dim() == 2 and x.size(1) == 8
        x = nqs.unpack(x, 36)
        assert x.dim() == 2 and x.size(1) == 36
        x = x.view(x.size(0), x.size(1), 1)
        # print(x.size())
        x = self.gc1(x, self.adj)
        x = torch.nn.functional.relu(x)
        x = self.gc2(x, self.adj)
        x = torch.nn.functional.relu(x)
        # x = self.gc3(x, self.adj)
        # x = torch.nn.functional.relu(x)
        # x = self.gc4(x, self.adj)
        # x = torch.nn.functional.relu(x)
        x = self.tail(x.view(x.size(0), -1))
        return x


if False:

    def checkerboard(shape):
        sites = np.arange(shape[0] * shape[1]).reshape(*shape)
        sites = sites % shape[1] + sites // shape[1]
        sites = sites % 2
        # print(sites)
        return torch.from_numpy(sites)

    class MarshallSignRule(torch.nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
            self.register_buffer("mask", checkerboard(shape).view(-1))

        def forward(self, x):
            if x.dtype == torch.int64:
                x = nqs.unpack(x, self.shape[0] * self.shape[1])
            # x = x.view(x.size(0), self.shape[0] * self.shape[1])
            mask = self.mask.to(dtype=x.dtype, device=x.device)
            bias = ((self.shape[0] * self.shape[1] - (x * mask).sum(dim=1)) // 2) % 2
            # logger.info(bias)
            bias = 2 * bias - 1
            # torch.where(bias > 1,
            #     torch.scalar_tensor(100.0, dtype=x.dtype, device=x.device),
            #     torch.scalar_tensor(-100.0, dtype=x.dtype, device=x.device)
            # )
            # bias = torch.cos(np.pi * 0.5 * (1 - (x * mask).sum(dim=1)))
            bias = torch.stack([torch.zeros_like(bias), bias], dim=1)
            return bias


if False:

    class Phase(torch.nn.Module):
        def __init__(self, shape, number_channels, kernel_size=3):
            super().__init__()
            number_blocks = len(number_channels)
            layers = [
                ConvBlock(in_channels=1, out_channels=number_channels[0], kernel_size=kernel_size)
            ]
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
            # self.tail = torch.nn.Linear(number_channels[-1], 2, bias=False)
            self.tail = torch.nn.Linear(shape[0] * shape[1] * number_channels[-1], 2, bias=False)
            self.msr = MarshallSignRule(shape)
            self.scale = 1.0

        def forward(self, x):
            number_spins = self.shape[0] * self.shape[1]
            input = nqs.unpack(x, number_spins)
            x = input.view(input.size(0), 1, *self.shape)
            x = self.layers(x)
            # x = x.view(x.size(0), x.size(1), -1).mean(dim=2)
            x = x.view(x.size(0), -1)
            return self.tail(x) + self.scale * self.msr(input)


if False:

    def pad_circular(x, pad):
        x = torch.cat([x, x[:, :, 0:pad, :]], dim=2)
        x = torch.cat([x, x[:, :, :, 0:pad]], dim=3)
        x = torch.cat([x[:, :, -2 * pad : -pad, :], x], dim=2)
        x = torch.cat([x[:, :, :, -2 * pad : -pad], x], dim=3)
        return x

    class Net(torch.nn.Module):
        def __init__(
            self,
            shape: Tuple[int, int],
            features1: int,
            features2: int,
            features3: int,
            window: int,
        ):
            super().__init__()
            self._shape = shape
            self._conv1 = torch.nn.Conv2d(
                1, features1, window, stride=1, padding=0, dilation=1, groups=1, bias=True
            )
            self._conv2 = torch.nn.Conv2d(
                features1, features2, window, stride=1, padding=0, dilation=1, groups=1, bias=True
            )
            self._conv3 = torch.nn.Conv2d(
                features2, features3, window, stride=1, padding=0, dilation=1, groups=1, bias=True
            )
            self._dense6 = torch.nn.Linear(features3, 2, bias=True)
            # self.dropout = torch.nn.Dropout(0.3)
            self._padding = window // 2
            self._features2 = features2
            self._features3 = features3

        #    @torch.jit.script_method
        def forward(self, x):
            x = nqs.unpack(x, self._shape[0] * self._shape[1])
            x = x.view((x.shape[0], 1, *self._shape))
            x = pad_circular(x, self._padding)
            x = self._conv1(x)
            x = torch.nn.functional.relu(x)
            x = pad_circular(x, self._padding)
            x = self._conv2(x)
            x = torch.nn.functional.relu(x)
            x = pad_circular(x, self._padding)
            x = self._conv3(x)
            x = torch.nn.functional.relu(x)
            x = x.view(x.shape[0], self._features3, -1)
            x = x.mean(dim=2)
            x = self._dense6(x)
            return x


def run_sa_only():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=int, default=10000, help="Number of SA sweeps.")
    parser.add_argument("--samples", type=int, default=100000, help="Number MC samples.")
    parser.add_argument("--iters", type=int, default=1, help="Number experiments.")
    parser.add_argument("--power", type=float, default=2, help="Power.")
    parser.add_argument("--seed", type=int, default=None, help="Seed.")
    args = parser.parse_args()

    ground_state, E, representatives = load_ground_state(
        # os.path.join(project_dir(), "data/symm/j1j2_triangle_6x6.h5")
        # os.path.join(project_dir(), "data/symm/j1j2_square_6x6_flipped.h5")
        # os.path.join(project_dir(), "data/symm/triangle_6x6.h5")
        os.path.join(project_dir(), "data/symm/heisenberg_kagome_36.h5")
    )
    basis, hamiltonian = load_basis_and_hamiltonian(
        # os.path.join(project_dir(), "data/symm/j1j2_triangle_6x6.yaml")
        # os.path.join(project_dir(), "data/symm/j1j2_square_6x6_flipped.yaml")
        # os.path.join(project_dir(), "data/symm/triangle_6x6.yaml")
        os.path.join(project_dir(), "data/symm/heisenberg_kagome_36.yaml")
    )
    basis.build(representatives)
    representatives = None
    logger.debug("Hilbert space dimension is {}", basis.number_states)

    torch.use_deterministic_algorithms(True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed + 1)
        logger.debug("Seeding PyTorch and NumPy with seed={}...", args.seed)

    test_simulated_annealing_on_patches(
        hamiltonian=hamiltonian,
        ground_state=ground_state,
        sampled_power=args.power,
        number_sweeps=args.sweeps,
        number_monte_carlo_samples=args.samples,
        number_outer_iterations=args.iters,
    )


def run_triangle_6x6():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=int, default=10000, help="Number of SA sweeps.")
    parser.add_argument("--samples", type=int, default=100000, help="Number MC samples.")
    parser.add_argument("--epochs", type=int, default=200, help="Number supervised epochs.")
    parser.add_argument("--iters", type=int, default=100, help="Number outer iterations.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Seed.")
    parser.add_argument("--device", type=str, default=None, help="Device.")
    parser.add_argument("--widths", type=str, required=True, help="Layer widths.")
    parser.add_argument("--kernels", type=str, help="Layer widths.")
    args = parser.parse_args()

    ground_state, E, representatives = load_ground_state(
        # os.path.join(project_dir(), "data/symm/j1j2_triangle_6x6.h5")
        # os.path.join(project_dir(), "data/symm/j1j2_square_6x6_flipped.h5")
        os.path.join(project_dir(), "data/symm/triangle_6x6.h5")
    )
    basis, hamiltonian = load_basis_and_hamiltonian(
        # os.path.join(project_dir(), "data/symm/j1j2_triangle_6x6.yaml")
        # os.path.join(project_dir(), "data/symm/j1j2_square_6x6_flipped.yaml")
        os.path.join(project_dir(), "data/symm/triangle_6x6.yaml")
    )
    graph = load_graph(
        # os.path.join(project_dir(), "data/symm/j1j2_triangle_6x6.yaml")
        os.path.join(project_dir(), "data/symm/triangle_6x6.yaml")
    )
    basis.build(representatives)
    representatives = None
    logger.debug("Hilbert space dimension is {}", basis.number_states)

    # torch.use_deterministic_algorithms(True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed + 1)
        logger.debug("Seeding PyTorch and NumPy with seed={}...", args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    widths = eval(args.widths)
    if args.kernels is not None:
        kernel_size = eval(args.kernels)
    else:
        kernel_size = 4
    # model = DenseModel((6, 6), number_features=widths, use_batchnorm=True).to(device)
    model = ConvModel((6, 6), number_channels=widths, kernel_size=kernel_size).to(device)
    # model = GraphModel(graph.to(device), number_features=128).to(device)
    # model.load_state_dict(torch.load("runs/symm/triangle_6x6/126/model_10.pt"))

    # [64, 64, 64] with kernel_size=4
    # SGD + CosineAnnealingLR initial lr=1e-2, final lr=1e-4, momentum=0.95
    # epochs=100, batch_size=256
    # First outer iteration without fields!
    # -> 99.3-99.4% overlap in ~10 outer iterations
    # model.load_state_dict(torch.load("runs/symm/triangle_6x6/132/model_1.pt"))
    # model.load_state_dict(torch.load("runs/symm/triangle_6x6/133/model_2.pt"))

    logger.info(model)
    logger.debug("Contains {} parameters", sum(t.numel() for t in model.parameters()))

    optimizer = lambda m: torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = lambda o: None
    # optimizer = lambda m: torch.optim.SGD(model.parameters(), lr=2e-2, momentum=0.95)
    # scheduler = lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(
    #     o, T_max=args.epochs, eta_min=1e-3
    # )
    config = Config(
        model=model,
        ground_state=ground_state,
        hamiltonian=hamiltonian,
        number_sa_sweeps=args.sweeps,
        number_supervised_epochs=args.epochs,
        number_monte_carlo_samples=args.samples,
        number_outer_iterations=args.iters,
        train_batch_size=args.batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output="runs/symm/triangle_6x6/{}".format(args.seed if args.seed is not None else "dummy"),
    )
    os.makedirs(config.output, exist_ok=True)
    find_ground_state(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=int, help="Number of SA sweeps.")
    parser.add_argument("--samples", type=int, help="Number MC samples.")
    parser.add_argument("--epochs", type=int, help="Number epochs.")
    parser.add_argument("--iters", type=int, help="Number outer iterations.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--w1", type=int, help="1st layer's width.")
    parser.add_argument("--w2", type=int, help="2nd layer's width.")
    parser.add_argument("--w3", type=int, help="3rd layer's width.")
    parser.add_argument("--seed", type=int, help="Seed.")
    parser.add_argument("--device", type=str, help="Device.")
    args = parser.parse_args()

    ground_state, E, representatives = load_ground_state(
        os.path.join(project_dir(), "data/symm/triangle_6x6.h5")
    )
    basis, hamiltonian = load_basis_and_hamiltonian(
        os.path.join(project_dir(), "data/symm/triangle_6x6.yaml")
    )
    basis.build(representatives)
    representatives = None
    logger.info("Hilbert space dimension is {}", basis.number_states)

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info("Using seed={}", args.seed)

    device = torch.device(args.device)  # "cuda" if torch.cuda.is_available() else "cpu")
    # model = Net((4, 6), 28, 28, 20, window=5) #.to(device)
    # model = Phase((6, 6), number_channels=[32, 32, 32], kernel_size=5).to(device)
    # model = MarshallSignRule((6, 6)).to(device)
    # model = Phase((6, 6), number_channels=[64, 64], kernel_size=5).to(device)
    # model.scale = 0.0
    widths = [args.w1]
    if args.w2 is not None:
        widths.append(args.w2)
    if args.w3 is not None:
        widths.append(args.w3)
    widths.append(2)
    layers = [nqs.Unpack(basis.number_spins), torch.nn.Linear(basis.number_spins, args.w1)]
    for i in range(1, len(widths)):
        layers += [
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(widths[i - 1]),
            torch.nn.Linear(widths[i - 1], widths[i]),
        ]

    model = torch.nn.Sequential(*layers).to(device)
    logger.info(model)
    #     nqs.Unpack(basis.number_spins),
    #     torch.nn.Linear(basis.number_spins, args.w1),
    #     torch.nn.ReLU(),
    #     torch.nn.BatchNorm1d(args.w1),
    #     torch.nn.Linear(args.w1, args.w2),
    #     torch.nn.ReLU(),
    #     torch.nn.BatchNorm1d(args.w2),
    #     torch.nn.Linear(args.w2, 2, bias=False),
    # ).to(device)
    # model = CombinedModel((6, 6), model)
    # model.load_state_dict(torch.load("runs/6x6/075/model_3.pt"))
    # model = torch.jit.script(model)
    logger.info("Model contains {} parameters", sum(t.numel() for t in model.parameters()))

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = Config(
        model=model,
        ground_state=ground_state,
        hamiltonian=hamiltonian,
        number_sa_sweeps=args.sweeps,
        number_supervised_epochs=args.epochs,
        number_monte_carlo_samples=args.samples,
        number_outer_iterations=args.iters,
        train_batch_size=args.batch_size,
        # lambda m: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5),
        # Settings for 64->64->64->64 with k=5
        # optimizer=lambda m: torch.optim.SGD(model.parameters(), lr=4e-2, momentum=0.95),
        optimizer=lambda m: torch.optim.AdamW(model.parameters(), lr=1e-3),  # , momentum=0.9),
        scheduler=lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(
            o, T_max=args.epochs, eta_min=1e-4
        ),
        # optimizer=lambda m: torch.optim.AdamW(model.parameters(), lr=1e-3), # , momentum=0.9),
        # torch.optim.SGD(model.parameters(), lr=2e-2, momentum=0.95), # , momentum=0.9),
        # torch.optim.AdamW(model.parameters(), lr=1e-3), # , momentum=0.9),
        # scheduler=lambda o: None,
        # torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=300, eta_min=5e-4),
        device=device,
        output="runs/symm/triangle_6x6/{}".format(args.seed),
    )
    os.makedirs(config.output, exists_ok=True)
    # supervised_learning_test(config)
    find_ground_state(config)


def load_input_files(prefix: str) -> [ls.Operator, Tensor]:
    ground_state, ground_state_energy, representatives = load_ground_state(prefix + ".h5")
    basis, hamiltonian = load_basis_and_hamiltonian(prefix + ".yaml")
    basis.build(representatives)
    representatives = None
    logger.info("Hilbert space dimension is {}", hamiltonian.basis.number_states)
    logger.info("Ground state energy is {}", ground_state_energy)
    return hamiltonian, ground_state


def make_deterministic(seed: Optional[int]) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    if seed is None:
        seed = torch.randint(1 << 31, (1,)).item()
    torch.manual_seed(seed)
    np.random.seed(seed + 1)
    logger.debug("Seeding PyTorch and NumPy with seed={}...", seed)


@torch.no_grad()
def compute_metrics_on_full_space(
    basis: ls.SpinBasis, ground_state: Tensor, model: torch.nn.Module, batch_size: int = 16384
) -> Dict[str, float]:
    model.eval()
    device = get_device(model)
    spins = torch.from_numpy(basis.states.view(np.int64))
    accuracy: float = 0.0
    overlap: float = 0.0
    for (spins_chunk, ground_state_chunk) in split_into_batches(
        (spins, ground_state), batch_size=batch_size, device=device
    ):
        predicted_sign_structure = model(spins_chunk).argmax(dim=1)
        correct_sign_structure = torch.where(
            ground_state_chunk >= 0,
            torch.scalar_tensor(0, dtype=torch.int64, device=device),
            torch.scalar_tensor(1, dtype=torch.int64, device=device),
        )
        mask = correct_sign_structure == predicted_sign_structure
        accuracy += torch.sum(mask, dim=0).item()
        overlap += torch.dot(
            2 * mask.to(ground_state_chunk.dtype) - 1, ground_state_chunk ** 2
        ).item()
    accuracy /= spins.size(0)
    return {"accuracy": accuracy, "overlap": overlap}


# NOTE: generated automatically, do not modify
KAGOME_12_ADJ = [
    (0, torch.tensor([6, 5, 9, 10, 7, 8, 11, 0, 11, 1, 2, 3, 4, 5, 6])),
    (2, torch.tensor([8, 9, 10, 11, 0, 3, 4, 1, 2, 3, 6, 5, 10, 7, 8])),
    (1, torch.tensor([9, 10, 7, 11, 0, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
    (2, torch.tensor([10, 7, 8, 0, 11, 1, 2, 3, 4, 1, 5, 6, 8, 9, 10])),
    (1, torch.tensor([7, 8, 9, 0, 11, 2, 3, 4, 1, 2, 6, 5, 9, 10, 7])),
    (0, torch.tensor([11, 0, 4, 1, 2, 3, 6, 5, 6, 10, 7, 8, 9, 0, 11])),
    (0, torch.tensor([0, 11, 2, 3, 4, 1, 5, 6, 5, 8, 9, 10, 7, 11, 0])),
    (1, torch.tensor([4, 1, 2, 6, 5, 9, 10, 7, 8, 9, 0, 11, 2, 3, 4])),
    (2, torch.tensor([1, 2, 3, 5, 6, 10, 7, 8, 9, 10, 0, 11, 3, 4, 1])),
    (1, torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 7, 11, 0, 4, 1, 2])),
    (2, torch.tensor([3, 4, 1, 6, 5, 8, 9, 10, 7, 8, 11, 0, 1, 2, 3])),
    (0, torch.tensor([5, 6, 7, 8, 9, 10, 0, 11, 0, 3, 4, 1, 2, 6, 5])),
]

# NOTE: generated automatically, do not modify
KAGOME_36_ADJ = [
    (0, torch.tensor([28, 29, 31, 32, 33, 34, 35, 0, 16, 3, 4, 5, 20, 7, 8])),
    (2, torch.tensor([14, 15, 30, 19, 35, 25, 26, 1, 2, 3, 29, 6, 34, 9, 10])),
    (1, torch.tensor([15, 30, 31, 19, 35, 26, 1, 2, 3, 4, 6, 7, 9, 10, 11])),
    (2, torch.tensor([30, 31, 32, 35, 0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12])),
    (1, torch.tensor([31, 32, 33, 35, 0, 2, 3, 4, 5, 20, 7, 8, 11, 12, 13])),
    (2, torch.tensor([32, 33, 34, 0, 16, 3, 4, 5, 20, 21, 7, 8, 12, 13, 14])),
    (0, torch.tensor([19, 35, 26, 1, 2, 3, 29, 6, 7, 34, 9, 10, 11, 16, 17])),
    (0, torch.tensor([35, 0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 18])),
    (0, torch.tensor([0, 16, 4, 5, 20, 21, 7, 8, 27, 12, 13, 14, 15, 18, 19])),
    (1, torch.tensor([26, 1, 2, 29, 6, 33, 34, 9, 10, 11, 16, 17, 20, 21, 22])),
    (2, torch.tensor([1, 2, 3, 6, 7, 34, 9, 10, 11, 12, 16, 17, 21, 22, 23])),
    (1, torch.tensor([2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 17, 18, 22, 23, 24])),
    (2, torch.tensor([3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 17, 18, 23, 24, 25])),
    (1, torch.tensor([4, 5, 20, 7, 8, 11, 12, 13, 14, 15, 18, 19, 24, 25, 26])),
    (2, torch.tensor([5, 20, 21, 8, 27, 12, 13, 14, 15, 30, 18, 19, 25, 26, 1])),
    (1, torch.tensor([20, 21, 22, 8, 27, 13, 14, 15, 30, 31, 19, 35, 26, 1, 2])),
    (0, torch.tensor([29, 6, 33, 34, 9, 10, 0, 16, 17, 5, 20, 21, 22, 8, 27])),
    (0, torch.tensor([6, 7, 9, 10, 11, 12, 16, 17, 18, 21, 22, 23, 24, 27, 28])),
    (0, torch.tensor([7, 8, 11, 12, 13, 14, 17, 18, 19, 23, 24, 25, 26, 28, 29])),
    (0, torch.tensor([8, 27, 13, 14, 15, 30, 18, 19, 35, 25, 26, 1, 2, 29, 6])),
    (1, torch.tensor([33, 34, 9, 0, 16, 4, 5, 20, 21, 22, 8, 27, 13, 14, 15])),
    (2, torch.tensor([34, 9, 10, 16, 17, 5, 20, 21, 22, 23, 8, 27, 14, 15, 30])),
    (1, torch.tensor([9, 10, 11, 16, 17, 20, 21, 22, 23, 24, 27, 28, 15, 30, 31])),
    (2, torch.tensor([10, 11, 12, 17, 18, 21, 22, 23, 24, 25, 27, 28, 30, 31, 32])),
    (1, torch.tensor([11, 12, 13, 17, 18, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33])),
    (2, torch.tensor([12, 13, 14, 18, 19, 23, 24, 25, 26, 1, 28, 29, 32, 33, 34])),
    (1, torch.tensor([13, 14, 15, 18, 19, 24, 25, 26, 1, 2, 29, 6, 33, 34, 9])),
    (0, torch.tensor([16, 17, 20, 21, 22, 23, 8, 27, 28, 14, 15, 30, 31, 19, 35])),
    (0, torch.tensor([17, 18, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 0])),
    (0, torch.tensor([18, 19, 24, 25, 26, 1, 28, 29, 6, 32, 33, 34, 9, 0, 16])),
    (2, torch.tensor([21, 22, 23, 27, 28, 14, 15, 30, 31, 32, 19, 35, 1, 2, 3])),
    (1, torch.tensor([22, 23, 24, 27, 28, 15, 30, 31, 32, 33, 35, 0, 2, 3, 4])),
    (2, torch.tensor([23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 0, 3, 4, 5])),
    (1, torch.tensor([24, 25, 26, 28, 29, 31, 32, 33, 34, 9, 0, 16, 4, 5, 20])),
    (2, torch.tensor([25, 26, 1, 29, 6, 32, 33, 34, 9, 10, 0, 16, 5, 20, 21])),
    (0, torch.tensor([27, 28, 15, 30, 31, 32, 19, 35, 0, 1, 2, 3, 4, 6, 7])),
]


def adj_to_device(adj, device):
    return [(i, t.to(device)) for (i, t) in adj]


class KagomeSignNetwork(torch.nn.Module):
    def __init__(self, number_spins: int, device):
        super().__init__()
        if number_spins == 12:
            self.adj = KAGOME_12_ADJ
        elif number_spins == 36:
            self.adj = KAGOME_36_ADJ
        self.adj = adj_to_device(self.adj, device)
        self.number_spins = number_spins
        self.sublattices = max(map(lambda t: t[0], self.adj)) + 1
        assert self.sublattices == 3
        channels = 32  # 16
        self.layers = torch.nn.Sequential(
            LatticeConvolution(1, channels, self.adj),
            torch.nn.ReLU(inplace=True),
            LatticeConvolution(channels, channels, self.adj),
            torch.nn.ReLU(inplace=True),
            LatticeConvolution(channels, channels, self.adj),
            torch.nn.ReLU(inplace=True),
            LatticeConvolution(channels, channels, self.adj),
            torch.nn.ReLU(inplace=True),
        )
        self.reduction = [
            torch.tensor([i for i in range(self.number_spins) if self.adj[i][0] == t])
            for t in range(self.sublattices)
        ]
        self.tail = torch.nn.Linear(channels * self.sublattices, 2)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(dim=1)
        x = unpack_bits.unpack(x, self.number_spins).unsqueeze(dim=1)
        x = self.layers(x)
        # x = torch.nn.functional.relu(x)
        # x = self.layer2(x)
        # x = torch.nn.functional.relu(x)
        # x0 = x[..., [2, 4, 7, 9]].sum(dim=2)
        # x1 = x[..., [0, 5, 6, 11]].sum(dim=2)
        # x2 = x[..., [1, 3, 8, 10]].sum(dim=2)
        x = torch.stack([x[..., indices].mean(dim=2) for indices in self.reduction], dim=2)
        x = x.view(x.size(0), -1)
        return self.tail(x)


def kagome_12_supervised():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        default="../data/no_symm/autogen/heisenberg_kagome_12",
        help="File basename.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed.")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device."
    )
    args = parser.parse_args()
    make_deterministic(args.seed)
    device = torch.device(args.device)
    logger.debug("The computation will run on {}", device)

    with torch.no_grad():
        hamiltonian, ground_state = load_input_files(args.prefix)
        ground_state = ground_state.to(device)

    sampler_fn = make_sampler(hamiltonian.basis, ground_state)

    model = KagomeSignNetwork(12, device).to(device)
    # model = torch.jit.script(
    #     torch.nn.Sequential(
    #         Unpack(12),
    #         torch.nn.Linear(12, 16),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(16, 16),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(16, 16),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(16, 2),
    #     )
    # ).to(device)

    def on_epoch_end(epoch: int, epochs: int, loss: float, _accuracy: Optional[float] = None):
        if epoch % 50 == 0:
            info = compute_metrics_on_full_space(hamiltonian.basis, ground_state, model)
            # tb_writer.add_scalar("loss", loss, epoch)
            # tb_writer.add_scalar("accuracy", accuracy, epoch)
            # tb_writer.add_scalar("overlap", overlap, epoch)
            logger.debug(
                "[{}/{}]: loss = {}, accuracy = {}, overlap = {}",
                epoch,
                epochs,
                loss,
                info["accuracy"],
                info["overlap"],
            )
        else:
            if _accuracy is not None:
                logger.debug("[{}/{}]: loss = {}, accuracy = {}", epoch, epochs, loss, _accuracy)
            # tb_writer.add_scalar("loss", loss, epoch)

    info = compute_metrics_on_full_space(hamiltonian.basis, ground_state, model)
    logger.debug("Accuracy: {}; overlap: {}", info["accuracy"], info["overlap"])

    spins, weights = sampler_fn(5000)
    spins, signs, counts = optimize_sign_structure(
        spins, weights, hamiltonian, ground_state, number_sweeps=10000, cheat=True
    )
    tune_neural_network(
        model,
        (spins, signs, counts),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-1),
        scheduler=None,
        epochs=300,
        batch_size=64,
        on_epoch_end=on_epoch_end,
    )


def compute_total_weight(spins, basis, ground_state):
    if spins.ndim > 1:
        spins = spins[:, 0]
    cpu_indices = basis.batched_index(spins.cpu().numpy().view(np.uint64))
    indices = torch.from_numpy(cpu_indices.view(np.int64)).to(spins.device)
    v = ground_state[indices]
    return torch.dot(v, v)


def kagome_36_supervised():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        default="../data/symm/autogen/heisenberg_kagome_36",
        help="File basename.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed.")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device."
    )
    parser.add_argument("--output", type=str, required=True, help="Output dir.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.8, help="Momentum")
    parser.add_argument("--number-spins", type=int, default=36, help="Number spins")
    args = parser.parse_args()
    make_deterministic(args.seed)
    device = torch.device(args.device)
    logger.debug("The computation will run on {}", device)

    with torch.no_grad():
        hamiltonian, ground_state = load_input_files(args.prefix)
        basis = hamiltonian.basis
        ground_state = ground_state.to(device)

    sampler_fn = make_sampler(hamiltonian.basis, ground_state)

    model = KagomeSignNetwork(args.number_spins, device).to(device)
    logger.info("Model contains {} parameters", sum(t.numel() for t in model.parameters()))
    # model.load_state_dict(torch.load("output/36/32x4_256_SGD_5e-3_0.8_0/model_weights_500.pt"))
    # assert model.layer1.adj[0][1].device == device
    # model = torch.jit.script(
    #     torch.nn.Sequential(
    #         Unpack(basis.number_spins),
    #         torch.nn.Linear(basis.number_spins, 512),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(512, 512),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(512, 512),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(512, 512),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(512, 512),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(512, 512),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(512, 512),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(512, 2),
    #     )
    # ).to(device)
    tb_writer = SummaryWriter(log_dir=args.output)

    def on_epoch_end(epoch: int, epochs: int, loss: float, _accuracy: Optional[float] = None):
        if epoch % 50 == 0:
            info = compute_metrics_on_full_space(hamiltonian.basis, ground_state, model)
            tb_writer.add_scalar("loss", loss, epoch)
            tb_writer.add_scalar("accuracy", info["accuracy"], epoch)
            tb_writer.add_scalar("overlap", info["overlap"], epoch)
            logger.debug(
                "[{}/{}]: loss = {}, accuracy = {}, overlap = {}",
                epoch,
                epochs,
                loss,
                info["accuracy"],
                info["overlap"],
            )
            torch.save(
                model.state_dict(), os.path.join(args.output, "model_weights_{}.pt".format(epoch))
            )
        elif _accuracy is not None:
            logger.debug("[{}/{}]: loss = {}, accuracy' = {}", epoch, epochs, loss, _accuracy)
            tb_writer.add_scalar("loss", loss, epoch)
            tb_writer.add_scalar("train_accuracy", loss, _accuracy)
        else:
            logger.debug("[{}/{}]: loss = {}", epoch, epochs, loss)
            tb_writer.add_scalar("loss", loss, epoch)

    # info = compute_metrics_on_full_space(hamiltonian.basis, ground_state, model)
    # logger.debug("Accuracy: {}; overlap: {}", info["accuracy"], info["overlap"])

    spins, weights = sampler_fn(50000)
    log_coeff_fn = _make_log_coeff_fn(
        ground_state.abs(), model, basis, get_dtype(model), get_device(model)
    )
    spins, signs, counts = optimize_sign_structure(
        spins, weights, hamiltonian, log_coeff_fn, ground_state, scale_field=0, cheat=False
    )
    _, signs_exact, _ = optimize_sign_structure(
        spins, weights, hamiltonian, log_coeff_fn, ground_state, cheat=True
    )
    sa_accuracy = torch.sum(signs_exact == signs).float() / spins.shape[0]
    # logger.debug("SA accuracy with exact fields: {}", accuracy_with_exact)
    logger.debug("Simulated Annealing accuracy: {:.3f}", sa_accuracy)
    logger.info("Sampled {} of the Hilbert space", compute_total_weight(spins, basis, ground_state))

    weights = counts  # torch.ones_like(counts)
    tune_neural_network(
        model,
        (spins, signs, weights),
        optimizer=torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum),
        # torch.optim.Adam(model.parameters(), lr=1e-3),
        scheduler=None,
        epochs=500,
        batch_size=256,
        on_epoch_end=on_epoch_end,
    )


def kagome_36_annealing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        default="../data/symm/autogen/heisenberg_kagome_36",
        help="File basename.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed.")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device."
    )
    parser.add_argument("--output", type=str, required=True, help="Output dir.")
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.8, help="Momentum")
    parser.add_argument("--number-spins", type=int, default=36, help="Number spins")
    args = parser.parse_args()
    make_deterministic(args.seed)
    device = torch.device(args.device)
    logger.debug("The computation will run on {}", device)

    with torch.no_grad():
        hamiltonian, ground_state = load_input_files(args.prefix)
        basis = hamiltonian.basis
        ground_state = ground_state.to(device)

    sampler_fn = make_sampler(hamiltonian.basis, ground_state)

    model = KagomeSignNetwork(args.number_spins, device).to(device)
    logger.info("Model contains {} parameters", sum(t.numel() for t in model.parameters()))

    def _on_epoch_end(
        tb_writer: SummaryWriter,
        epoch: int,
        epochs: int,
        loss: float,
        _accuracy: Optional[float] = None,
    ):
        if epoch % 50 == 0:
            info = compute_metrics_on_full_space(hamiltonian.basis, ground_state, model)
            tb_writer.add_scalar("loss", loss, epoch)
            tb_writer.add_scalar("accuracy", info["accuracy"], epoch)
            tb_writer.add_scalar("overlap", info["overlap"], epoch)
            logger.debug(
                "[{}/{}]: loss = {}, accuracy = {}, overlap = {}",
                epoch,
                epochs,
                loss,
                info["accuracy"],
                info["overlap"],
            )
            torch.save(
                model.state_dict(), os.path.join(args.output, "model_weights_{}.pt".format(epoch))
            )
        elif _accuracy is not None:
            logger.debug("[{}/{}]: loss = {}, accuracy' = {}", epoch, epochs, loss, _accuracy)
            tb_writer.add_scalar("loss", loss, epoch)
            tb_writer.add_scalar("train_accuracy", loss, _accuracy)
        else:
            logger.debug("[{}/{}]: loss = {}", epoch, epochs, loss)
            tb_writer.add_scalar("loss", loss, epoch)

    log_coeff_fn = _make_log_coeff_fn(ground_state.abs(), model, basis)
    spins, weights = sampler_fn(50000)

    for outer_iteration in range(10):
        output = os.path.join(args.output, "{:02i}".format(outer_iteration))
        tb_writer = SummaryWriter(log_dir=output)
        spins, signs, counts = optimize_sign_structure(
            spins, weights, hamiltonian, log_coeff_fn, ground_state, scale_field=0, cheat=False
        )
        _, signs_exact, _ = optimize_sign_structure(
            spins, weights, hamiltonian, log_coeff_fn, ground_state, cheat=True
        )
        sa_accuracy = torch.sum(signs_exact == signs).float() / spins.shape[0]
        logger.debug("Simulated Annealing accuracy: {:.3f}", sa_accuracy)
        logger.info(
            "Sampled {} of the Hilbert space", compute_total_weight(spins, basis, ground_state)
        )

        weights = counts  # torch.ones_like(counts)
        tune_neural_network(
            model,
            (spins, signs, weights),
            optimizer=torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum),
            scheduler=None,
            epochs=500,
            batch_size=256,
            on_epoch_end=on_epoch_end,
        )
