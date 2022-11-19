import annealing_sign_problem
import h5py
from loguru import logger
import numpy as np
import time
import torch
from torch import Tensor
import torch.utils.data
from typing import List
import unpack_bits


MAPPINGS = {
    3: [0, 1, 2] * 12,
    6: ([0, 1, 2] * 3 + [3, 4, 5] * 3) * 2,
    9: list(range(9)) * 4,
    18: list(range(18)) * 2,
}


for m in MAPPINGS.values():
    assert len(m) == 36

LAYOUTS = {
    12: torch.tensor([[10, -1, 11, -1], [6, 7, 8, 9], [4, -1, 5, -1], [0, 1, 2, 3]]),
    18: torch.tensor(
        [
            [15, -1, 16, -1, 17, -1],
            [9, 10, 11, 12, 13, 14],
            [6, -1, 7, -1, 8, -1],
            [0, 1, 2, 3, 4, 5],
        ]
    ),
}


class Dense(torch.nn.Module):
    def __init__(self, n: int, widths: List[int]):
        super().__init__()

        self.n = n
        self.mapping = MAPPINGS[n]

        layers = []
        layers.append(torch.nn.Linear(36, widths[0]))
        for i in range(1, len(widths)):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(widths[i - 1], widths[i]))
        layers.append(torch.nn.Linear(widths[-1], 1, bias=True))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 1:
            x = x.view(-1, 1)
        x = unpack_bits.unpack(x, self.n)
        x = x[:, self.mapping]
        return self.layers(x)


class Padding(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        # x[Nbatch, 3, W, H] -> x[Nbatch, 1, W + 2 pad, H + 2 pad] (periodic padding)
        x = torch.cat([x, x[:, :, 0 : self.kernel_size[0] - 1, :]], dim=2)
        x = torch.cat([x, x[:, :, :, 0 : self.kernel_size[1] - 1]], dim=3)
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, kernel_size, channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            Padding(kernel_size),
            torch.nn.Conv2d(channels, channels, kernel_size),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layers(x)


class Convolutional(torch.nn.Module):
    def __init__(self, shape, kernel_size=(2, 2), channels=64, blocks=3, scale=1):
        super().__init__()
        self.shape = shape
        self.n = shape[0] * shape[1] * 3
        self.channels = channels
        self.scale = scale

        self.head = torch.nn.Sequential(
            Padding(kernel_size),
            torch.nn.Conv2d(3, channels, kernel_size),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
        )
        self.blocks = torch.nn.Sequential(
            *[ResidualBlock(kernel_size, channels) for _ in range(blocks)]
        )
        self.tail = torch.nn.Linear(channels, 1, bias=False)

    def forward_impl(self, x):
        x = self.head(x)
        x = self.blocks(x)
        x = x.reshape(x.size(0), self.channels, -1).mean(dim=2)
        x = self.tail(x)
        return self.scale * x

    def forward(self, x):
        if x.dim() == 1:
            x = x.view(-1, 1)
        x = unpack_bits.unpack(x, self.n)
        x = x.view((x.shape[0], self.shape[0], self.shape[1], 3))
        x = torch.permute(x, (0, 3, 1, 2))

        return 0.5 * (self.forward_impl(x) + self.forward_impl(-x))


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


class EmbeddingConv(torch.nn.Module):
    def __init__(self, layout, number_physical_spins, number_channels, kernel_size=3, dropout=None):
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

        self.number_physical_spins = number_physical_spins
        self.layout = layout
        self.layers = torch.nn.Sequential(*layers)
        self.tail = torch.nn.Linear(layout.size()[0] * layout.size()[1] * number_channels[-1], 1)

    def forward(self, x):
        dim1 = self.layout.size()[0]
        dim2 = self.layout.size()[1]
        layout = self.layout.view(dim1 * dim2)

        if x.dim() == 1:
            x = x.view(-1, 1)
        x = unpack_bits.unpack(x, self.number_physical_spins)
        zeros = torch.zeros(x.size()[0], 1).to(x.device)

        x = torch.cat((x, zeros), 1)
        x = x.transpose(0, 1)
        x = x[layout]
        x = x.transpose(0, 1)
        x = x.view(-1, 1, dim1, dim2)
        x = self.layers(x)
        x = self.tail(x.view(x.size(0), -1))

        return x


def load_basis(n):
    with h5py.File("kagome_{}.h5".format(n), "r") as f:
        return torch.from_numpy(f["basis/representatives"][:].view(np.int64))


def load_ground_state(n):
    with h5py.File("kagome_{}.h5".format(n), "r") as f:
        a = torch.from_numpy(f["hamiltonian/eigenvectors"][0, :])
        # b = torch.from_numpy(f["hamiltonian/eigenvectors"][1, :])
        # print(torch.dot(torch.abs(a), torch.abs(b)))
        return a


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


def negative_log_overlap(log_ψ: Tensor, log_φ: Tensor, log_weights: Tensor) -> Tensor:
    log_ψ, log_φ, log_weights = log_ψ.squeeze(), log_φ.squeeze(), log_weights.squeeze()
    dot_part = torch.logsumexp(log_weights + log_φ - log_ψ, dim=0)
    norm_ψ_part = torch.logsumexp(log_weights, dim=0)
    norm_φ_part = torch.logsumexp(log_weights + 2 * (log_φ - log_ψ), dim=0)
    return -dot_part + 0.5 * (norm_ψ_part + norm_φ_part)


def supervised_loop_once(
    dataset,
    model,
    optimizer,
    scheduler,
    loss_fn,
):
    tick = time.time()
    model.train()
    total_loss = 0
    total_count = 0
    for batch in dataset:
        (x, y, w) = batch
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
def compute_average_loss(dataset, model, loss_fn):
    tick = time.time()
    model.eval()
    total_loss = 0
    total_count = 0
    for batch in dataset:
        (x, y, w) = batch
        ŷ = model(x)
        loss = loss_fn(ŷ, y, w)
        total_loss += x.size(0) * loss.item()
        total_count += x.size(0)
    tock = time.time()
    return {"loss": total_loss / total_count, "time": tock - tick}


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


@torch.no_grad()
def compute_overlap(model, basis_states, amplitudes):
    model.eval()
    predicted = forward_with_batches(model, basis_states, batch_size=40960)
    predicted = torch.exp(predicted).squeeze(dim=1)
    overlap = (
        torch.dot(amplitudes.to(predicted.dtype), predicted)
        / torch.norm(predicted)
        / torch.norm(amplitudes)
    )
    return overlap.item()


def product_state_test():
    h9 = annealing_sign_problem.load_hamiltonian("kagome_9.yaml")
    h9.basis.build()
    h18 = annealing_sign_problem.load_hamiltonian("kagome_18.yaml")
    h18.basis.build()

    gs9 = load_ground_state(9)
    gs18 = load_ground_state(18)

    predicted = torch.zeros_like(gs18)
    for i in range(h18.basis.number_states):
        mask = int("1" * 9, base=2)
        i0 = int(h18.basis.states[i]) & mask
        i1 = (int(h18.basis.states[i]) >> 9) & mask
        ψ = torch.abs(gs9[h9.basis.index(i0)] * gs9[h9.basis.index(i1)])
        predicted[i] = ψ
    predicted = predicted / torch.norm(predicted)

    order = torch.argsort(torch.abs(gs18), descending=True)
    np.savetxt(
        "analysis.dat", torch.stack((torch.abs(gs18[order]), predicted[order]), dim=1).numpy()
    )


@torch.no_grad()
def preprocess_amplitudes(n, model, device, batch_size, shift=True):
    basis_states0 = load_basis(n).to(device)
    ground_state0 = load_ground_state(n).to(device)
    amplitudes0 = torch.abs(ground_state0)

    mask = amplitudes0 > torch.max(amplitudes0) * 1e-6
    basis_states = basis_states0[mask]
    amplitudes = amplitudes0[mask]

    log_amplitudes = torch.log(amplitudes).to(torch.float32)
    # print(torch.min(log_amplitudes), torch.max(log_amplitudes))
    if shift:
        log_amplitudes = log_amplitudes - 0.5 * torch.mean(log_amplitudes)
        # scale = (torch.max(log_amplitudes) - torch.min(log_amplitudes)).item()

    weights = amplitudes ** 2
    weights /= torch.max(weights)
    weights = weights.to(log_amplitudes.dtype)
    log_weights = torch.log(weights)

    dataset = TensorIterableDataset(
        basis_states,
        log_amplitudes,
        weights,
        batch_size=batch_size,
        shuffle=True,
    )

    def overlap_fn():
        overlap = compute_overlap(model, basis_states0, amplitudes0)
        return overlap

    return dataset, overlap_fn


@torch.no_grad()
def prepare_for_testing(model, device, n=36):
    basis_states = load_basis(n).to(device)
    amplitudes = torch.abs(load_ground_state(n).to(device))

    def overlap_fn():
        old_shape = model.shape
        old_n = model.n
        if n == 36:
            model.shape = (4, 3)
            model.n = 36
        elif n == 27:
            model.shape = (3, 3)
            model.n = 27
        elif n == 18:
            model.shape = (3, 2)
            model.n = 18
        else:
            assert False
        overlap = compute_overlap(model, basis_states, amplitudes)
        model.shape = old_shape
        model.n = old_n
        return overlap

    return overlap_fn


def main():
    n = 27
    testing_n = 36
    batch_size = {12: 8, 18: 256, 27: 4096, 36: 1024}[n]
    lr = {12: 1e0, 18: 4e0, 27: 4e0, 36: 1e0}[n]
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gave almost 70%:
    # model = Convolutional(
    #     shape=(3, 3), kernel_size=(2, 2), channels=4, blocks=16, scale=1e-1
    # )  # Dense(n, [1024, 1024, 1024, 1024])
    model = Convolutional(
        shape=(3, 3), kernel_size=(2, 2), channels=32, blocks=4, scale=1e-1
    )  # Dense(n, [1024, 1024, 1024, 1024])
    model.load_state_dict(torch.load("_partial_weights_18_40.pt"))
    model.to(device)

    number_parameters = sum((p.numel() for p in model.parameters()))
    logger.info("{} parameters", number_parameters)

    dataset, overlap_fn = preprocess_amplitudes(n, model, device, batch_size)
    overlap36_fn = prepare_for_testing(model, device, n=testing_n)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.7)
    # torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None

    _loss = torch.nn.MSELoss(reduction="none")

    def loss(predicted, expected, weights):
        if predicted.dim() > 1:
            predicted = predicted.squeeze(dim=1)
        if expected.dim() > 1:
            expected = expected.squeeze(dim=1)
        return torch.dot(_loss(predicted, expected), weights) / torch.sum(weights)

    loss_fn = loss  # negative_log_overlap

    info = compute_average_loss(dataset, model, loss_fn)
    logger.debug("Initial loss: {}, overlap: {}", info["loss"], overlap_fn())
    for epoch in range(epochs):
        info = supervised_loop_once(dataset, model, optimizer, scheduler, loss_fn)
        msg = "{}: loss: {}".format(epoch + 1, info["loss"])
        if (epoch + 1) % 10 == 0:
            model.eval()
            torch.save(model.state_dict(), "_partial_weights_{}_{}.pt".format(n, epoch + 1))
            overlap = overlap_fn()
            overlap_36 = overlap36_fn()
            msg += "; overlap: {}; overlap on {}: {}".format(overlap, testing_n, overlap_36)
        logger.debug(msg)
    info = compute_average_loss(dataset, model, loss_fn)
    logger.debug(
        "Final loss: {}; overlap: {}; overlap on {}: {}"
        "".format(info["loss"], overlap_fn(), testing_n, overlap36_fn())
    )

    model.eval()
    torch.save(model.cpu().state_dict(), "weights_{}.pt".format(n))


if __name__ == "__main__":
    # product_state_test()
    main()
