from . import *
from typing import Tuple
import lattice_symmetries as ls
import nqs_playground as nqs
import torch
from torch import Tensor
import time
from loguru import logger

def project_dir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def _load_ground_state(filename: str):
    import h5py

    with h5py.File(filename, "r") as f:
        ground_state = f["/hamiltonian/eigenvectors"][:]
        ground_state = ground_state.squeeze()
        energy = f["/hamiltonian/eigenvalues"][0]
        basis_representatives = f["/basis/representatives"][:]
    return torch.from_numpy(ground_state), energy, basis_representatives


def _load_basis_and_hamiltonian(filename: str):
    import yaml

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


def supervised_loop_once(dataset, model, optimizer, scheduler, loss_fn, global_index):
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
        global_index += 1
    tock = time.time()
    return {"loss": total_loss / total_count, "global_index": global_index, "time": tock - tick}


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


def tune_neural_network(
    model, spins, target_signs, weights=None, epochs: int = 100, batch_size: int = 32, learning_rate=1e-3, w_d=5e-5
):

    if weights is None:
        weights = torch.ones_like(target_signs, dtype=torch.float32)
    target_signs = torch.where(
        target_signs > 0,
        torch.scalar_tensor(0, dtype=torch.int64, device=target_signs.device),
        torch.scalar_tensor(1, dtype=torch.int64, device=target_signs.device),
    )

    def loss_fn(ŷ, y, w):
        r = torch.nn.functional.cross_entropy(ŷ, y, reduction="none")
        return torch.dot(r, w)

    def accuracy_fn(ŷ, y, w):
        return torch.dot((ŷ.argmax(dim=1) == y).to(w.dtype), w)

    dataset = TensorIterableDataset(
        spins,
        target_signs,
        weights,
        batch_size=batch_size,
        shuffle=True,
    )
    #optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_d)
    scheduler = None

    info = compute_average_loss(dataset, model, loss_fn, accuracy_fn)
    logger.debug("Initially: loss = {}, accuracy = {}", info["loss"], info["accuracy"])
    # if info["accuracy"] < 0.3:
    #     logger.warning("Flipping signs globally...")
    #     target_signs = 1 - target_signs

    global_index = 0
    for epoch in range(epochs):
        info = supervised_loop_once(dataset, model, optimizer, scheduler, loss_fn, global_index)
        global_index = info["global_index"]
        # logger.debug("[{}/{}]  : loss = {}", epoch + 1, epochs, info["loss"])
    info = compute_average_loss(dataset, model, loss_fn, accuracy_fn)
    logger.debug("Finally  : loss = {}, accuracy = {}", info["loss"], info["accuracy"])


def optimize_sign_structure(spins, hamiltonian, log_psi, number_sweeps=10000, beta0=10, beta1=10000, sampled=False):
    # extract_classical_ising_model(spins, hamiltonian, log_psi, sampled=sampled)
    ising_hamiltonian, spins, x0 = extract_classical_ising_model(
        spins, hamiltonian, log_psi, sampled=sampled
    )
    #configuration, _, best_energy = sa.anneal(
    #    ising_hamiltonian, x0, number_sweeps=10000, beta0=10, beta1=10000
    #)
    configuration, _, best_energy = sa.anneal(
        ising_hamiltonian, x0, number_sweeps, beta0, beta1
    )
    # print(best_energy[0], best_energy[-1])
    i = np.arange(spins.shape[0], dtype=np.uint64)
    signs = 2 * ((configuration[i // 64] >> (i % 64)) & 1).astype(np.float64) - 1
    r = list(zip(spins, signs))
    # r = []
    # for i in range(spins.shape[0]):
    #     sign = (int(configuration[i // 64]) >> (i % 64)) & 1
    #     sign = 2 * int(sign) - 1
    #     r.append((spins[i], sign))
    return r


def find_sign_structure_neural(model, ground_state, hamiltonian, beta0=10, beta1=10000, sweep_sa=10000, sign_batch=40960, lr=1e-3, weight_decay=5e-5, instances=100, epochs=100, learning_batch=32):

    basis = hamiltonian.basis
    all_spins = torch.from_numpy(basis.states.view(np.int64))
    correct_sign_structure = torch.where(
        ground_state > 0.0,
        torch.scalar_tensor(1.0, dtype=ground_state.dtype),
        torch.scalar_tensor(-1.0, dtype=ground_state.dtype),
    )

    #predict_signs = lambda: 1 - 2 * model(all_spins).argmax(dim=1).float()
    predict_signs = lambda: 1 - 2 * (nqs.forward_with_batches(model, all_spins, 1000)).argmax(dim=1).float()

    get_energy = lambda: hamiltonian.expectation(
        (ground_state.abs() * predict_signs()).numpy()
    ).real
    get_accuracy = lambda: (correct_sign_structure == predict_signs()).float().mean().item()
    get_overlap = lambda: torch.dot(ground_state, ground_state.abs() * predict_signs()).item()
    # print("Ground state energy: ", hamiltonian.expectation(ground_state.numpy()).real)
    print("Initially: ", get_energy(), get_accuracy(), get_overlap())
    p = ground_state.abs() ** 2

    for i in range(instances):
        # order = torch.randperm(basis.number_states)
        # batch_indices = order[:1024]
        batch_indices = np.random.choice(basis.number_states, size=sign_batch, replace=True, p=p)

        spins = basis.states[batch_indices]
        log_psi = make_log_coeff_fn(ground_state.abs() * predict_signs(), basis)
        r = optimize_sign_structure(spins, hamiltonian, log_psi, sweep_sa, beta0, beta1, sampled=True)
        spins = np.stack([t[0] for t in r])
        if spins.ndim > 1:
            spins = spins[:, 0]
        signs = torch.tensor([t[1] for t in r])
        weights = None
        # (ground_state.abs() ** 2)[basis.batched_index(spins).view(np.int64)].float()
        tune_neural_network(model, torch.from_numpy(spins.view(np.int64)), signs, weights, epochs, learning_batch, lr, weight_decay)
        #if i % 5 == 4:
        #    print("Energy: ", get_energy(), get_accuracy(), get_overlap())

    return 1 - np.abs(get_overlap())

def pad_circular(x, pad):
    x = torch.cat([x, x[:, :, 0:pad, :]], dim=2)
    x = torch.cat([x, x[:, :, :, 0:pad]], dim=3)
    x = torch.cat([x[:, :, -2 * pad:-pad, :], x], dim=2)
    x = torch.cat([x[:, :, :, -2 * pad:-pad], x], dim=3)
    return x

class Net(torch.nn.Module):   

    def __init__(self, shape: Tuple[int, int], features1: int, features2: int, window: int):
        super().__init__()
        self._shape = shape
        self._conv1 = torch.nn.Conv2d(1, features1, window, stride=1, padding = 0, dilation=1, groups=1, bias=True)
        self._conv2 = torch.nn.Conv2d(features1, features2, window, stride=1, padding = 0, dilation=1, groups=1, bias=True)
        #self._conv3 = torch.nn.Conv2d(64, 128, 5, stride=1, padding = 0, dilation=1, groups=1, bias=True)
        self._dense6 = torch.nn.Linear(features2, 2, bias=True)
        self.dropout = torch.nn.Dropout(0.3)
        self._padding = window//2
        self._features2 = features2
#    @torch.jit.script_method
    def forward(self, x):
        x = nqs.unpack(x, self._shape[0]*self._shape[1])
        x = x.view((x.shape[0], 1, *self._shape))
        x = pad_circular(x, self._padding)
        x = self._conv1(x)
        x = torch.nn.functional.relu(x)
        x = pad_circular(x, self._padding)
        x = self._conv2(x)
        x = torch.nn.functional.relu(x)
        #x = pad_circular(x, 1)
        #x = self._conv3(x)
        #x = torch.nn.functional.relu(x)
        x = x.view(x.shape[0], self._features2, -1)
        x = x.mean(dim = 2)
        x = self._dense6(x)
        return x


def main(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, window):

    ground_state, E, representatives = _load_ground_state(
        # "/home/tom/src/annealing-sign-problem/data/j1j2_square_4x4.h5"
        os.path.join(project_dir(), "data/nonsym/j1j2_square_4x6_nonsym.h5")
        # "/home/tom/src/spin-ed/data/heisenberg_kagome_16.h5"
    )
    basis, hamiltonian = _load_basis_and_hamiltonian(
        # "/home/tom/src/annealing-sign-problem/data/j1j2_square_4x4.yaml"
        os.path.join(project_dir(), "data/nonsym/j1j2_square_4x6_nonsym.yaml")
        # "/home/tom/src/spin-ed/example/heisenberg_kagome_16.yaml"
    )
    basis.build(representatives)
    representatives = None
    #print(E)
      
    torch.manual_seed(123)
    np.random.seed(127)

    #model = torch.nn.Sequential(
    #    nqs.Unpack(basis.number_spins),
    #    torch.nn.Linear(basis.number_spins, 128),
    #    torch.nn.ReLU(),
    #    torch.nn.Linear(128, 128),
    #    torch.nn.ReLU(),
    #    torch.nn.Linear(128, 2, bias=False),
    #)

    model = Net((4, 6), features1, features2, window)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Parameters in total", pytorch_total_params)

    #sys.exit()

    loss = find_sign_structure_neural(model, ground_state, hamiltonian, beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch)
    print("loss = ", loss)
    return loss

if __name__ == "__main__":
    main(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, window)
