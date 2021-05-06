from . import *

import lattice_symmetries as ls
import nqs_playground as nqs
import torch
from torch import Tensor


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


def optimize_sign_structure(spins, hamiltonian, log_psi, sampled=False):
    ising_hamiltonian, _ = extract_classical_ising_model(spins, hamiltonian, log_psi, sampled=False)
    configuration, energy = sa.anneal(ising_hamiltonian, seed=42, number_sweeps=2000, beta0=0.1, beta1=10000)
    r = []
    for i in range(spins.shape[0]):
        sign = (int(configuration[i // 64]) >> (i % 64)) & 1
        sign = 2 * int(sign) - 1
        r.append((spins[i], sign))
    return r


def find_sign_structure_explicit(ground_state, hamiltonian):
    basis = hamiltonian.basis
    correct_sign_structure = torch.where(
        ground_state > 0.0,
        torch.scalar_tensor(1.0, dtype=ground_state.dtype),
        torch.scalar_tensor(-1.0, dtype=ground_state.dtype),
    )
    sign_structure = torch.where(
        torch.rand(ground_state.numel()) < 0.5,
        torch.scalar_tensor(1.0, dtype=ground_state.dtype),
        torch.scalar_tensor(-1.0, dtype=ground_state.dtype),
    )
    get_energy = lambda: hamiltonian.expectation((ground_state.abs() * sign_structure).numpy()).real
    get_accuracy = lambda: (correct_sign_structure == sign_structure).float().mean().item()
    get_overlap = lambda: torch.dot(ground_state, ground_state.abs() * sign_structure)
    print("Ground state energy: ", hamiltonian.expectation(ground_state.numpy()).real)
    print("Initially: ", get_energy(), get_accuracy(), get_overlap())

    for i in range(100):
        order = torch.randperm(basis.number_states)
        # for batch_indices in torch.chunk(order, 256):
        batch_indices = order[:1024]
        spins = basis.states[batch_indices]
        log_psi = make_log_coeff_fn(ground_state.abs() * sign_structure, basis)
        for (σ, s) in optimize_sign_structure(spins, hamiltonian, log_psi):
            sign_structure[basis.index(σ)] = s
        print("Energy: ", get_energy(), get_accuracy(), get_overlap())



def main():
    ground_state, E, representatives = _load_ground_state(
        "/home/tom/src/annealing-sign-problem/data/j1j2_square_4x4.h5"
        # "/home/tom/src/spin-ed/data/heisenberg_kagome_16.h5"
    )
    basis, hamiltonian = _load_basis_and_hamiltonian(
        "/home/tom/src/annealing-sign-problem/data/j1j2_square_4x4.yaml"
        # "/home/tom/src/spin-ed/example/heisenberg_kagome_16.yaml"
    )
    basis.build(representatives)
    representatives = None
    print(E)

    torch.manual_seed(123)

    find_sign_structure_explicit(ground_state, hamiltonian)
    return

    _, indices = torch.sort(torch.abs(ground_state))
    indices = indices[-500:] # torch.randperm(200)[:50]]
    # print(indices)
    initial_sign_structure = torch.where(
        torch.rand(indices.numel()) < 0.5,
        torch.scalar_tensor(1.0, dtype=ground_state.dtype),
        torch.scalar_tensor(-1.0, dtype=ground_state.dtype),
    )
    initial_state = torch.clone(ground_state)
    initial_state[indices] = abs(initial_state[indices]) * initial_sign_structure
    print(hamiltonian.expectation(initial_state.numpy()).real)

    log_psi = make_log_coeff_fn(initial_state, basis)

    ising_hamiltonian, _ = extract_classical_ising_model(basis.states[indices], hamiltonian, log_psi, sampled=False)
    print("constructed the hamiltonian")
    configuration, energy = sa.anneal(ising_hamiltonian, seed=42, number_sweeps=2000, beta0=0.1, beta1=10000)
    # print("initial:", initial_state[indices])
    # print("exact  :", ground_state[indices])
    print(configuration, energy)

    for i in range(len(indices)):
        sign = (int(configuration[i // 64]) >> (i % 64)) & 1
        sign = 2 * int(sign) - 1
        # print(i, sign)
        initial_state[indices[i]] = abs(initial_state[indices[i]]) * sign
    print(hamiltonian.expectation(initial_state.numpy()))


if __name__ == "__main__":
    main()
