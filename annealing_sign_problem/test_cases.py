from .common import *
import lattice_symmetries as ls
import os
import numpy as np
import torch
import h5py
import yaml


def make_test_case(basename: str, output: str, sampled: bool = False, seed: int = 123):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed + 1)

    ground_state, E, _representatives = load_ground_state(
        os.path.join(project_dir(), "{}.h5".format(basename))
    )
    basis, hamiltonian = load_basis_and_hamiltonian(
        os.path.join(project_dir(), "{}.yaml".format(basename))
    )
    basis.build(_representatives)
    _representatives = None

    spins = basis.states
    classical_hamiltonian, _spins, x0 = extract_classical_ising_model(
        spins, hamiltonian, make_log_coeff_fn(ground_state, basis), sampled=sampled
    )
    if not sampled:
        assert np.all(spins == _spins[:, 0])
    matrix = classical_hamiltonian._keep_alive[0]
    field = classical_hamiltonian._keep_alive[1]

    with h5py.File(output, "w") as f:
        f["energy"] = E
        f["ground_state"] = ground_state
        f["row_indices"] = matrix.row
        f["col_indices"] = matrix.col
        f["elements"] = matrix.data
        f["field"] = field
