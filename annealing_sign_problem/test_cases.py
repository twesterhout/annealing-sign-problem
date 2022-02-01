from .common import *
import lattice_symmetries as ls
import os
import numpy as np
import torch
import h5py
import yaml


def make_test_case(basename: str, output: str):
    ground_state, E, _representatives = load_ground_state("{}.h5".format(basename))
    basis, hamiltonian = load_basis_and_hamiltonian("{}.yaml".format(basename))
    basis.build(_representatives)
    _representatives = None

    spins = basis.states
    classical_hamiltonian, _spins, x0, _counts = extract_classical_ising_model(
        spins, hamiltonian, make_log_coeff_fn(ground_state, basis), sampled_power=None
    )
    assert np.all(spins == _spins[:, 0])
    matrix = classical_hamiltonian.exchange
    field = classical_hamiltonian.field

    with h5py.File(output, "w") as f:
        f["energy"] = E
        f["ground_state"] = ground_state
        f["row_indices"] = matrix.row
        f["col_indices"] = matrix.col
        f["elements"] = matrix.data
        f["field"] = field
