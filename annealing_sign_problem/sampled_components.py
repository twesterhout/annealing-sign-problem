from annealing_sign_problem import *
import argparse
import datetime
import ising_glass_annealer as sa

class Simulation:
    def __init__(self, yaml_filename, hdf5_filename):
        hamiltonian = load_hamiltonian(yaml_filename)
        ground_state, ground_state_energy, _representatives = load_ground_state(hdf5_filename)
        hamiltonian.basis.build(_representatives)
        print("Ground state energy is", ground_state_energy)
        log_coeff_fn = ground_state_to_log_coeff_fn(ground_state, hamiltonian.basis)
        # classical_hamiltonian, spins, x_exact, counts = extract_classical_ising_model(
        #     hamiltonian.basis.states, hamiltonian, log_coeff_fn
        # )
        # assert np.all(spins[:, 0] == hamiltonian.basis.states)
        probabilities = np.abs(ground_state) ** 2

        self.hamiltonian = hamiltonian
        self.classical_hamiltonian = classical_hamiltonian
        self.exact_solution = x_exact
        self.ground_state = ground_state
        self.ground_state_energy = ground_state_energy
        self.probabilities = probabilities

    def perform_sampling():
        pass

