import argparse
import datetime
from typing import List, Optional

import ising_glass_annealer as sa

from annealing_sign_problem import *

# def compute_accuracy(x_predicted: np.ndarray, x_expected: np.ndarray, number_spins: int) -> float:
#     assert x_predicted.dtype == np.uint64 and x_expected.dtype == np.uint64
#     number_correct = sum(int(m).bit_count() for m in (x_predicted & x_expected))
#     return number_correct / number_spins


class Simulation:
    exact_model: IsingModel
    ground_state: NDArray[np.float64]
    energy: float

    def __init__(self, yaml_filename: str, hdf5_filename: str):
        hamiltonian = load_hamiltonian(yaml_filename)
        basis = hamiltonian.basis
        ground_state, ground_state_energy, _representatives = load_ground_state(hdf5_filename)
        basis.build(_representatives)
        logger.debug("Ground state energy is", ground_state_energy)

        log_coeff_fn = ground_state_to_log_coeff_fn(ground_state, hamiltonian.basis)
        # classical_hamiltonian, spins, x_exact, counts = extract_classical_ising_model(
        #     hamiltonian.basis.states, hamiltonian, log_coeff_fn
        # )
        # assert np.all(spins[:, 0] == hamiltonian.basis.states)

        self.exact_model = make_ising_model(
            hamiltonian.basis.states, hamiltonian, log_psi_fn=log_coeff_fn
        )
        self.ground_state = ground_state
        self.energy = ground_state_energy

        # self.hamiltonian = hamiltonian
        # self.classical_hamiltonian = other_model.ising_hamiltonian # classical_hamiltonian
        # self.exact_solution = other_model.initial_signs # x_exact
        # self.ground_state = ground_state
        # self.ground_state_energy = ground_state_energy

        # self.greedy_approach_quality()
        # self.presense_of_noise()

    def presense_of_noise(self):
        number_spins = self.classical_hamiltonian.shape[0]
        exact_signs = extract_signs_from_bits(self.exact_solution, number_spins)

        weights = np.abs(self.ground_state) ** 2
        weights /= np.sum(weights)

        log_amplitudes = np.log(np.abs(self.ground_state))
        print(np.min(log_amplitudes), np.max(log_amplitudes))
        with open("presense_of_noise_approx_j1j2.dat", "w") as out:
            for noise_level in np.linspace(np.log(1e-1), np.log(1e2), 120):
                noisy_ground_state = np.sign(self.ground_state) * np.exp(
                    np.log(np.abs(self.ground_state))
                    + np.exp(noise_level) * (np.random.rand(number_spins) - 0.5)
                )
                log_noisy_coeff_fn = ground_state_to_log_coeff_fn(
                    noisy_ground_state, self.hamiltonian.basis
                )
                noisy_classical_hamiltonian, _, x_noisy_exact, _ = extract_classical_ising_model(
                    self.hamiltonian.basis.states, self.hamiltonian, log_noisy_coeff_fn
                )

                other_model = make_ising_model(
                    self.hamiltonian.basis.states, self.hamiltonian, log_psi_fn=log_noisy_coeff_fn
                )
                assert not (
                    noisy_classical_hamiltonian.exchange != other_model.ising_hamiltonian.exchange
                ).any()

                x_noisy_predicted, e_noisy = sa.anneal(
                    noisy_classical_hamiltonian,
                    seed=123,
                    number_sweeps=5120,
                    repetitions=64,
                    only_best=True,
                )

                noisy_exact_signs = extract_signs_from_bits(x_noisy_predicted, number_spins)
                accuracy = np.sum(exact_signs == noisy_exact_signs) / number_spins
                if accuracy < 0.5:
                    accuracy = 1 - accuracy
                overlap = abs(np.dot(exact_signs * noisy_exact_signs, weights))
                amplitude_overlap = abs(np.dot(noisy_ground_state, self.ground_state))
                amplitude_overlap /= np.linalg.norm(noisy_ground_state)
                amplitude_overlap /= np.linalg.norm(self.ground_state)

                out.write(
                    "{}\t{}\t{}\t{}\n"
                    "".format(np.exp(noise_level), accuracy, overlap, amplitude_overlap)
                )

    def greedy_approach_quality(self):
        number_spins = self.classical_hamiltonian.shape[0]
        exact_signs = extract_signs_from_bits(self.exact_solution, number_spins)

        matrix = self.classical_hamiltonian.exchange

        aligned = exact_signs[matrix.row] == exact_signs[matrix.col]
        frustrated_mask = ((matrix.data > 0) & aligned) | ((matrix.data < 0) & ~aligned)
        normal_mask = ((matrix.data > 0) & ~aligned) | ((matrix.data < 0) & aligned)

        frustrated_dist = np.sort(np.abs(matrix.data[frustrated_mask]))[::-1]
        np.savetxt("frustrated_dist.dat", frustrated_dist)
        normal_dist = np.sort(np.abs(matrix.data[normal_mask]))[::-1]
        np.savetxt("normal_dist.dat", normal_dist)

        matrix = matrix.tocsr()
        weight = 0
        count = 0
        for i in range(number_spins):
            elements = matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]]
            index = np.argmax(np.abs(elements))
            h = elements[index]
            j = matrix.indices[matrix.indptr[i] + index]
            if h * exact_signs[i] * exact_signs[j] < 0:
                count += 1
                weight += self.ground_state[i] ** 2
        print(count / number_spins)
        print(weight / np.linalg.norm(self.ground_state) ** 2)

        # graph = networkx.convert_matrix.from_scipy_sparse_matrix(matrix)
        #     assert networkx.is_connected(graph)
        #     positive_graph = networkx.convert_matrix.from_scipy_sparse_matrix(extract(matrix.data > 0))
        #     positive_coloring = networkx.coloring.greedy_color(
        #         positive_graph, strategy="connected_sequential"
        #     )

        assert np.sum(frustrated_mask) + np.sum(normal_mask) == matrix.nnz
        assert np.isclose(
            self.classical_hamiltonian.energy(self.exact_solution), self.ground_state_energy
        )

    # def compute_accuracy(self, x):
    #     number_spins = self.classical_hamiltonian.shape[0]
    #     exact_signs = extract_signs_from_bits(self.exact_solution, number_spins)
    #     predicted_signs = extract_signs_from_bits(x, number_spins)
    #     accuracy = np.sum(exact_signs == predicted_signs) / number_spins
    #     if accuracy < 0.5:
    #         accuracy = 1 - accuracy
    #     return accuracy

    # def compute_overlap(self, x):
    #     weights = np.abs(self.ground_state) ** 2
    #     weights /= np.sum(weights)
    #     number_spins = self.classical_hamiltonian.shape[0]
    #     exact_signs = extract_signs_from_bits(self.exact_solution, number_spins)
    #     predicted_signs = extract_signs_from_bits(x, number_spins)
    #     return abs(np.dot(exact_signs * predicted_signs, weights))

    def _analyze(
        self,
        xs: List[NDArray[np.uint64]],
        es: List[float],
        accuracy_threshold: float = 0.995,
        overlap_threshold: float = 0.995,
        residual_threshold: float = 1e-12,
    ):
        count = len(xs)
        results = np.zeros((count, 3), dtype=np.float64)
        weights = self.ground_state
        for i, (x, e) in enumerate(zip(xs, es)):
            accuracy, overlap = compute_accuracy_and_overlap(
                predicted=x, exact=self.exact_model.initial_signs, weights=weights
            )
            error = abs((e - self.energy) / self.energy)
            results[i, :] = [accuracy, overlap, error]

        accuracy_prob = np.mean(results[:, 0] > accuracy_threshold)
        overlap_prob = np.mean(results[:, 1] > overlap_threshold)
        residual_prob = np.mean(results[:, 2] > residual_threshold)
        return (accuracy_prob, overlap_prob, residual_prob)

    # def summary(
    #     self, results, accuracy_threshold=0.995, overlap_threshold=0.995, residual_threshold=1e-12
    # ):
    #     def mean_std(xs):
    #         xs = np.asarray(xs)
    #         return np.mean(xs), np.std(xs)

    #     accuracy_prob = sum((r["accuracy"] > accuracy_threshold for r in results)) / len(results)
    #     overlap_prob = sum((r["overlap"] > overlap_threshold for r in results)) / len(results)
    #     residual_prob = sum((r["energy_error"] < residual_threshold for r in results)) / len(
    #         results
    #     )
    #     return (accuracy_prob, 0, overlap_prob, 0, residual_prob, 0)

    # def dump_results_to_csv(self, results, output):
    #     with open(output, "w") as f:
    #         date = datetime.datetime.today()
    #         f.write("# Generated by full_hilbert_space.py at {:%Y-%m-%d %H:%M:%S}\n".format(date))
    #         f.write("accuracy,overlap,residual\n")
    #         for r in results:
    #             f.write("{},{},{}\n".format(r["accuracy"], r["overlap"], r["energy_error"]))

    def run(self, number_sweeps: int, repetitions: int, seed=None, **kwargs):
        (xs, es) = sa.anneal(
            self.exact_model.ising_hamiltonian,
            seed=seed,
            number_sweeps=number_sweeps,
            repetitions=repetitions,
            only_best=False,
        )
        return self._analyze(xs, es, **kwargs)


def parse_command_line():
    parser = argparse.ArgumentParser(description="Test Simulated Annealing on a small system.")
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--hdf5", type=str)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--number-sweeps", type=str, required=True)
    parser.add_argument("--repetitions", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main():
    args = parse_command_line()

    yaml_filename = args.yaml
    if args.hdf5 is not None:
        hdf5_filename = args.hdf5
    else:
        hdf5_filename = yaml_filename.replace(".yaml", ".h5")
    simulation = Simulation(yaml_filename, hdf5_filename)

    sweeps = list(map(int, args.number_sweeps.split(",")))
    np.random.seed(args.seed)

    with open(args.output, "w") as f:
        columns = [
            "number_sweeps",
            "accuracy_prob",
            "accuracy_err",
            "overlap_prob",
            "overlap_err",
            "residual_prob",
            "residual_err",
        ]
        f.write(",".join(columns) + "\n")
    for number_sweeps in sweeps:
        results = np.zeros((args.trials, 3), dtype=np.float64)
        for trial in range(args.trials):
            results[trial] = simulation.run(number_sweeps, args.repetitions)
        with open(args.output, "a") as f:
            s = "{},{},{},{},{},{},{}\n".format(
                number_sweeps,
                np.mean(results[:, 0]),
                np.std(results[:, 0]),
                np.mean(results[:, 1]),
                np.std(results[:, 1]),
                np.mean(results[:, 2]),
                np.std(results[:, 2]),
            )
            f.write(s)


if __name__ == "__main__":
    main()
