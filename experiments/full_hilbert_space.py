from annealing_sign_problem import *
import argparse
import datetime
import ising_glass_annealer as sa


# def compute_accuracy(x_predicted: np.ndarray, x_expected: np.ndarray, number_spins: int) -> float:
#     assert x_predicted.dtype == np.uint64 and x_expected.dtype == np.uint64
#     number_correct = sum(int(m).bit_count() for m in (x_predicted & x_expected))
#     return number_correct / number_spins


class Simulation:
    def __init__(self, yaml_filename, hdf5_filename):
        hamiltonian = load_hamiltonian(yaml_filename)
        ground_state, ground_state_energy, _representatives = load_ground_state(hdf5_filename)
        hamiltonian.basis.build(_representatives)
        print("Ground state energy is", ground_state_energy)
        log_coeff_fn = ground_state_to_log_coeff_fn(ground_state, hamiltonian.basis)
        classical_hamiltonian, spins, x_exact, counts = extract_classical_ising_model(
            hamiltonian.basis.states, hamiltonian, log_coeff_fn
        )
        assert np.all(spins[:, 0] == hamiltonian.basis.states)

        self.hamiltonian = hamiltonian
        self.classical_hamiltonian = classical_hamiltonian
        self.exact_solution = x_exact
        self.ground_state = ground_state
        self.ground_state_energy = ground_state_energy

    def compute_accuracy(self, x):
        number_spins = self.classical_hamiltonian.shape[0]
        exact_signs = extract_signs_from_bits(self.exact_solution, number_spins)
        predicted_signs = extract_signs_from_bits(x, number_spins)
        return np.sum(exact_signs == predicted_signs) / number_spins

    def compute_overlap(self, x):
        weights = np.abs(self.ground_state) ** 2
        weights /= np.sum(weights)
        number_spins = self.classical_hamiltonian.shape[0]
        exact_signs = extract_signs_from_bits(self.exact_solution, number_spins)
        predicted_signs = extract_signs_from_bits(x, number_spins)
        return abs(np.dot(exact_signs * predicted_signs, weights))

    def analyze(self, xs, es):
        results = []
        for x, e in zip(xs, es):
            accuracy = self.compute_accuracy(x)
            overlap = self.compute_overlap(x)
            error = abs((e - self.ground_state_energy) / self.ground_state_energy)
            results.append({"accuracy": accuracy, "overlap": overlap, "energy_error": error})
        return results

    def summary(self, results):
        def mean_std(xs):
            xs = np.asarray(xs)
            return np.mean(xs), np.std(xs)

        accuracy_mean, accuracy_err = mean_std([r["accuracy"] for r in results])
        overlap_mean, overlap_err = mean_std([r["overlap"] for r in results])
        residual_mean, residual_err = mean_std([r["energy_error"] for r in results])
        return (accuracy_mean, accuracy_err, overlap_mean, overlap_err, residual_mean, residual_err)

    def dump_results_to_csv(self, results, output):
        with open(output, "w") as f:
            date = datetime.datetime.today()
            f.write("# Generated by full_hilbert_space.py at {:%Y-%m-%d %H:%M:%S}\n".format(date))
            f.write("accuracy,overlap,residual\n")
            for r in results:
                f.write("{},{},{}\n".format(r["accuracy"], r["overlap"], r["energy_error"]))

    def run(self, number_sweeps, repetitions, seed=None):
        return sa.anneal(
            self.classical_hamiltonian,
            seed=seed,
            number_sweeps=number_sweeps,
            repetitions=repetitions,
            only_best=False,
        )


def parse_command_line():
    parser = argparse.ArgumentParser(description="Test Simulated Annealing on a small system.")
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--hdf5", type=str)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--number-sweeps", type=str, required=True)
    parser.add_argument("--repetitions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main():
    args = parse_command_line()

    yaml_filename = args.yaml
    if args.hdf5 is not None:
        hdf5_filename = args.hdf5
    else:
        hdf5_filename = yaml_filename.replace(".yaml", ".h5")
    output = args.output
    repetitions = args.repetitions
    simulation = Simulation(yaml_filename, hdf5_filename)

    sweeps = list(map(int, args.number_sweeps.split(",")))
    np.random.seed(args.seed)

    for number_sweeps in sweeps:
        results = simulation.run(number_sweeps=number_sweeps, repetitions=repetitions)
        results = simulation.analyze(*results)
        simulation.dump_results_to_csv(
            results, output.replace(".csv", ".raw.{}.csv".format(number_sweeps))
        )
        with open(output, "a") as f:
            f.write("{},{},{},{},{},{},{}\n".format(number_sweeps, *simulation.summary(results)))


if __name__ == "__main__":
    main()
