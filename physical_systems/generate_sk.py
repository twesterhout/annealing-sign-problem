import numpy as np
from typing import Optional

np.random.seed(167453)


def generate_yaml(number_spins: int, output: Optional[str] = None, mu: float = 0, sigma: float = 1):
    if output is None:
        output = "sk_{}.yaml".format(number_spins)
    assert ".yaml" in output

    with open(output, "w") as f:
        s = """\
basis:
  number_spins: {}
  hamming_weight: {}
  symmetries: []
hamiltonian:
  name: "Sherrington-Kirkpatrick"
  terms:
"""
        f.write(s.format(number_spins, number_spins // 2))

        matrix = np.array([[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]], dtype=float)
        for i in range(number_spins - 1):
            for j in range(i + 1, number_spins):
                coupling = np.random.normal(mu, sigma)
                f.write("    - matrix: {}\n".format((coupling * matrix).tolist()))
                f.write("      sites: [[{}, {}]]\n".format(i, j))

        f.write("observables: []\n")
        f.write("output: \"{}\"\n".format(output.replace(".yaml", ".h5")))

if __name__ == '__main__':
    # Generate different instances
    generate_yaml(16, "sk_16_1.yaml")
    generate_yaml(16, "sk_16_2.yaml")
    generate_yaml(16, "sk_16_3.yaml")
