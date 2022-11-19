import numpy as np
import os
import scipy
import scipy.stats


def estimate_pdf(filename: str):
    table = np.loadtxt(filename)
    order = (table.shape[1] - 1) // 2
    kernels = [
        scipy.stats.gaussian_kde(table[:, 2 + 2 * i], bw_method=0.1)
        for i in range(order)
    ]
    x = np.linspace(0, 1, 2000)
    y = [kernel(x) for kernel in kernels]
    np.savetxt("density_of_states.{}".format(filename), np.vstack([x] + y).T)


def main():
    for filename in ["kagome_sampled_power=0.1_cutoff=0.0002.dat",
                     "pyrochlore_sampled_power=0.1_cutoff=0.0002.dat",
                     "sk_sampled_power=0.1_cutoff=0.0002.dat"]:
        if os.path.exists(filename):
            estimate_pdf(filename)
        else:
            print("Warning! {} does not exist ...".format(filename))

if __name__ == '__main__':
    main()
