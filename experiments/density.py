import numpy as np
import scipy
import scipy.stats

def main():
    table = np.loadtxt("stats_pyrochlore_2x2x2_sampled_power=0.5_cutoff=0.005.dat")
    kernel_2 = scipy.stats.gaussian_kde(table[:, 2])
    kernel_4 = scipy.stats.gaussian_kde(table[:, 4])
    kernel_6 = scipy.stats.gaussian_kde(table[:, 6])
    x = np.linspace(0, 1, 500)
    y_2 = kernel_2(x)
    y_4 = kernel_4(x)
    y_6 = kernel_6(x)
    np.savetxt("density_of_states.dat", np.vstack([x, y_2, y_4, y_6]).T)

if __name__ == '__main__':
    main()
