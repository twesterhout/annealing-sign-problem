import numpy as np
import os
import glob
import scipy
import scipy.stats


def estimate_overlap_pdf(table, bw_method=None, size_range=None):
    if size_range != None:
        min_size, max_size = size_range
        is_within_range = (min_size <= table[:, 0]) & (table[:, 0] <= max_size)
        table = table[is_within_range]
    print("Using {} datapoints for KDE ...".format(table.shape[0]))
    order = table.shape[1] // 6
    kernels = [
        scipy.stats.gaussian_kde(table[:, 6 * i + 2], bw_method=bw_method)
        for i in range(order)
    ]
    x = np.linspace(0, 1, 2000)
    y = np.vstack([x] + [kernel(x) for kernel in kernels]).T
    return y


def estimate_size_pdf(table, bw_method=None):
    print("Using {} datapoints for KDE ...".format(table.shape[0]))
    order = table.shape[1] // 6
    sizes = [np.log10(table[:, 6 * i + 0]) for i in range(order)]
    kernels = [scipy.stats.gaussian_kde(ns, bw_method=bw_method) for ns in sizes]
    x = np.linspace(0, 7, 2000)
    y = np.vstack([x] + [kernel(x) for kernel in kernels]).T
    return y


def estimate_amplitude_overlap_pdf(table, bw_method=None):
    print("Using {} datapoints for KDE ...".format(table.shape[0]))
    order = table.shape[1] // 6
    kernels = [
        scipy.stats.gaussian_kde(table[:, 6 * i + 5], bw_method=bw_method)
        for i in range(order)
    ]
    x = np.linspace(0, 1.01, 2000)
    y = np.vstack([x] + [kernel(x) for kernel in kernels]).T
    return y


def load_pyrochlore():
    fs = glob.glob(
        "../experiments/lilo/pyrochlore/noise_0/cutoff_1e-5/pyrochlore_32.csv*"
    )
    fs += glob.glob(
        "../experiments/snellius/pyrochlore/noise_0/cutoff_1e-5/pyrochlore_32.csv*"
    )
    data = []
    for f in fs:
        data.append(np.loadtxt(f, delimiter=","))
    return np.vstack(data)


def load_kagome(noise: str = "0"):
    fs = glob.glob(
        "../experiments/lilo/kagome/noise_{}/cutoff_2e-6/kagome_36.csv*".format(noise)
    )
    fs += glob.glob(
        "../experiments/snellius/kagome/noise_{}/cutoff_2e-6/kagome_36.csv*".format(
            noise
        )
    )
    data = []
    for f in fs:
        data.append(np.loadtxt(f, delimiter=","))
    return np.vstack(data)


def load_sk(noise: str = "0"):
    fs = glob.glob(
        "../experiments/lilo/sk/noise_{}/cutoff_2e-6/sk_32_1.csv*".format(noise)
    )
    data = []
    for f in fs:
        data.append(np.loadtxt(f, delimiter=","))
    return np.vstack(data)


# def load_noisy_kagome():
#     fs = glob.glob("../experiments/lilo/kagome/noise_1e0/cutoff_2e-6/kagome_36.csv*")
#     fs += glob.glob("../experiments/snellius/kagome/noise_1e0/cutoff_2e-6/kagome_36.csv*")
#     data = []
#     for f in fs:
#         data.append(np.loadtxt(f, delimiter=","))
#     return np.vstack(data)

# def local_energies(filename: str):
#     table = np.loadtxt(filename)
#     kernels = [
#         scipy.stats.gaussian_kde(table[:, 0], bw_method=0.07),
#         scipy.stats.gaussian_kde(table[:, 1], bw_method=0.02),
#     ]
#     x = np.linspace(-0.65, -0.4, 1000)
#     y = [kernel(x) for kernel in kernels]
#     np.savetxt("density_of_states.{}".format(filename), np.vstack([x] + y).T)
#     mask = (-0.6 < table[:, 1]) & (table[:, 1] < -0.4)
#     print(np.mean(table[:, 0]))
#     print(np.mean(table[mask, 1]))
#
#
def main():
    # table = load_pyrochlore()
    # np.savetxt("_pyrochlore_overlap_pdf.csv", estimate_overlap_pdf(table), delimiter=",")
    # min_size = np.min(table[:, 0])
    # max_size = np.max(table[:, 0])
    # bins = np.round(np.exp(np.linspace(np.log(min_size), np.log(max_size), 5))).astype(np.int32)
    # size_ranges = list(zip(bins[:-1], bins[1:]))
    # for r in size_ranges:
    #     pdf = estimate_overlap_pdf(table, size_range=r)
    #     np.savetxt("_pyrochlore_overlap_pdf_{}_{}.csv".format(r[0], r[1]), pdf, delimiter=",")
    # np.savetxt("_pyrochlore_size_pdf.csv", estimate_size_pdf(table), delimiter=",")

    for noise in ["5e-1", "1e0", "2e0"]:
        table = load_kagome(noise)
        # np.savetxt(
        #     "_kagome_overlap_pdf_{}.csv".format(noise),
        #     estimate_overlap_pdf(table),
        #     delimiter=",",
        # )
        # np.savetxt(
        #     "_kagome_size_pdf_{}.csv".format(noise),
        #     estimate_size_pdf(table),
        #     delimiter=",",
        # )
        np.savetxt(
            "_kagome_amplitude_overlap_pdf_{}.csv".format(noise),
            estimate_amplitude_overlap_pdf(table),
            delimiter=",",
        )

    table = load_sk()
    np.savetxt("_sk_overlap_pdf.csv", estimate_overlap_pdf(table), delimiter=",")
    np.savetxt("_sk_size_pdf.csv", estimate_size_pdf(table), delimiter=",")
    min_size = np.min(table[:, 0])
    max_size = np.max(table[:, 0])
    bins = np.round(np.exp(np.linspace(np.log(min_size), np.log(max_size), 5))).astype(
        np.int32
    )
    size_ranges = list(zip(bins[:-1], bins[1:]))
    for r in size_ranges:
        pdf = estimate_overlap_pdf(table, size_range=r)
        np.savetxt("_sk_overlap_pdf_{}_{}.csv".format(r[0], r[1]), pdf, delimiter=",")

    # table = load_kagome()
    # np.savetxt("_kagome_overlap_pdf.csv", estimate_overlap_pdf(table), delimiter=",")
    # np.savetxt("_kagome_size_pdf.csv", estimate_size_pdf(table), delimiter=",")
    # min_size = np.min(table[:, 0])
    # max_size = np.max(table[:, 0])
    # bins = np.round(np.exp(np.linspace(np.log(min_size), np.log(max_size), 5))).astype(
    #     np.int32
    # )
    # size_ranges = list(zip(bins[:-1], bins[1:]))
    # for r in size_ranges:
    #     pdf = estimate_overlap_pdf(table, size_range=r)
    #     np.savetxt(
    #         "_kagome_overlap_pdf_{}_{}.csv".format(r[0], r[1]), pdf, delimiter=","
    #     )


if __name__ == "__main__":
    main()
