import numpy as np
import re
import glob
import scipy
import scipy.stats


def _get_overlap(table, i):
    return table[:, 6 * i + 2]


def _get_amplitude_overlap(table, i):
    return table[:, 6 * i + 5]


def _select_correct_size(table, size_range):
    if size_range != None:
        min_size, max_size = size_range
        is_within_range = (min_size <= table[:, 0]) & (table[:, 0] <= max_size)
        table = table[is_within_range]
    return table


def estimate_overlap_pdf(table, bw_method=None, size_range=None, points=2000):
    table = _select_correct_size(table, size_range)
    print("Using {} datapoints for KDE ...".format(table.shape[0]))
    order = table.shape[1] // 6
    kernels = [
        scipy.stats.gaussian_kde(_get_overlap(table, i), bw_method=bw_method)
        for i in range(order)
    ]

    x = np.linspace(-0.05, 1.05, points)
    y = np.vstack([x] + [kernel(x) for kernel in kernels]).T
    return y


def estimate_overlap_integrated(table, size_range=None, points=500):
    table = _select_correct_size(table, size_range)
    order = table.shape[1] // 6
    xs = np.linspace(0, 1, points)
    ys = np.zeros((len(xs), order))
    for i, b in enumerate(xs):
        ys[i, :] = [(b <= _get_overlap(table, i)).mean() for i in range(order)]
    return np.hstack([xs.reshape(-1, 1), ys])


def estimate_size_pdf(table, bw_method=None, points=5000):
    print("Using {} datapoints for KDE ...".format(table.shape[0]))
    order = table.shape[1] // 6
    sizes = [np.log10(table[:, 6 * i + 0]) for i in range(order)]
    kernels = [scipy.stats.gaussian_kde(ns, bw_method=bw_method) for ns in sizes]
    x = np.linspace(0, 7, points)
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


def walk_files():
    matcher = re.compile(
        R".*/(pyrochlore|kagome|sk)/noise_([^/]+)/cutoff_([^/]+)/.*\.csv"
    )

    for cluster in ["lilo", "snellius"]:
        for filename in glob.iglob(
            "../experiments/{}/**/*.csv*".format(cluster), recursive=True
        ):
            m = matcher.match(filename)
            if m is not None:
                yield {
                    "system": m.group(1),
                    "noise": float(m.group(2)),
                    "cutoff": float(m.group(3)),
                    "original": filename,
                }


def process_results(system: str, noise: float, cutoff: float):
    data = [
        np.loadtxt(m["original"], delimiter=",")
        for m in walk_files()
        if m["system"] == system
        and np.isclose(m["noise"], noise)
        and np.isclose(m["cutoff"], cutoff)
    ]
    data = [arr for arr in data if arr.shape[0] > 0]
    data = np.vstack(data)

    # for bw_method in [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, None]:
    #     np.savetxt(
    #         "_{}_overlap_pdf_{}.csv".format(system, bw_method),
    #         estimate_overlap_pdf(data, bw_method),
    #         delimiter=",",
    #     )

    # for bw_method in [0.01, 0.05, 0.1, None]:
    #     np.savetxt(
    #         "_{}_size_pdf_{}.csv".format(system, bw_method),
    #         estimate_size_pdf(data, bw_method),
    #         delimiter=",",
    #     )

    # np.savetxt(
    #     "_{}_overlap_integrated.csv".format(system),
    #     estimate_overlap_integrated(data, points=200),
    #     delimiter=",",
    # )

    min_size = np.min(data[:, 0])
    max_size = np.max(data[:, 0])
    bins = np.round(np.exp(np.linspace(np.log(min_size), np.log(max_size), 5))).astype(
        np.int32
    )
    size_ranges = list(zip(bins[:-1], bins[1:]))
    for r in size_ranges:
        np.savetxt(
            "_{}_overlap_integrated_{}_{}.csv".format(system, r[0], r[1]),
            estimate_overlap_integrated(data, size_range=r),
            delimiter=",",
        )


def process_noisy_results(system: str = "kagome", cutoff: float = 2e-6, order: int = 3):
    filename = "_{}_noisy_{}.csv".format(system, order)
    with open(filename, "w") as out:
        out.write(
            "# noise, amplitude overlap (25, 50, and 75 percentile), "
            "sign overlap (25, 50, and 75 percentile)\n"
        )
    for m in sorted(walk_files(), key=lambda t: t["noise"]):
        if m["system"] != system or not np.isclose(m["cutoff"], cutoff):
            continue
        data = np.loadtxt(m["original"], delimiter=",")
        if data.shape[0] <= 100:
            continue
        if data.shape[1] < (order + 1) * 6:
            continue
        sign_overlap = np.percentile(_get_overlap(data, order), [25, 50, 75])
        amplitude_overlap = np.percentile(
            _get_amplitude_overlap(data, order), [25, 50, 75]
        )
        print("noise={}: used {} data points".format(m["noise"], data.shape[0]))
        with open(filename, "a") as out:
            out.write(
                "{},{},{},{},{},{},{}\n".format(
                    m["noise"], *amplitude_overlap, *sign_overlap
                ),
            )


def load_noisy_kagome():
    lines = []
    for dir in glob.glob("../experiments/lilo/kagome/noise_*"):
        noise = float(dir.split("noise_")[1])
        fs = glob.glob(dir + "/cutoff_2e-6/kagome_36.csv*")
        table = [np.loadtxt(f, delimiter=",") for f in fs]
        table = [t for t in table if t.shape[1] // 6 > 3]
        print(len(table))
        if len(table) == 0:
            continue
        table = np.vstack(table)
        if table.shape[0] > 100:
            order = table.shape[1] // 6
            sign_overlap = np.percentile(table[:, 6 * (order - 1) + 2], [25, 50, 75])
            amplitude_overlap = np.percentile(
                table[:, 6 * (order - 1) + 5], [25, 50, 75]
            )
            lines.append(
                (
                    noise,
                    "{},{},{},{},{},{},{}".format(
                        noise, *amplitude_overlap, *sign_overlap
                    ),
                )
            )
    with open("_kagome_noisy_3.csv", "w") as f:
        for (_, l) in sorted(lines, key=lambda t: t[0]):
            f.write(l + "\n")


def main():
    process_noisy_results(order=2)
    return
    process_results("pyrochlore", 0, 1e-5)
    process_results("kagome", 0, 2e-6)
    process_results("sk", 0, 2e-6)
    # load_noisy_kagome()
    # return

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

    # for noise in ["5e-1", "1e0", "2e0"]:
    #     table = load_kagome(noise)
    #     # np.savetxt(
    #     #     "_kagome_overlap_pdf_{}.csv".format(noise),
    #     #     estimate_overlap_pdf(table),
    #     #     delimiter=",",
    #     # )
    #     # np.savetxt(
    #     #     "_kagome_size_pdf_{}.csv".format(noise),
    #     #     estimate_size_pdf(table),
    #     #     delimiter=",",
    #     # )
    #     np.savetxt(
    #         "_kagome_amplitude_overlap_pdf_{}.csv".format(noise),
    #         estimate_amplitude_overlap_pdf(table),
    #         delimiter=",",
    #     )

    table = load_sk()
    np.savetxt(
        "_sk_overlap_integrals.csv",
        estimate_integrals_under_the_curve(table, points=100),
        delimiter=",",
    )
    return
    np.savetxt("_sk_overlap_pdf_None.csv", estimate_overlap_pdf(table), delimiter=",")
    np.savetxt(
        "_sk_overlap_pdf_0.1.csv",
        estimate_overlap_pdf(table, bw_method=0.1),
        delimiter=",",
    )
    np.savetxt(
        "_sk_overlap_pdf_0.05.csv",
        estimate_overlap_pdf(table, bw_method=0.05),
        delimiter=",",
    )
    np.savetxt(
        "_sk_overlap_pdf_0.01.csv",
        estimate_overlap_pdf(table, bw_method=0.01),
        delimiter=",",
    )
    # np.savetxt("_sk_size_pdf.csv", estimate_size_pdf(table), delimiter=",")
    # min_size = np.min(table[:, 0])
    # max_size = np.max(table[:, 0])
    # bins = np.round(np.exp(np.linspace(np.log(min_size), np.log(max_size), 5))).astype(
    #     np.int32
    # )
    # size_ranges = list(zip(bins[:-1], bins[1:]))
    # for r in size_ranges:
    #     pdf = estimate_overlap_pdf(table, size_range=r)
    #     np.savetxt("_sk_overlap_pdf_{}_{}.csv".format(r[0], r[1]), pdf, delimiter=",")

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
