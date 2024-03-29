<h1 align="center">
Many-body quantum sign structures as non-glassy Ising models
</h1>

<div align="center">

<img src="assets/construction_1f6a7.png" width="32">This is a research project that is not meant for general usage.<img src="assets/construction_1f6a7.png" width="32"><br>

[**Paper**](https://arxiv.org/abs/2207.10675) | [**Data**](https://doi.org/10.5281/zenodo.8221180)

[![License](https://img.shields.io/github/license/twesterhout/annealing-sign-problem.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8221332.svg)](https://doi.org/10.5281/zenodo.8221332)

<table>
<tr>
<td>

The non-trivial phase structure of the eigenstates of many-body quantum systems
severely limits the applicability of quantum Monte Carlo, variational, and machine learning methods.
Here, we study real-valued signful ground-state wave functions of frustrated quantum spin systems
and, assuming that the tasks of finding wave function amplitudes and signs can be separated, show
that the signs can be easily bootstrapped from the amplitudes. We map the problem of finding the
sign structure to an auxiliary classical Ising model defined on a subset of the Hilbert space basis.
We show that the Ising model does not exhibit significant frustrations even for highly frustrated
parental quantum systems, and is solvable with a fully deterministic O(K log K)-time combinatorial
algorithm (where K is the Ising model size). Given the ground state amplitudes, we reconstruct
the signs of the ground states of several frustrated quantum models, thereby revealing the hidden
simplicity of many-body sign structures.

</td>
</tr>
</table>

</div>

<br />

If either this code base or the paper has benefited your research, consider citing it:

```
@article{westerhout2022unveiling,
  title={Unveiling ground state sign structures of frustrated quantum systems via non-glassy Ising models},
  author={Westerhout, Tom and Katsnelson, Mikhail I and Bagrov, Andrey A},
  journal={arXiv preprint arXiv:2207.10675},
  year={2022}
}
```

## A few words about the data

Our analysis consists of multiple stages.

The first stage is running the exact diagonalization for all studied models. We
used SpinED version 4c3305a to perform the diagonalization.

The commands are of the form

```bash
OMP_NUM_THREADS=`nproc` /path/to/SpinED-4c3305a /path/to/input.yaml
```

Both the input files and the output files are located in the `physical_systems/` folder:

```
physical_systems
├── data-small
│   ├── heisenberg_kagome_16.h5
│   ├── heisenberg_kagome_18.h5
│   ├── j1j2_square_4x4.h5
│   ├── sk_16_1.h5
│   ├── sk_16_2.h5
│   └── sk_16_3.h5
├── data-large
│   ├── heisenberg_kagome_36.h5
│   ├── heisenberg_pyrochlore_2x2x2.h5
│   └── sk_32_1.h5
├── generate_sk.py
├── heisenberg_kagome_16.yaml
├── heisenberg_kagome_18.yaml
├── heisenberg_kagome_36.yaml
├── heisenberg_pyrochlore_2x2x2.yaml
├── j1j2_square_4x4.yaml
├── sk_16_1.yaml
├── sk_16_2.yaml
├── sk_16_3.yaml
└── sk_32_1.yaml
```

All the HDF5 (`.h5`) files are available for download from
[Zenodo](https://doi.org/10.5281/zenodo.8221180)


### Figure 2

To generate the data, we used `make small`, for plotting, we used [this
script](./figures/plot_annealing_on_small_systems.gnu). Raw data can be found
on Surfdrive:

```
experiments
├── ...
├── heisenberg_kagome_16.csv
├── heisenberg_kagome_18.csv
├── j1j2_square_4x4.csv
├── sk_16_1.csv
├── sk_16_2.csv
└── sk_16_3.csv
```

### Figure 3a

To generate the data, we used `make experiments/couplings/%.csv` where `%` is
`heisenberg_kagome_16`, `heisenberg_kagome_18`, or `sk_16_3`. For plotting,
[this script](./figures/plot_coupling_distribution.gnu) was used. Raw data can be found on Surfdrive:

```
experiments/couplings
├── heisenberg_kagome_16.csv
├── heisenberg_kagome_18.csv
├── j1j2_square_4x4.csv
├── sk_16_1.csv
├── sk_16_2.csv
└── sk_16_3.csv
```

### Figure 3b

To generate the data, we used `make is_frustrated`. For plotting, [this
script](./figures/plot_frustration_probability.gnu) was used. Raw data can be
found on Surfdrive:

```
experiments/is_frustrated
├── heisenberg_kagome_16.csv
├── heisenberg_kagome_18.csv
├── j1j2_square_4x4.csv
├── sk_16_1.csv
├── sk_16_2.csv
└── sk_16_3.csv
```

### Table 1

The data was generated using `make quality_check`.

### Figure 4

To generate the data, we used `make experiments/noise/%.csv` where `%` is
`heisenberg_kagome_16`, `heisenberg_kagome_18`, or `sk_16_3`. For
post-processing, we used the
`annealing_sign_problem.common.postprocess_influence_of_noise` function. For
plotting, [this script](./figures/plot_amplitude_vs_sign_overlap.gnu) was used.
Raw data can be found on Surfdrive:

```
experiments/lilo/noise/
├── heisenberg_kagome_16.csv
├── heisenberg_kagome_16_stats.csv
├── heisenberg_kagome_18.csv
├── heisenberg_kagome_18_stats.csv
├── j1j2_square_4x4.csv
├── j1j2_square_4x4_stats.csv
├── sk_16_1.csv
├── sk_16_1_stats.csv
├── sk_16_2.csv
├── sk_16_2_stats.csv
├── sk_16_3.csv
└── sk_16_3_stats.csv
```

### Figure 6

The data was generated using `make pyrochlore_32`, `make kagome_36`, or `make
sk_32_1`. The Makefile also accepts the `NOISE` and `CUTOFF` arguments that can
be used to analyze the influence of noise in the amplitudes and to tune the
cutoff rate for cluster extensions, respectively. The jobs generate files that look like:

```
...
└── noise_7.9e-01
    └── cutoff_2e-6
        ├── kagome_36.csv7665204
...
```

the number appended to the file name indicates the job id. In this way, one can
start multiple independent jobs to gather more statistical data. The raw data
that we generated that way can be found in the `experiments/lilo` and
`experiments/snellius` directories on Surfdrive. Keep in mind, that not all
of the data was used for plotting.

The data was the pre-processed using [this script](./figures/density.py) to
generate various probability distributions. We then used
[plot_greedy_overlap_density.gnu](./figures/plot_greedy_overlap_density.gnu),
[plot_overlap_integrated.gnu](./figures/plot_overlap_integrated.gnu), and
[plot_size_density.gnu](./figures/plot_size_density.gnu) for plotting.

### Figure 7

The data from Figure 6 was reused. The plotting was done with [this
script](./figures/plot_greedy_overlap_density.gnu).

### Supplemental information

Data for the figures in the Supplemental Information were be generated similarly to those in the main text.
