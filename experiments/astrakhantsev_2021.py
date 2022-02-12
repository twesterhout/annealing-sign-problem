import numpy as np
import torch
import unpack_bits
import nqs_playground as nqs


class Net_2x2x2_dense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dense = torch.nn.Linear(32, 128, bias=True)

    def forward(self, x):
        x = self._dense(x)
        x = torch.log(torch.cosh(x))
        return torch.sum(x, axis=1).view(x.shape[0], 1)


def combine_amplitude_and_phase(*modules) -> torch.nn.Module:
    class CombiningState(torch.nn.Module):
        def __init__(self, amplitude, phase, kx=0, ky=0, kz=0, nx=0, ny=0, nz=0):
            super().__init__()
            self.amplitude = amplitude
            self.phase = phase
            self.amplitude_correction = 1

        def forward(self, x):
            ampl = self.amplitude(x)
            phase = self.phase(x)
            return torch.cat([ampl * self.amplitude_correction, phase], dim=1)

    return CombiningState(*modules)


def logmeanexp(inputs_logampls, inputs_phases, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs_logampls = inputs_logampls.view(-1)
        inputs_phases = inputs_phases.view(-1)
        dim = 0
    # print(inputs_logampls.size(), inputs_phases.size())

    s, _ = torch.max(inputs_logampls, dim=dim, keepdim=True)
    # print(inputs_logampls, s)
    inputs_logampls -= s
    real = torch.sum(torch.exp(inputs_logampls) * torch.cos(inputs_phases), axis=1).unsqueeze(1)
    imag = torch.sum(torch.exp(inputs_logampls) * torch.sin(inputs_phases), axis=1).unsqueeze(1)
    # print('realimag', real.size(), imag.size())
    logampls = (
        torch.log(real ** 2 + imag ** 2) / 2.0
        + s
        - torch.log(torch.ones(real.size(0), device=s.device) * inputs_logampls.size(1)).unsqueeze(
            1
        )
    )
    phases = torch.atan2(imag, real)

    # print('after', logampls.size(), phases.size())
    return logampls, phases


class Net_nonsymmetric_2l_2x2x2(torch.nn.Module):
    def __init__(self, nx: int, ny: int, nz: int):
        super().__init__()
        self.mask_size = (2, 2, 2)
        self._conv1 = torch.nn.Conv3d(
            4, 16, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._conv2 = torch.nn.Conv3d(
            16, 32, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._dense = torch.nn.Linear(32, 1, bias=False)

        self.nx = nx
        self.ny = ny
        self.nz = nz

    def pad_circular(
        self, x
    ):  # x[Nbatch, 1, W, H] -> x[Nbatch, 1, W + 2 pad, H + 2 pad] (pariodic padding)
        if self.nx > 1:
            x = torch.cat([x, x[:, :, 0 : self.mask_size[0] - 1, :, :]], dim=2)
        if self.ny > 1:
            x = torch.cat([x, x[:, :, :, 0 : self.mask_size[1] - 1, :]], dim=3)
        if self.nz > 1:
            x = torch.cat([x, x[:, :, :, :, 0 : self.mask_size[2] - 1]], dim=4)
        return x

    def forward(self, x):
        x = x.view((x.shape[0], 4, self.nx, self.ny, self.nz))
        x = self.pad_circular(x)
        x = self._conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pad_circular(x)
        x = self._conv2(x)
        x = torch.nn.functional.relu(x)
        x = x.view(x.shape[0], 32, -1).mean(dim=2)
        x = self._dense(x)
        return x


class Net_nonsymmetric_1l_2x2x2_narrowing_simplephase(torch.nn.Module):
    def __init__(self, nx: int, ny: int, nz: int):
        super().__init__()
        self.mask_size = (2, 2, 2)
        self._conv = torch.nn.Conv3d(
            4, 32, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._dense = torch.nn.Linear(32, 1, bias=False)

        self.nx = nx
        self.ny = ny
        self.nz = nz

    def pad_circular(
        self, x
    ):  # x[Nbatch, 1, W, H] -> x[Nbatch, 1, W + 2 pad, H + 2 pad] (pariodic padding)
        if self.nx > 1:
            x = torch.cat([x, x[:, :, 0 : self.mask_size[0] - 1, :, :]], dim=2)
        if self.ny > 1:
            x = torch.cat([x, x[:, :, :, 0 : self.mask_size[1] - 1, :]], dim=3)
        if self.nz > 1:
            x = torch.cat([x, x[:, :, :, :, 0 : self.mask_size[2] - 1]], dim=4)

        return x

    def forward(self, x):
        x = x.view((x.shape[0], 4, self.nx, self.ny, self.nz))
        x = self.pad_circular(x)
        x = self._conv(x)
        x = torch.nn.functional.elu(x)
        x = x.view(x.shape[0], 32, -1).mean(dim=2)

        x = self._dense(x)

        return x


class Net_nonsymmetric_3l_2x2x2_narrowing(torch.nn.Module):
    def __init__(self, nx: int, ny: int, nz: int):
        super().__init__()
        self.mask_size = (2, 2, 2)
        self._conv1 = torch.nn.Conv3d(
            4, 16, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._conv2 = torch.nn.Conv3d(
            16, 12, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._conv3 = torch.nn.Conv3d(
            12, 8, self.mask_size, stride=1, padding=0, dilation=1, groups=1
        )
        self._dense1 = torch.nn.Linear(8, 8, bias=True)
        self._dense2 = torch.nn.Linear(8, 1, bias=True)

        self.nx = nx
        self.ny = ny
        self.nz = nz

    def pad_circular(
        self, x
    ):  # x[Nbatch, 1, W, H] -> x[Nbatch, 1, W + 2 pad, H + 2 pad] (pariodic padding)
        if self.nx > 1:
            x = torch.cat([x, x[:, :, 0 : self.mask_size[0] - 1, :, :]], dim=2)
        if self.ny > 1:
            x = torch.cat([x, x[:, :, :, 0 : self.mask_size[1] - 1, :]], dim=3)
        if self.nz > 1:
            x = torch.cat([x, x[:, :, :, :, 0 : self.mask_size[2] - 1]], dim=4)

        return x

    def forward(self, x):
        x = x.view((x.shape[0], 4, self.nx, self.ny, self.nz))
        x = self.pad_circular(x)
        x = self._conv1(x)
        x = torch.nn.functional.elu(x)
        x = self.pad_circular(x)
        x = self._conv2(x)
        x = torch.nn.functional.elu(x)
        x = self.pad_circular(x)
        x = self._conv3(x)
        x = torch.nn.functional.elu(x)

        x = x.view(x.shape[0], 8, -1).mean(dim=2)

        x = self._dense1(x)
        x = torch.nn.functional.elu(x)
        x = self._dense2(x)

        return x


def combine_amplitude_and_phase_all_2x2x2(*modules) -> torch.nn.Module:
    class CombiningState(torch.nn.Module):
        def __init__(self, amplitude, phase):
            super().__init__()
            self.amplitude = amplitude
            self.phase = phase
            # fmt: off
            M = np.array([24, 25, 27, 26, 29, 28, 30, 31,  8,  9, 11, 10, 13, 12, 14, 15, 16, 17, 19, 18, 21, 20, 22, 23,  0,  1,  3,  2,  5,  4,  6,  7])
            R = np.array([ 0,  2,  4,  6,  1,  3,  5,  7, 24, 26, 28, 30, 25, 27, 29, 31,  8, 10, 12, 14,  9, 11, 13, 15, 16, 18, 20, 22, 17, 19, 21, 23])
            I = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 12, 13, 14, 15,  8,  9, 10, 11, 18, 19, 16, 17, 22, 23, 20, 21, 25, 24, 27, 26, 29, 28, 31, 30])
            xy = np.array([ 0,  2,  1,  3,  4,  6,  5,  7,  8, 10,  9, 11, 12, 14, 13, 15, 24, 26, 25, 27, 28, 30, 29, 31, 16, 18, 17, 19, 20, 22, 21, 23])
            tr_x = np.array([ 4,  5,  6,  7,  0,  1,  2,  3, 12, 13, 14, 15,  8,  9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19, 28, 29, 30, 31, 24, 25, 26, 27])
            tr_y = np.array([ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9, 14, 15, 12, 13, 18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 30, 31, 28, 29])
            tr_z = np.array([ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30])
            # fmt: on

            self.symms = [np.arange(32)]
            self.symms = self.symms + [symm[tr_x] for symm in self.symms]
            self.symms = self.symms + [symm[tr_y] for symm in self.symms]
            self.symms = self.symms + [symm[tr_z] for symm in self.symms]

            self.symms = self.symms + [symm[M] for symm in self.symms]
            self.symms = self.symms + [symm[I] for symm in self.symms]
            self.symms = self.symms + [symm[xy] for symm in self.symms]
            self.symms = (
                self.symms + [symm[R] for symm in self.symms] + [symm[R[R]] for symm in self.symms]
            )

        def forward(self, x):
            ampl_log, phase = logmeanexp(
                torch.cat(
                    [self.amplitude(x[..., symm]) for symm in self.symms]
                    + [self.amplitude(-x[..., symm]) for symm in self.symms],
                    dim=1,
                ),
                torch.cat(
                    [self.phase(x[..., symm]) for symm in self.symms]
                    + [self.phase(-x[..., symm]) + np.pi for symm in self.symms],
                    dim=1,
                ),
                dim=1,
            )
            return torch.cat([ampl_log, phase], dim=1).double()

    return CombiningState(*modules)


@torch.no_grad()
def load_unsymmetrized(path="dense_largeMC_symm_0.050_1.000_onlyamplitude.data"):
    model = combine_amplitude_and_phase(Net_2x2x2_dense(), Net_2x2x2_dense())
    # model = combine_amplitude_and_phase_all_2x2x2(Net_2x2x2_dense(), Net_2x2x2_dense())
    model = model.double()
    model.load_state_dict(torch.load(path))
    for name, p in model.named_parameters():
        assert not torch.any(torch.isnan(p))
    return model

@torch.no_grad()
def load_cnn(path="3f_simplephase_0.000_1.000_onlyamplitude.data_200"):
    nx = 2
    ny = 2
    nz = 2
    model = combine_amplitude_and_phase(
        Net_nonsymmetric_3l_2x2x2_narrowing(nx, ny, nz),
        Net_nonsymmetric_1l_2x2x2_narrowing_simplephase(nx, ny, nz)
    )
    # model = combine_amplitude_and_phase_all_2x2x2(Net_2x2x2_dense(), Net_2x2x2_dense())
    model = model.double()
    model.load_state_dict(torch.load(path))
    for name, p in model.named_parameters():
        assert not torch.any(torch.isnan(p))
    return model


@torch.no_grad()
def establish_baseline():
    yaml_path = "../physical_systems/heisenberg_pyrochlore_2x2x2_no_symmetries.yaml"
    hamiltonian = nqs.load_hamiltonian(yaml_path)
    # model = load_unsymmetrized()
    model = load_cnn()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda:0")
    for name, p in model.named_parameters():
        print(name, p.device)

    def log_amplitude_fn(x):
        x = unpack_bits.unpack(x, 32).double()
        r = model(x)[:, 0]
        # if not torch.all(r < 100):
        #     print(r)
        #     assert False
        # if not torch.all(r > -100):
        #     print(r)
        #     assert False
        # assert torch.all(r < 100)
        # assert torch.all(r > -100)
        return r

    hamiltonian.basis.build()
    spins, log_probs, weights, info = nqs.sample_some(
        log_amplitude_fn,
        hamiltonian.basis,
        nqs.SamplingOptions(
            number_samples=200,
            number_chains=2,
            sweep_size=5,
            number_discarded=10,
            mode="zanella",
            device=device,
        ),
    )
    # (np.sin(0.025) * np.array([[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]])).tolist()

    def log_coeff_fn(x):
        x = unpack_bits.unpack(x, 32).double()
        r = model(x)
        return torch.complex(r[:, 0], r[:, 1])

    local_energies = nqs.local_values(spins, hamiltonian, log_coeff_fn)
    print(local_energies)
    print(torch.mean(local_energies) / 4 / 32)
    print(torch.std(local_energies) / 4 / 32)

    pass


if __name__ == "__main__":
    establish_baseline()
    # print(load_unsymmetrized())
