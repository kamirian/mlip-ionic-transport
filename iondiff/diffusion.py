# Copyright (c) MoGroup at UMD. Distributed under the terms of the MIT License.
#
# Adapted from the diffusion module originally written by Xingfeng He (MoGroup, UMD, 2017),
# which implements the methods of:
#   X. He, Y. Zhu, A. Epstein, Y. Mo, "Statistical variances of diffusional properties from
#   ab initio molecular dynamics simulations", npj Comput. Mater. 4, 18 (2018).
# Built on pymatgen (Pymatgen Development Team).
#
# Changes relative to the original module:
#   * MSD algorithm (FFT autocorrelation, framework-drift removal, fitting window) is unchanged;
#     the S1 term is vectorized (numerically identical, checked in tests/).
#   * Displacements can be built from per-frame lattices, so NPT trajectories are handled
#     correctly (the original used the first frame's lattice for every frame).
#   * ErrorAnalysis summary: the per-axis *relative* standard deviation is no longer overwritten
#     by the per-axis absolute standard deviation (both are now returned under separate keys).
#   * Arrhenius fitting takes explicit diffusivity standard deviations; there is no placeholder.
"""Diffusivity, statistical error and Arrhenius analysis of MD trajectories."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Boltzmann constant in eV/K, as used in the original module.
K_B_EV = 8.617e-5
# Physical constants used in the Nernst-Einstein conversion (as in the original module).
_N_A = 6.022140857e23
_E_CHARGE = 1.6021766208e-19
_R_GAS = 8.3144598

DEFAULT_SPEC = {"lower_bound": 4.5, "upper_bound": 0.5, "minimum_msd_diff": 4.5}


def _autocorrelation_fft(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    f = np.fft.fft(x, n=2 * n)
    res = np.fft.ifft(f * f.conjugate())[:n].real
    return res / (n - np.arange(n))


def one_ion_msd_fft(r: np.ndarray, dt_indices: np.ndarray):
    """MSD of one ion for every lag in ``dt_indices``.

    r: unwrapped Cartesian displacement of one ion, shape (n_steps, 3).
    Returns (msd, msd_component) with shapes (n_dt,) and (3, n_dt).
    """
    n_step, dim = r.shape
    r2 = np.square(r)
    total = r2.sum(axis=0)
    # S1[m] = (sum_{k=0}^{N-m-1} r2[k+m] + r2[k]) / (N - m), computed with cumulative sums.
    csum = np.cumsum(r2, axis=0)
    front = np.vstack([np.zeros((1, dim)), csum[:-1]])                   # sum_{k<m} r2[k]
    back = np.vstack([np.zeros((1, dim)), (total - csum[::-1][1:])[:n_step - 1]])  # sum of last m rows
    q = 2 * total[None, :] - front - back
    s1_component = (q / (n_step - np.arange(n_step))[:, None]).T        # (dim, N)
    s2_component = np.array([_autocorrelation_fft(r[:, i]) for i in range(dim)])
    msd_component = (s1_component - 2 * s2_component)[:, dt_indices]
    return msd_component.sum(axis=0), msd_component


def unwrap_displacements(frac_coords: np.ndarray, lattices: np.ndarray) -> np.ndarray:
    """Unwrapped Cartesian displacements from wrapped fractional coordinates.

    frac_coords: (n_frames, n_atoms, 3). lattices: (n_frames, 3, 3) row-vector lattice matrices
    (a single (3, 3) matrix is broadcast for NVT). Each frame-to-frame step uses the minimum
    image in fractional space and is converted to Cartesian with the lattice of the later frame.
    Returns displacements with shape (n_atoms, n_frames, 3); frame 0 has zero displacement.
    """
    frac_coords = np.asarray(frac_coords, dtype=float)
    lattices = np.asarray(lattices, dtype=float)
    if lattices.ndim == 2:
        lattices = np.broadcast_to(lattices, (frac_coords.shape[0], 3, 3))
    dfrac = np.diff(frac_coords, axis=0)
    dfrac -= np.round(dfrac)
    dcart = np.einsum("fai,fij->faj", dfrac, lattices[1:])
    disp = np.concatenate([np.zeros_like(dcart[:1]), np.cumsum(dcart, axis=0)], axis=0)
    return disp.transpose(1, 0, 2)


class DiffusivityAnalyzer:
    """MSD and diffusivity of one species from pre-processed displacements.

    Parameters follow the original module:
      symbols: chemical symbol of every site (length n_atoms).
      displacements: (n_atoms, n_steps, 3) unwrapped Cartesian displacements in Angstrom.
      specie: diffusing element symbol, e.g. "Li".
      temperature: K.  time_step: fs between stored frames (after any step_skip).
      spec_dict: fitting window. lower_bound [A^2], upper_bound [fraction of total time],
        minimum_msd_diff [A^2].
    """

    def __init__(self, symbols, displacements, specie, temperature, time_step,
                 step_skip=1, time_intervals_number=1000, spec_dict=None):
        spec_dict = dict(DEFAULT_SPEC if spec_dict is None else spec_dict)
        if not {"lower_bound", "upper_bound", "minimum_msd_diff"} <= set(spec_dict):
            raise ValueError("spec_dict needs lower_bound, upper_bound and minimum_msd_diff")
        symbols = list(symbols)
        indices = [i for i, s in enumerate(symbols) if s == specie]
        framework_indices = [i for i, s in enumerate(symbols) if s != specie]
        if not indices:
            raise ValueError(f"There is no specie {specie} in the structure")

        displacements = np.asarray(displacements, dtype=float)
        if framework_indices:
            drift = np.average(displacements[framework_indices], axis=0)[None, :, :]
            dc = displacements - drift
        else:
            dc = displacements
        disp_ions = dc[indices]
        n_ions, n_steps, dim = disp_ions.shape

        time_step_displacements = time_step * step_skip
        dt_indices = np.arange(1, n_steps, max(int((n_steps - 1) / time_intervals_number), 1))
        dt = dt_indices * time_step_displacements

        msd_by_ions = np.empty((n_ions, len(dt_indices)))
        msd_component_by_ions = np.empty((3, n_ions, len(dt_indices)))
        for i in range(n_ions):
            msd_by_ions[i], msd_component_by_ions[:, i, :] = one_ion_msd_fft(disp_ions[i], dt_indices)
        msd = msd_by_ions.mean(axis=0)
        msd_component = msd_component_by_ions.mean(axis=1)

        lower_bound_index = len(msd[msd < spec_dict["lower_bound"]])
        upper_bound_index = int(len(msd) * spec_dict["upper_bound"]) - 1
        if (lower_bound_index >= upper_bound_index - 2
                or msd[upper_bound_index] - msd[lower_bound_index] < spec_dict["minimum_msd_diff"]):
            slope = -1.0
            slope_components = np.zeros(dim)
            fit_ok = False
        else:
            window = slice(lower_bound_index, upper_bound_index + 1)
            slope = stats.linregress(dt[window], msd[window]).slope
            slope_components = np.array(
                [stats.linregress(dt[window], msd_component[i, window]).slope for i in range(dim)])
            fit_ok = True

        self.symbols = symbols
        self.indices = indices
        self.framework_indices = framework_indices
        self.specie = specie
        self.temperature = temperature
        self.time_step = time_step
        self.step_skip = step_skip
        self.time_intervals_number = time_intervals_number
        self.spec_dict = spec_dict
        self.n_steps = n_steps
        self.dt = dt
        self.msd = msd
        self.msd_by_ions = msd_by_ions
        self.msd_component = msd_component
        self.total_msd = msd_by_ions.sum(axis=0)
        self.lower_bound_index = lower_bound_index
        self.upper_bound_index = upper_bound_index
        self.fit_ok = fit_ok
        if framework_indices:
            max_ion_disp = np.max(np.linalg.norm(dc, axis=-1), axis=1)
            fw = max_ion_disp[framework_indices]
            self.max_framework_displacement = float(np.max(fw))
            # Diagnostics that separate collective melting from isolated framework hops:
            # fraction of framework atoms that ever moved > 4 A, and the framework MSD (time-averaged,
            # same lags) at the end of the fitting window.
            self.framework_fraction_moved_4A = float(np.mean(fw > 4.0))
            fw_msd = np.mean([one_ion_msd_fft(dc[i], dt_indices[upper_bound_index:upper_bound_index + 1])[0][0]
                              for i in framework_indices]) if upper_bound_index >= 0 else float("nan")
            self.framework_msd_at_fit_end = float(fw_msd)
        else:
            self.max_framework_displacement = 0.0
            self.framework_fraction_moved_4A = 0.0
            self.framework_msd_at_fit_end = 0.0
        # MSD in A^2, dt in fs: D [cm^2/s] = slope / (2 * dim) * 1e-16 / 1e-15 = slope / (20 * dim)
        self.diffusivity = slope / (20 * dim)
        self.diffusivity_components = slope_components / 20

    @classmethod
    def from_frames(cls, symbols, frac_coords, lattices, specie, temperature, time_step,
                    **kwargs):
        """Build from wrapped fractional coordinates and (per-frame or single) lattices."""
        return cls(symbols, unwrap_displacements(frac_coords, lattices), specie, temperature,
                   time_step, **kwargs)


class ErrorAnalysis:
    """Statistical error of D from the number of ion jumps (He et al. 2018):

        RSD = 3.43 / sqrt(N_jump) + 0.04,   N_jump = n_ions * max(MSD) / a^2

    where ``a`` is the (average) site distance of the diffusing species.
    """

    def __init__(self, analyzer: DiffusivityAnalyzer, site_distance: float = 3.0):
        a2 = site_distance * site_distance
        n_ions = len(analyzer.indices)
        self.n_jump = n_ions * np.max(analyzer.msd) / a2
        self.n_jump_component = n_ions * np.max(analyzer.msd_component, axis=1) / a2
        self.rsd = 3.43 / np.sqrt(self.n_jump) + 0.04
        self.rsd_component = 3.43 / np.sqrt(self.n_jump_component) + 0.04
        self.analyzer = analyzer
        self.site_distance = site_distance

    @property
    def diffusivity_std(self) -> float:
        return float(self.rsd * self.analyzer.diffusivity)

    @property
    def diffusivity_component_std(self) -> np.ndarray:
        return self.rsd_component * self.analyzer.diffusivity_components


def get_conversion_factor(n_specie: int, volume_A3: float, charge: int, temperature: float) -> float:
    """Nernst-Einstein factor: conductivity [mS/cm] = factor * D [cm^2/s].

    n_specie: number of diffusing ions in the cell; volume_A3: cell volume in A^3;
    charge: oxidation state of the diffusing ion.
    """
    vol_cm3 = volume_A3 * 1e-24
    return 1000 * n_specie / (vol_cm3 * _N_A) * charge ** 2 * (_N_A * _E_CHARGE) ** 2 / (_R_GAS * temperature)


class ArrheniusAnalyzer:
    """Fit log10(D) = slope * (1000 / T) + intercept, Ea = -k_B ln(10) 1000 * slope.

    If ``diffusivity_errors`` (absolute standard deviations of D) are given, the fit is weighted
    with sigma_log10D = log10(e) * sigma_D / D and absolute_sigma=True, as in the original module.
    """

    def __init__(self, temperatures, diffusivities, diffusivity_errors=None):
        self.temperatures = np.asarray(temperatures, dtype=float)
        self.diffusivities = np.asarray(diffusivities, dtype=float)
        self.diffusivity_errors = (None if diffusivity_errors is None
                                   else np.asarray(diffusivity_errors, dtype=float))
        if len(self.temperatures) < 2:
            raise ValueError("Need at least two temperatures for an Arrhenius fit")
        self.x = 1000.0 / self.temperatures
        self.y = np.log10(self.diffusivities)
        slope_to_ev = -K_B_EV * 1000 * np.log(10)

        def linear(x, k, b):
            return k * x + b

        if self.diffusivity_errors is None:
            (k, b), cov = curve_fit(linear, self.x, self.y)
            self.y_error = None
        else:
            self.y_error = np.log10(np.e) * self.diffusivity_errors / self.diffusivities
            (k, b), cov = curve_fit(linear, self.x, self.y, sigma=self.y_error, absolute_sigma=True)
        self.slope, self.intercept = float(k), float(b)
        self.slope_sigma, self.intercept_sigma = (float(v) for v in np.sqrt(np.diag(cov)))
        self.Ea = slope_to_ev * self.slope
        self.Ea_error = -slope_to_ev * self.slope_sigma
        resid = self.y - linear(self.x, k, b)
        ss_tot = np.sum((self.y - self.y.mean()) ** 2)
        self.r_squared = float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    def predict_diffusivity(self, temperature: float):
        """D at ``temperature`` and the [min, max] range from the fit uncertainty."""
        log_d = self.slope * (1000.0 / temperature) + self.intercept
        log_d_sigma = np.hypot(self.slope_sigma * (1000.0 / temperature), self.intercept_sigma)
        return 10 ** log_d, [10 ** (log_d - log_d_sigma), 10 ** (log_d + log_d_sigma)]

    def predict_conductivity(self, temperature, n_specie, volume_A3, charge):
        d, (d_min, d_max) = self.predict_diffusivity(temperature)
        f = get_conversion_factor(n_specie, volume_A3, charge, temperature)
        return f * d, [f * d_min, f * d_max]
