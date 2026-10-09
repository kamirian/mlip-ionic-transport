"""iondiff: diffusivity, statistical error and Arrhenius analysis of MD trajectories.

Implements the statistical-variance analysis of
  X. He, Y. Zhu, A. Epstein, Y. Mo, "Statistical variances of diffusional properties from
  ab initio molecular dynamics simulations", npj Comput. Mater. 4, 18 (2018),
adapted from the MoGroup (UMD) diffusion module by Xingfeng He (MIT License), built on pymatgen.
Please cite the paper above when using this package.
"""

__version__ = "0.1.0"

from .diffusion import (  # noqa: E402,F401
    ArrheniusAnalyzer,
    DiffusivityAnalyzer,
    ErrorAnalysis,
    get_conversion_factor,
    one_ion_msd_fft,
    unwrap_displacements,
)
