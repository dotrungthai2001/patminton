"""patminton — GPU-accelerated 3D photoacoustic tomography operators.

CUDA implementations of the forward and adjoint operators of 3D
photoacoustic tomography (PAT) as on-the-fly matrix-vector products, plus
iterative reconstruction algorithms built on top of them.
"""

from .operator import MODES, PAT
from .transducers import (
    discretize_cylindrical_transducers,
    discretize_planar_transducers,
    translation_rotation_system,
)
from .algorithms import (
    PSNR,
    SNR,
    compute_ssim_3d,
    estimate_lipschitz,
    least_squares_CG,
    least_squares_CP_TV,
    least_squares_LBFGSB,
    least_squares_PGD,
    normalize_operator,
)

__version__ = "0.1.1"

__all__ = [
    "PAT",
    "MODES",
    "translation_rotation_system",
    "discretize_cylindrical_transducers",
    "discretize_planar_transducers",
    "least_squares_CG",
    "least_squares_LBFGSB",
    "least_squares_PGD",
    "least_squares_CP_TV",
    "estimate_lipschitz",
    "normalize_operator",
    "SNR",
    "PSNR",
    "compute_ssim_3d",
    "__version__",
]
