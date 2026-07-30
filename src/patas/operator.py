"""GPU forward and adjoint operators for 3D photoacoustic tomography.

The :class:`PAT` class wraps the CUDA library and exposes the on-the-fly
matrix-vector products ``s = A p`` (forward) and ``p = A^T s`` (adjoint)
without ever storing the system matrix ``A``. The strategy follows
Ding, Razansky and Deán-Ben, *"Model-based reconstruction of large
three-dimensional optoacoustic datasets"*, IEEE TMI, 2020.
"""

from ctypes import POINTER, c_bool, c_double, c_int, c_void_p

import numpy as np
import scipy
import torch

from ._lib import get_lib

c_double_p = POINTER(c_double)

MODES = (
    "points",
    "cylinder_exact",
    "cylinder_lut",
    "cylinder_far_field",
    "cylinder_arcs",
    "cylinder_planes",
)
"""Available transducer models (values of the ``mode`` argument)."""


class PAT:
    """Photoacoustic forward/adjoint operator on the GPU.

    An instance represents one acquisition setup: a Cartesian voxel grid, a
    time axis, a speed of sound and a set of transducers. It implements the
    matrix-vector products

    - forward: ``s = A p`` — simulate the signals from the initial pressure,
    - adjoint: ``p = A^T s`` — back-propagate signals onto the grid,

    where ``p`` is a ``(Nx, Ny, Nz)`` tensor and ``s`` a ``(nTrans, nT)``
    tensor, both ``torch.float64`` and allocated on the GPU.

    The products can be called explicitly (:meth:`PMV`, :meth:`PMVT`) with
    preallocated outputs, or through the ``@`` operator::

        s = pat @ p       # forward
        p = pat.T @ s     # adjoint

    Args:
        Nx, Ny, Nz: Grid size in voxels.
        Lx, Ly, Lz: Grid half-extent in meters (the grid spans ``[-Lx, Lx]``
            along x, etc.).
        nT: Number of time samples per transducer.
        tStart: Time of the first sample (s).
        dt: Time step (s).
        c: Speed of sound (m/s).
        mode: Transducer model, one of :data:`MODES`. See the documentation
            page *Transducer models* for the trade-offs.
        locPoints: ``(nTrans, nSensors, 3)`` array with the positions of the
            discretization points of each sensor. Required for
            ``mode='points'``.
        area: ``(nTrans, nSensors)`` array with the surface area associated
            with each discretization point. Required for ``mode='points'``.
        infos_transducers: ``(nTrans, 12)`` array describing the cylindrical
            elements (center, axis ``e1``, radial vector ``e2``, radius,
            half-aperture, half-height). Required for all ``cylinder_*``
            modes. See :func:`patas.transducers.translation_rotation_system`.
        n_arcs_per_cylinder: Number of arcs for ``mode='cylinder_arcs'``.
        n_planes_per_cylinder: Number of planes for ``mode='cylinder_planes'``.
        upsample: Time-axis oversampling factor used for the convolution with
            the Green's function kernel. Larger is more accurate but uses more
            GPU memory. Typical values: 11-51.
        steps: Coarse time-step stride in the smooth regions of the
            integration loop. Defaults to ``upsample``.
        steps_border: Fine stride near the integration boundaries
            (singularities). Defaults to ``upsample``; 1 gives maximum
            accuracy.
        blockSize: CUDA thread block edge size (3D blocks of
            ``blockSize**3`` threads).
        laser_pulse_variance: Variance (s^2) of the Gaussian envelope
            modeling the laser pulse. 0 disables it.
        eir: 1D array with the Electronic Impulse Response, sampled at
            ``dt``, with the center of the EIR at index 0 (zero-phase
            convention). ``None`` disables the EIR convolution.
        Nphi, Nk: Lookup-table resolution for ``mode='cylinder_lut'``.
        use_sparse_optimization: Enable the sparse-signal optimization in the
            CUDA kernels.

    Raises:
        ValueError: If ``mode`` is unknown or the arrays required by the
            chosen mode are missing or malformed.
        OSError: If the compiled CUDA library cannot be found.

    Example:
        >>> pat = PAT(100, 100, 100, 5e-3, 5e-3, 5e-3,
        ...           nT=512, tStart=0.0, dt=25e-9, c=1540.0,
        ...           mode='cylinder_lut', infos_transducers=infos)
        >>> s = pat @ p
        >>> p_bp = pat.T @ s
    """

    def __init__(
        self,
        Nx,
        Ny,
        Nz,
        Lx,
        Ly,
        Lz,
        nT,
        tStart,
        dt,
        c,
        mode=None,
        locPoints=None,
        area=None,
        infos_transducers=None,
        n_arcs_per_cylinder=25,
        n_planes_per_cylinder=25,
        upsample=11,
        steps=None,
        steps_border=None,
        blockSize=8,
        laser_pulse_variance=5e-9,
        eir=None,
        Nphi=1000,
        Nk=1000,
        use_sparse_optimization=False,
    ):
        self.instance = None
        self._lib = get_lib()

        _ = torch.zeros(
            1, device="cuda"
        )  # dummy init to initialize cuda, otherwise first call to cufft raise error if no tensor allocated before PAT init

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.nT = nT
        self.transpose = False

        ptr_eir = None
        # preprocessing eir: zero-padding the EIR to match the grid size: ease FFT in cuda
        if eir is not None:
            eir_padded = np.zeros(nT)
            split = (eir.shape[0] + 1) // 2
            eir_padded[:split] = eir[:split]
            eir_padded[-(eir.shape[0] - split) :] = eir[split:]
            eir_padded = np.ascontiguousarray(eir_padded)
            ptr_eir = eir_padded.ctypes.data_as(POINTER(c_double))

        if mode == "points" and locPoints is not None:
            nTrans = locPoints.shape[0]
            nSensors = locPoints.shape[1]
            self.nTrans = nTrans
            if area is None:
                raise ValueError("For point sensors you must provide 'area' value.")
            self.instance = self._lib.createPAT_points(
                c_int(Nx),
                c_int(Ny),
                c_int(Nz),
                c_double(Lx),
                c_double(Ly),
                c_double(Lz),
                c_int(nT),
                c_double(tStart),
                c_double(dt),
                c_double(c),
                area.ctypes.data_as(POINTER(c_double)),
                c_int(nTrans),
                c_int(nSensors),
                locPoints.ctypes.data_as(POINTER(c_double)),
                c_int(upsample),
                c_int(blockSize),
                c_double(laser_pulse_variance),
                ptr_eir,
                c_bool(use_sparse_optimization),
            )

        elif (
            mode
            in [
                "cylinder_exact",
                "cylinder_lut",
                "cylinder_far_field",
                "cylinder_arcs",
                "cylinder_planes",
            ]
            and infos_transducers is not None
        ):
            if infos_transducers.shape[1] != 12:
                raise ValueError(
                    "Cylinder definition must have 12 columns per transducer"
                )

            nTrans = infos_transducers.shape[0]
            self.nTrans = nTrans
            if steps_border is None:
                steps_border = upsample
            if steps is None:
                steps = upsample

            if mode == "cylinder_exact":
                self.instance = self._lib.createPAT_cylinder_exact(
                    c_int(Nx),
                    c_int(Ny),
                    c_int(Nz),
                    c_double(Lx),
                    c_double(Ly),
                    c_double(Lz),
                    c_int(nT),
                    c_double(tStart),
                    c_double(dt),
                    c_double(c),
                    c_int(nTrans),
                    infos_transducers.ctypes.data_as(POINTER(c_double)),
                    c_int(upsample),
                    c_int(steps_border),
                    c_int(steps),
                    c_int(blockSize),
                    c_double(laser_pulse_variance),
                    ptr_eir,
                    c_bool(use_sparse_optimization),
                )
            elif mode == "cylinder_lut":
                # build LUT for elliptic integrals; the axes are clustered
                # near the singularity (phi, k) -> (pi/2, 1)
                eps = 1e-16

                x = np.linspace(0, 1, Nphi)
                y = np.linspace(0, 1, Nk)

                sinphi0 = (1 - eps) * (1 - (1 - x) ** 4)
                k0 = (1 - eps) * (1 - (1 - y) ** 4)

                Phi, K = np.meshgrid(sinphi0, k0, indexing="ij")
                lut_E = scipy.special.ellipeinc(np.arcsin(Phi), K)
                lut_F = scipy.special.ellipkinc(np.arcsin(Phi), K)

                lut = np.zeros((lut_E.shape[0], lut_F.shape[1], 2))
                lut[:, :, 0] = lut_E
                lut[:, :, 1] = lut_F

                self.instance = self._lib.createPAT_cylinder_lut(
                    c_int(Nx),
                    c_int(Ny),
                    c_int(Nz),
                    c_double(Lx),
                    c_double(Ly),
                    c_double(Lz),
                    c_int(nT),
                    c_double(tStart),
                    c_double(dt),
                    c_double(c),
                    c_int(nTrans),
                    infos_transducers.ctypes.data_as(c_double_p),
                    lut.ctypes.data_as(c_double_p),
                    sinphi0.ctypes.data_as(c_double_p),
                    k0.ctypes.data_as(c_double_p),
                    c_int(Nphi),
                    c_int(Nk),
                    c_double(eps),
                    c_int(upsample),
                    c_int(steps_border),
                    c_int(steps),
                    c_int(blockSize),
                    c_double(laser_pulse_variance),
                    ptr_eir,
                    c_bool(use_sparse_optimization),
                )
            elif mode == "cylinder_far_field":
                self.instance = self._lib.createPAT_cylinder_far_field(
                    c_int(Nx),
                    c_int(Ny),
                    c_int(Nz),
                    c_double(Lx),
                    c_double(Ly),
                    c_double(Lz),
                    c_int(nT),
                    c_double(tStart),
                    c_double(dt),
                    c_double(c),
                    c_int(nTrans),
                    infos_transducers.ctypes.data_as(POINTER(c_double)),
                    c_int(upsample),
                    c_int(steps_border),
                    c_int(steps),
                    c_int(blockSize),
                    c_double(laser_pulse_variance),
                    ptr_eir,
                    c_bool(use_sparse_optimization),
                )
            elif mode == "cylinder_arcs":
                self.instance = self._lib.createPAT_cylinder_as_arcs(
                    c_int(Nx),
                    c_int(Ny),
                    c_int(Nz),
                    c_double(Lx),
                    c_double(Ly),
                    c_double(Lz),
                    c_int(nT),
                    c_double(tStart),
                    c_double(dt),
                    c_double(c),
                    c_int(nTrans),
                    infos_transducers.ctypes.data_as(POINTER(c_double)),
                    c_int(n_arcs_per_cylinder),
                    c_int(upsample),
                    c_int(steps_border),
                    c_int(steps),
                    c_int(blockSize),
                    c_double(laser_pulse_variance),
                    ptr_eir,
                    c_bool(use_sparse_optimization),
                )
            elif mode == "cylinder_planes":
                self.instance = self._lib.createPAT_cylinder_as_planes(
                    c_int(Nx),
                    c_int(Ny),
                    c_int(Nz),
                    c_double(Lx),
                    c_double(Ly),
                    c_double(Lz),
                    c_int(nT),
                    c_double(tStart),
                    c_double(dt),
                    c_double(c),
                    c_int(nTrans),
                    infos_transducers.ctypes.data_as(POINTER(c_double)),
                    c_int(n_planes_per_cylinder),
                    c_int(upsample),
                    c_int(steps_border),
                    c_int(steps),
                    c_int(blockSize),
                    c_double(laser_pulse_variance),
                    ptr_eir,
                    c_bool(use_sparse_optimization),
                )
        else:
            raise ValueError(
                "mode should be 'points', 'cylinder_exact', 'cylinder_lut', "
                "'cylinder_far_field', 'cylinder_arcs', 'cylinder_planes'"
            )

    def __del__(self):
        if getattr(self, "instance", None) is not None:
            self._lib.destroyPAT(self.instance)

    def checkEntry(self, p, t):
        """Check that a tensor is a contiguous float64 CUDA tensor of shape ``t``."""
        valid = False
        if p.is_cuda:
            if p.shape == t:
                if p.is_contiguous():
                    if p.dtype == torch.float64:
                        valid = True
                    else:
                        print("Array should a torch.float64")
                else:
                    print("Array should be contiguous")
            else:
                print("Array should be of size (%s)" % (t,))
        else:
            print(
                "Array should be allocated on the gpu before calling PMV. \n"
                " Consider using tensor.zeros(()).to(device) first"
            )
        return valid

    def PMV(self, p, s):
        """Forward product ``s = A p`` (in place).

        Args:
            p: Input pressure, ``(Nx, Ny, Nz)`` float64 CUDA tensor.
            s: Output signals, ``(nTrans, nT)`` float64 CUDA tensor,
                overwritten with the result.
        """
        if self.checkEntry(p, (self.Nx, self.Ny, self.Nz)) and self.checkEntry(
            s, (self.nTrans, self.nT)
        ):
            self._lib.PMV(self.instance, c_void_p(p.data_ptr()), c_void_p(s.data_ptr()))

    def PMVT(self, p, s):
        """Adjoint product ``p = A^T s`` (in place).

        Args:
            p: Output pressure, ``(Nx, Ny, Nz)`` float64 CUDA tensor,
                overwritten with the result.
            s: Input signals, ``(nTrans, nT)`` float64 CUDA tensor.
        """
        if self.checkEntry(p, (self.Nx, self.Ny, self.Nz)) and self.checkEntry(
            s, (self.nTrans, self.nT)
        ):
            self._lib.PMVT(self.instance, c_void_p(p.data_ptr()), c_void_p(s.data_ptr()))

    def __matmul__(self, p):
        """Compute ``pat @ p`` (forward) or ``pat.T @ s`` (adjoint).

        Allocates and returns the output tensor. The transpose flag set by
        :attr:`T` is consumed and reset by this call.
        """
        if self.transpose:
            s = torch.zeros(
                (self.Nx, self.Ny, self.Nz), device=p.device, dtype=torch.float64
            )
            self.PMVT(s, p)
        else:
            s = torch.zeros(
                (self.nTrans, self.nT), device=p.device, dtype=torch.float64
            )
            self.PMV(p, s)

        self.transpose = False
        return s

    @property
    def T(self):
        """Mark the next ``@`` product as adjoint: ``p = pat.T @ s``."""
        self.transpose = True
        return self
