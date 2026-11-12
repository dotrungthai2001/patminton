"""GPU forward and adjoint operators for 3D photoacoustic tomography.

The :class:`PAT` class wraps the CUDA library and exposes the on-the-fly
matrix-vector products ``s = A p`` (forward) and ``p = A^T s`` (adjoint)
without ever storing the system matrix ``A``.
"""

import warnings
from ctypes import POINTER, c_bool, c_double, c_int, c_void_p

import numpy as np
import scipy
import torch

from ._lib import get_lib

c_double_p = POINTER(c_double)

MODES = (
    "points",
    "plane",
    "cylinder_exact",
    "cylinder_lut",
    "cylinder_trapezoidal",
)
"""Available transducer models (values of the ``mode`` argument)."""

_STEPS_MAP_MODES = ("cylinder_exact", "cylinder_lut", "cylinder_trapezoidal")
# models whose kernels read a steps map (see PAT.set_steps_map)


def _steps_map_arrays(steps_map, steps_coords):
    """Check a steps map and return it as the contiguous arrays passed to the
    CUDA library: ``(steps_map int32 (nx, ny, nz), [x, y, z] float64)``."""
    steps_map = np.ascontiguousarray(steps_map, dtype=np.int32)
    coords = [np.ascontiguousarray(np.ravel(x), dtype=np.float64) for x in steps_coords]
    if steps_map.ndim != 3 or len(coords) != 3:
        raise ValueError("steps_map must be 3D and steps_coords = (x, y, z)")
    if tuple(x.size for x in coords) != steps_map.shape:
        raise ValueError(
            f"steps_coords sizes {tuple(x.size for x in coords)} do not match "
            f"steps_map {steps_map.shape}"
        )
    if any(np.any(np.diff(x) <= 0) for x in coords):
        raise ValueError("steps_coords must be strictly increasing")
    if steps_map.min() < 1:
        raise ValueError("steps must be >= 1")
    return steps_map, coords


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
        infos_transducers: ``(nTrans, 12)`` array describing the elements,
            flat rectangles for ``mode='plane'`` and cylindrically focused
            elements for the ``cylinder_*`` modes (see
            :mod:`patminton.transducers` for the two row formats). See
            :func:`patminton.transducers.translation_rotation_system`.
        upsample: Time-axis oversampling factor used for the convolution with
            the system kernel of the radial basis function. Larger is more
            accurate but uses more GPU memory. Typical values: 11-51.
        steps: Coarse time-step stride in the smooth regions of the
            integration loop. Defaults to ``upsample``. For the
            ``cylinder_*`` modes, ``steps`` can also be a 3D integer array:
            a map of steps in the element frame, read at the nodes
            ``steps_coords`` (see :meth:`set_steps_map`).
        steps_border: Fine stride near the integration boundaries
            (singularities). Defaults to ``upsample``, or to the step of
            each voxel when ``steps`` is a map; 1 gives maximum accuracy.
        steps_coords: ``(x, y, z)`` node coordinates (m) of the map given
            as ``steps``.
        blockSize: CUDA thread block edge size (3D blocks of
            ``blockSize**3`` threads).
        laser_pulse_variance: Standard deviation (s) of the Gaussian envelope
            modeling the laser pulse. Despite the name, the value is used as
            sigma, not sigma^2: the envelope is exp(-t^2 / (2 sigma^2)). The
            default 5e-9 is a 5 ns pulse. 0 disables it.
        eir: 1D array with the Electronic Impulse Response, sampled at
            ``dt``, with the center of the EIR at index 0 (zero-phase
            convention). ``None`` disables the EIR convolution.
        Nphi, Nk: Lookup-table resolution for ``mode='cylinder_lut'``, along
            the ``sin(theta)`` and elliptic-parameter axes respectively. Note
            that the second axis holds the *parameter* (the ``m`` argument of
            ``scipy.special.ellipeinc``), not the modulus ``k = sqrt(m)``.
        use_sparse_optimization: Enable the sparse-signal optimization in the
            CUDA kernels.

    Raises:
        ValueError: If ``mode`` is unknown or the arrays required by the
            chosen mode are missing or malformed.
        OSError: If the compiled CUDA library cannot be found.

    Example:
        >>> pat = PAT(200, 200, 200, 5e-3, 5e-3, 5e-3,
        ...           nT=1024, tStart=12.5e-6, dt=16e-9, c=1500.0,
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
        upsample=11,
        steps=None,
        steps_border=None,
        steps_coords=None,
        blockSize=8,
        laser_pulse_variance=5e-9,
        eir=None,
        Nphi=1000,
        Nk=1000,
        use_sparse_optimization=False,
    ):
        if mode == "cylinder_far_field":
            warnings.warn(
                "mode='cylinder_far_field' is renamed 'cylinder_trapezoidal'",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "cylinder_trapezoidal"

        self.instance = None
        self._lib = get_lib()

        _ = torch.zeros(
            1, device="cuda"
        )  # dummy init to initialize cuda, otherwise first call to cufft raise error if no tensor allocated before PAT init

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.nT = nT
        self.dt = dt
        self.upsample = upsample
        self.mode = mode
        self.transpose = False

        ptr_eir = None
        self.eir_padded = None  # EIR as passed to the CUDA library (length nT, center at index 0)
        # preprocessing eir: zero-padding the EIR to match the grid size: ease FFT in cuda
        if eir is not None:
            eir_padded = np.zeros(nT)
            split = (eir.shape[0] + 1) // 2
            eir_padded[:split] = eir[:split]
            eir_padded[-(eir.shape[0] - split) :] = eir[split:]
            eir_padded = np.ascontiguousarray(eir_padded)
            ptr_eir = eir_padded.ctypes.data_as(POINTER(c_double))
            self.eir_padded = eir_padded

        steps_map = None
        if steps is not None and np.ndim(steps) > 0:
            # map of steps: one value per voxel, read in the transducer frame
            if mode not in _STEPS_MAP_MODES:
                raise ValueError(
                    "a steps map is only supported by 'cylinder_exact', "
                    "'cylinder_lut' and 'cylinder_trapezoidal'"
                )
            if steps_coords is None:
                raise ValueError("a steps map needs steps_coords = (x, y, z)")
            steps_map, steps_coords = _steps_map_arrays(steps, steps_coords)
            steps = int(steps_map.max())  # scalar steps of the C++ object, unused with a map
            if steps_border is None:
                steps_border = 0  # 0: steps_border = steps of the voxel, read in the map

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
                "plane",
                "cylinder_exact",
                "cylinder_lut",
                "cylinder_trapezoidal",
            ]
            and infos_transducers is not None
        ):
            if infos_transducers.shape[1] != 12:
                raise ValueError(
                    "Element definition must have 12 columns per transducer"
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
                # build LUT for elliptic integrals E and F, tabulated as
                # functions of (sin(theta), nu) on [0,1]^2, where nu is the
                # parameter (scipy's `m`), not the modulus. The axes are
                # clustered near the singularity (sin(theta), nu) -> (1, 1),
                # where F diverges logarithmically.
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
            elif mode == "cylinder_trapezoidal":
                self.instance = self._lib.createPAT_cylinder_trapezoidal(
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
            elif mode == "plane":
                self.instance = self._lib.createPAT_plane(
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

            if steps_map is not None:
                self.set_steps_map(steps_map, steps_coords)
        else:
            raise ValueError(
                "mode should be 'points', 'plane', 'cylinder_exact', "
                "'cylinder_lut', 'cylinder_trapezoidal'"
            )

    def __del__(self):
        if getattr(self, "instance", None) is not None:
            self._lib.destroyPAT(self.instance)

    def set_steps_map(self, steps_map, steps_coords):
        """Use a map of steps instead of the scalar ``steps``.

        For each element, every voxel is converted to the local frame of that
        element and uses the step of the nearest node of the map (positions
        outside the map take the value of its border), so one map serves all
        elements of the same shape. ``steps_border`` keeps the value set at
        construction: with a map passed to the constructor and
        ``steps_border=None``, each voxel uses its own step there too.
        Replaces any previous map.

        Args:
            steps_map: ``(nx, ny, nz)`` integers >= 1, the steps in upsampled
                time bins (``dt / upsample``).
            steps_coords: ``(x, y, z)`` strictly increasing 1D arrays of
                sizes ``nx``, ``ny``, ``nz``: node coordinates (m) in the
                element frame, with x along ``e2`` (toward the middle of the
                arc), y along ``e1 x e2``, z along the cylinder axis ``e1``,
                and the origin at the cylinder center ``c``.

        Raises:
            ValueError: If the mode is not ``cylinder_exact``,
                ``cylinder_lut`` or ``cylinder_trapezoidal``, or the arrays
                are malformed.
        """
        if self.mode not in _STEPS_MAP_MODES:
            raise ValueError(
                "a steps map is only supported by 'cylinder_exact', "
                "'cylinder_lut' and 'cylinder_trapezoidal'"
            )
        steps_map, coords = _steps_map_arrays(steps_map, steps_coords)
        if self._lib.setStepsMap(
            self.instance,
            steps_map.ctypes.data_as(POINTER(c_int)),
            *[x.ctypes.data_as(c_double_p) for x in coords],
            *[c_int(n) for n in steps_map.shape],
        ):
            raise ValueError("steps map only for the elliptic cylinder models")
        self.steps_map, self.steps_coords = steps_map, coords

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

    @property
    def nTu(self):
        """Number of upsampled time samples, ``upsample * nT``."""
        return self._lib.getNtu(self.instance)

    def get_signal_u(self, su=None):
        """Copy of the upsampled signal buffer ``d_su_``, ``(nTrans, nTu)``.

        After a forward product it holds the convolved upsampled signal,
        without the ``1 / nTu`` FFT scaling:
        ``s = su.reshape(nTrans, nT, upsample).mean(-1) / nTu``. After an
        adjoint product it holds the scaled correlated signal fed to the
        back-projection; for the ``cylinder_*`` modes, its suffix sums over
        time, which their back-projection reads.

        Args:
            su: Optional preallocated ``(nTrans, nTu)`` float64 CUDA tensor,
                filled in place.

        Returns:
            The tensor ``su``.
        """
        if su is None:
            su = torch.empty((self.nTrans, self.nTu), device="cuda", dtype=torch.float64)
        if self.checkEntry(su, (self.nTrans, self.nTu)):
            self._lib.getSignalU(self.instance, c_void_p(su.data_ptr()))
        return su

    def compute_signal_u(self, p, su=None):
        """Upsampled signal before the convolution with the system kernel.

        Runs only the geometry kernel, so that ``pat @ p`` equals
        ``irfft(rfft(u) * get_kernel_fft(), n=nTu).reshape(nTrans, nT,
        upsample).mean(-1)`` (``torch.fft`` conventions). For a unit impulse
        at one voxel, ``u`` is the spatial impulse response (SIR) of that
        voxel, averaged over each upsampled time bin and up to a constant
        factor. Overwrites ``d_su_``.

        Args:
            p: Input pressure, ``(Nx, Ny, Nz)`` float64 CUDA tensor.
            su: Optional preallocated ``(nTrans, nTu)`` float64 CUDA tensor,
                filled in place.

        Returns:
            The ``(nTrans, nTu)`` tensor ``su``.
        """
        if su is None:
            su = torch.empty((self.nTrans, self.nTu), device=p.device, dtype=torch.float64)
        if self.checkEntry(p, (self.Nx, self.Ny, self.Nz)) and self.checkEntry(
            su, (self.nTrans, self.nTu)
        ):
            self._lib.computeSignalU(
                self.instance, c_void_p(p.data_ptr()), c_void_p(su.data_ptr())
            )
        return su

    def get_kernel_fft(self):
        """Copy of the convolution kernel used by the forward and adjoint products.

        Returns:
            ``complex128`` CUDA tensor of size ``nTu // 2 + 1``: the
            unnormalized real FFT of :meth:`get_kernel`.
        """
        fg = torch.empty(self.nTu // 2 + 1, device="cuda", dtype=torch.complex128)
        self._lib.getKernel(self.instance, c_void_p(fg.data_ptr()))
        return fg

    def get_kernel(self):
        """Convolution kernel in time, on the upsampled grid (step ``dt / upsample``).

        The kernel is the system kernel of the radial basis function,
        convolved with the Gaussian laser pulse and the EIR when they are
        set, computed on a 101 times finer grid and averaged back. It is
        circular, with zero lag at index 0 and negative lags at the end;
        ``torch.fft.fftshift`` centers it.

        Returns:
            float64 CUDA tensor of size ``nTu``.
        """
        return torch.fft.irfft(self.get_kernel_fft(), n=self.nTu)

    def get_upsampled_eir(self):
        """EIR on the upsampled grid, interpolated as in the CUDA library.

        The real FFT of the zero-padded EIR (length ``nT``) is zero-padded to
        ``nTu``, that is, band-limited interpolation. With this scaling the
        sum equals the sum of the EIR (amplitude ``eir / upsample``), so the
        result is a discrete kernel on the upsampled grid.

        Returns:
            ``(nTu,)`` NumPy array with zero lag at index 0, or ``None`` if
            no EIR was given.
        """
        if self.eir_padded is None:
            return None
        return np.fft.irfft(np.fft.rfft(self.eir_padded), n=self.nTu)

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
