# Physical and mathematical model

## Physical problem

A short laser pulse illuminates biological tissue. The absorbed optical
energy causes a rapid thermoelastic expansion that generates a broadband
ultrasonic pressure wave, which propagates through the tissue and is recorded
at the surface by an array of ultrasound transducers.

The scanner considered here consists of cylindrical piezoelectric elements on
a rotating and translating probe that circles the sample. Each element
integrates the pressure field over its surface, so the received signal
depends on the element geometry and its position relative to the tissue.

**Goal:** recover the 3D initial pressure distribution $p(x)$ — which
reflects the optical absorption map — from the time-domain signals $s(t)$
of all transducers.

## Forward operator

The initial pressure is discretized as a finite sum of radial functions
(a Kansa representation),

$$
p_0(x) = \sum_i \mathbf{p}_0[i]\, \phi(\|x - x_i\|),
$$

over a uniform Cartesian grid $\{x_i\}$. Because $\phi$ is radially
symmetric, wave propagation and interpolation decouple: each voxel can be
treated as an isolated point source, and the effect of $\phi$ is a single
**time convolution** that does not depend on the direction $x - x_i$. The
signal at transducer $k$ is then

$$
s_k(t) = [A \mathbf{p}_0]_k(t) * h(t) * e(t)
$$

where

- $A$ is the **acoustic propagation operator**, mapping the discrete
  pressure $\mathbf{p}_0$ to the raw signals at each transducer,
- $*$ denotes time convolution,
- $h(t)$ is the **system kernel** carrying the radial function $\phi$,
- $e(t)$ is the **Electronic Impulse Response (EIR)** of the transducer,
  its frequency-dependent sensitivity.

The system kernel is $h(t) = -\tfrac{1}{2} c t\, \phi(|ct|)$, supported on
$|ct| \le \kappa$ with $\kappa$ the support radius of $\phi$. `patminton`
uses the linear (hat) radial function of support $dx$, for which $h$
reduces to

$$
h(t) \propto dI(ct/dx), \qquad
dI(x) = x\,(x\,\mathrm{sign}(x) - 1) \ \text{for } |x| \le 1, \ 0 \text{ otherwise,}
$$

implemented as `g = 2*M_PI*dx*dI(c*t/dx)` in `initConvKernel`. The
$4\pi$ between that expression and $-\tfrac{1}{2}ct\,\phi(|ct|)$ is a
global constant: it scales $A$ and its adjoint identically and therefore
cancels in any reconstruction, but it means absolute signal amplitudes are
not in pascals.

In practice the full forward operator is

$$
F = \text{Downsample} \circ \text{IFFT} \circ [\,\cdot\, H_f E_f] \circ \text{FFT} \circ \text{Upsample} \circ A
$$

where $H_f$ and $E_f$ are the frequency-domain representations of $h$
and $e$, precomputed once at initialization with cuFFT. The laser pulse
envelope, when enabled, is folded into the same product.

## Adjoint operator

The adjoint $F^\top$, used by all iterative solvers, mirrors the forward
pass with conjugate multiplication in the frequency domain:

$$
F^\top = A^\top \circ \text{IFFT} \circ [\,\cdot\, \overline{H_f E_f}] \circ \text{FFT} \circ \text{Upsample}
$$

Adjoint consistency $\langle Fp, s\rangle = \langle p, F^\top s\rangle$ is
verified by a dot-product test (see [Quickstart](quickstart.md#check-the-adjoint)).

## Reconstruction problem

The inverse problem is formulated as regularized least squares:

$$
\min_u \; \tfrac{1}{2}\|F u - s\|^2 + \tfrac{\lambda}{2}\|u\|^2,
$$

optionally with the constraint $u \ge 0$ (pressure is non-negative in a
purely absorbing medium) or a total-variation term. See
[Reconstruction algorithms](solvers.md).

## Elliptic integrals for cylindrical transducers

The operator $A$ for a cylindrical element integrates the pressure field
over the element surface. For a grid point at distance $\rho$ from the
cylinder axis (radius $R$), the area of the transducer surface reached by
a wavefront of radius $r = ct$ involves the arc integral

$$
I(\alpha, \beta; r) = \int_\alpha^\beta \sqrt{b(r) + a\cos u}\, du,
\qquad a = 2R\rho, \quad b(r) = r^2 - \rho^2 - R^2.
$$

With the half-angle substitution $u = 2t$ and the parameter
$\nu(r) = 2a/(a + b(r)) = 4R\rho/(2R\rho + b(r))$,

$$
I = 2\sqrt{a + b(r)}\,\bigl[E(\beta/2, \nu) - E(\alpha/2, \nu)\bigr],
\qquad E(\vartheta, \nu) = \int_0^\vartheta \sqrt{1 - \nu \sin^2 t}\, dt,
$$

where $E$ is the incomplete elliptic integral of the second kind. Note that
$\nu$ is the **parameter** — the $m$ argument of `scipy.special.ellipeinc`,
not the modulus $k = \sqrt{\nu}$; the lookup table is sampled on this same
parameter axis.

Since $2R\rho + b(r) = r^2 - (R - \rho)^2$, one has $\nu(r) > 1$ exactly
when $r < R + \rho$, that is, as long as the wavefront has not yet reached
the far edge of the circle of radius $R$ in the plane of the grid point.
This happens for part of the time steps of every voxel, and standard
routines require $\nu \in [0,1]$, so the reciprocal-modulus transformation

$$
E(\vartheta, \nu) = \sqrt{\nu}\Bigl[E\bigl(\tilde\vartheta, \tfrac{1}{\nu}\bigr)
- \bigl(1 - \tfrac{1}{\nu}\bigr) F\bigl(\tilde\vartheta, \tfrac{1}{\nu}\bigr)\Bigr],
\qquad \tilde\vartheta = \arcsin\bigl(\sqrt{\nu}\,\sin\vartheta\bigr)
$$

brings the parameter back into $[0,1]$, at the cost of one incomplete
elliptic integral of the first kind $F$. On the GPU, the integrals are
evaluated with the Carlson symmetric forms ($R_f$, $R_d$, $R_c$),
implemented in `src/patminton/cuda/src/elliptic_integrals.cuh` — or replaced
by a lookup table, see [Transducer models](transducer-models.md).

## Data flow of a forward pass

```
p  ∈ ℝ^{Nx×Ny×Nz}   (GPU)
    │  compute_signal_u()      geometry-specific kernel (A·p)
    ▼
d_su ∈ ℝ^{nTrans×nTu}          upsampled raw signal
    │  cuFFT forward (batched)
    ▼
d_fs ∈ ℂ^{nTrans×fftNtu}
    │  × H_f × E_f             pointwise in frequency domain
    ▼
    │  cuFFT inverse (batched)
    ▼
    │  downsample + FFT scale
    ▼
s  ∈ ℝ^{nTrans×nT}
```

The adjoint pass mirrors this with $\overline{H_f E_f}$ and the
back-projection kernel `bp_signal_u()`.
