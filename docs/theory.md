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

The signal at transducer $i$ is modeled as

$$
s_i(t) = [A p]_i(t) * g(t) * h(t)
$$

where

- $A$ is the **acoustic propagation operator**, mapping the 3D pressure
  $p$ to the raw signals at each transducer,
- $*$ denotes time convolution,
- $g(t)$ is the **1D acoustic Green's function kernel** accounting for the
  discretization of the wave equation on a Cartesian grid,
- $h(t)$ is the **Electronic Impulse Response (EIR)** of the transducer,
  its frequency-dependent sensitivity.

The kernel $g$ derives from the 1D free-space Green's function:

$$
g(t) = 2\pi\, dx \cdot dI(ct/dx), \qquad
dI(x) = x\,(x\,\mathrm{sign}(x) - 1) \ \text{for } |x| \le 1, \ 0 \text{ otherwise.}
$$

In practice the full forward operator is

$$
F = \text{Downsample} \circ \text{IFFT} \circ [\,\cdot\, G_f H_f] \circ \text{FFT} \circ \text{Upsample} \circ A
$$

where $G_f$ and $H_f$ are the frequency-domain representations of $g$
and $h$, precomputed once at initialization with cuFFT.

## Adjoint operator

The adjoint $F^\top$, used by all iterative solvers, mirrors the forward
pass with conjugate multiplication in the frequency domain:

$$
F^\top = A^\top \circ \text{IFFT} \circ [\,\cdot\, \overline{G_f H_f}] \circ \text{FFT} \circ \text{Upsample}
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
over the element surface. For a grid point at distance $D$ from the
cylinder axis (radius $R$), the area of the transducer surface reached by
a wavefront of radius $r = ct$ involves

$$
I(\alpha, \beta; r) = \int_\alpha^\beta \sqrt{b + a\cos\theta}\, d\theta,
\qquad a = 2RD, \quad b = r^2 - D^2 - R^2.
$$

With the half-angle substitution $\theta = 2u$ and $m = 2a/(a+b)$,

$$
I = 2\sqrt{a+b}\,\bigl[E(\beta/2, \sqrt{m}) - E(\alpha/2, \sqrt{m})\bigr],
$$

where $E(\varphi, k)$ is the incomplete elliptic integral of the second
kind. When $m > 1$ (grid point closer to the cylinder than the wavefront
radius), the reciprocal-modulus transformation

$$
E(\varphi, k) = k\,E(\psi, 1/k) - \Bigl(k - \tfrac{1}{k}\Bigr) F(\psi, 1/k),
\qquad \psi = \arcsin(k \sin\varphi)
$$

reduces the problem to standard $k < 1$ integrals. On the GPU, the
integrals are evaluated with the Carlson symmetric forms
($R_f$, $R_d$, $R_c$), implemented in
`src/patminton/cuda/src/elliptic_integrals.cuh` — or replaced by a lookup table,
see [Transducer models](transducer-models.md).

## Data flow of a forward pass

```
p  ∈ ℝ^{Nx×Ny×Nz}   (GPU)
    │  compute_signal_u()      geometry-specific kernel (A·p)
    ▼
d_su ∈ ℝ^{nTrans×nTu}          upsampled raw signal
    │  cuFFT forward (batched)
    ▼
d_fs ∈ ℂ^{nTrans×fftNtu}
    │  × G_f × H_f             pointwise in frequency domain
    ▼
    │  cuFFT inverse (batched)
    ▼
    │  downsample + FFT scale
    ▼
s  ∈ ℝ^{nTrans×nT}
```

The adjoint pass mirrors this with $\overline{G_f H_f}$ and the
back-projection kernel `bp_signal_u()`.
