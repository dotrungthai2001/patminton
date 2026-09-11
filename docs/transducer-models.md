# Transducer models

## Element parametrization

Each cylindrical element is described by 12 parameters, one row of the
`infos_transducers` array (used by all `cylinder_*` modes):

| Index | Parameter | Description |
| ----- | --------- | ----------- |
| 0–2 | $(x_c, y_c, z_c)$ | Center position (m) |
| 3–5 | $e_1$ | Axis unit vector (cylinder axis) |
| 6–8 | $e_2$ | Radial unit vector (normal to axis, toward imaging center) |
| 9 | $R$ | Cylinder radius (m) |
| 10 | $\theta_{\max}$ | Half-aperture angle (rad) |
| 11 | $h$ | Half-height of the element (m) |

`translation_rotation_system()`
builds this array for a rotation + translation scan.

## Available modes

Six forward/adjoint implementations are selected via the `mode` argument of
`PAT`. `points` applies to any transducer geometry; the five `cylinder_*`
modes are specific to cylindrically focused elements and take the
`infos_transducers` array described above.

### `points`

The transducer surface is discretized into $Q$ point sensors (built with
`discretize_cylindrical_transducers()`);
each point contributes a $1/r$-weighted pressure sample, i.e. the surface
integral is replaced by a quadrature rule. This applies to any geometry and
is the simplest to implement, but its error carries an extra $O(Q^{-1})$
quadrature term that the `cylinder_*` modes do not have — those evaluate
the surface integral in closed form — so accuracy is governed by two
parameters ($Q$ and `upsample`) instead of one, and fine discretizations
get expensive.

### `cylinder_exact`

The surface integral is evaluated analytically at each time step with
**Carlson symmetric elliptic integrals** computed on the GPU. Most accurate;
slower than the LUT variant because the Carlson iteration runs per thread at
every time step.

### `cylinder_lut`

Same formula as `cylinder_exact`, but the incomplete elliptic integrals
$E(\vartheta, \nu)$ and $F(\vartheta, \nu)$ are replaced by **bilinear
interpolation in a precomputed lookup table** (LUT). Both are tabulated as
functions of $(\sin\vartheta, \nu)$ on $[0,1]^2$ — where $\nu$ is the
*parameter*, not the modulus — on a non-uniform grid clustered near the
logarithmic singularity of $F$ at $(\sin\vartheta, \nu) \to (1, 1)$:

$$
\sin\vartheta = (1-\varepsilon)\bigl(1 - (1-x)^4\bigr), \qquad
\nu = (1-\varepsilon)\bigl(1 - (1-y)^4\bigr), \qquad x, y \in [0, 1],
$$

with $\varepsilon = 10^{-16}$. Tabulating in $\sin\vartheta$ rather than
$\vartheta$ saves an `arcsin` at query time; the price of the non-uniform
axes is that locating the surrounding grid cell requires inverting the
sampling law, i.e. two square roots.

The LUT is built in Python with `scipy.special.ellipeinc` / `ellipkinc` and
transferred to the GPU at initialization. At the default 1000×1000 the
interpolation error is below single precision, so the LUT **matches the
accuracy of `cylinder_exact`** while running about 10× faster, and is also
about 3× faster than `points`. **Recommended default.**

### `cylinder_far_field`

Despite its name, this mode makes no far-field assumption. It replaces the
arc integral by the **trapezoidal rule**: when $\alpha_l$ and $\beta_l$ are
close, the curve $u \mapsto \Psi(u, c\tau_l)$ is approximated by the chord
joining its endpoints,

$$
I_l \approx \tfrac{1}{2}(\beta_l - \alpha_l)\bigl[\Psi(\alpha_l, c\tau_l)
+ \Psi(\beta_l, c\tau_l)\bigr].
$$

No elliptic integral or table is needed — two square roots per time step.
This is the **fastest mode overall** (about 20 % faster than
`cylinder_lut` on a fine grid, 50 % on a coarse one). The error is of
relative order $(\beta_l - \alpha_l)^2$ and, unlike the temporal
quantization error, is *not* reduced by increasing `upsample`: it is set by
the transducer geometry and the incidence of the wavefront, and grows near
grazing incidence. In practice this costs about 0.3 dB of reconstruction
PSNR against `cylinder_exact`.

### `cylinder_planes`

Piecewise-planar approximation of the cylinder. The angular aperture
$[-\theta_{\max}, \theta_{\max}]$ is split into `n_planes_per_cylinder`
sectors of width $\Delta\theta$, and each sector is replaced by the plane
tangent to the cylinder at its central angle — a rectangular element of
half-width $\tfrac{1}{2}R\Delta\theta$ and half-height $h$. The
contributions of the patches are summed. The error comes from the curvature
neglected inside each patch and vanishes as $\Delta\theta \to 0$, so
accuracy is controlled by the user, but the cost grows with the number of
patches. Measured on the reference geometry it is expensive *and* less
accurate than `cylinder_lut` (about 0.7 dB of PSNR below
`cylinder_exact`).

### `cylinder_arcs`

The element is decomposed into a stack of `n_arcs_per_cylinder` **arcs at
fixed heights** along the cylinder axis, each integrated independently over
the angle. This mode is a validation variant, not one of the operators
benchmarked in the paper.

## Choosing a mode

Accuracy is quoted as the reconstruction PSNR relative to
`cylinder_exact`, on the fine noiseless grid of Experiment 1 of the paper;
speed is relative to `cylinder_lut`.

| Mode | Accuracy | Speed | Use case |
| ---- | -------- | ----- | -------- |
| `cylinder_lut` | matches `cylinder_exact` | 1× | **default** for reconstruction |
| `cylinder_exact` | reference | ~10× slower | reference / validation |
| `cylinder_far_field` (trapezoidal) | −0.3 dB | 1.2–1.5× faster | fastest; when the bias is acceptable |
| `cylinder_planes` (piecewise plane) | −0.7 dB | slower | tunable via `n_planes_per_cylinder` |
| `points` | −1.2 dB | ~3× slower | arbitrary (non-cylindrical) geometries |
| `cylinder_arcs` | not benchmarked | moderate | validation variant |

With 1 % measurement noise the five operators become quantitatively
comparable; the differences above are visible mainly in the noiseless
fine-grid regime.
