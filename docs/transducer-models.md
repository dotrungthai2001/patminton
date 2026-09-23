# Transducer models

Every mode computes the same quantity: the pressure integrated over the
element surface. The modes differ in the element geometry they describe and
in how they evaluate the surface integral.

| Geometry | Mode | Surface integral |
| -------- | ---- | ---------------- |
| Flat rectangle | `plane` | closed form |
| Cylindrically focused | `cylinder_exact` | closed form, Carlson elliptic integrals |
| | `cylinder_lut` | closed form, tabulated elliptic integrals |
| | `cylinder_trapezoidal` | trapezoidal rule on the arc integral |
| Any surface | `points` | quadrature over points |

`plane` and the `cylinder_*` modes each exploit one geometry. `points` is a
numerical method rather than a geometry: it replaces the surface by points
with areas, so it applies to any element shape, including the two above.

## Element parametrization

The closed-form modes take one row of 12 parameters per element, the
`infos_transducers` array:

| Index | Cylindrical element | Flat element |
| ----- | ------------------- | ------------ |
| 0–2 | center $c$ of the cylinder axis (m) | center $c$ of the face (m) |
| 3–5 | axis $e_1$ | unit vector $e_1$ in the face |
| 6–8 | $e_2$, from the axis through the middle of the arc | face normal $e_2$ |
| 9 | radius $R$ (m) | unused |
| 10 | half-aperture angle $\theta_{\max}$ (rad) | half-extent $w$ along $e_3 = e_1 \times e_2$ (m) |
| 11 | half-height $h$ along $e_1$ (m) | half-extent $h$ along $e_1$ (m) |

`translation_rotation_system(..., element="cylinder")` or
`element="plane"` builds these rows for a rotation + translation scan; the
flat face lies at the middle of the arc, $c + R e_2$.
`discretize_cylindrical_transducers()` and
`discretize_planar_transducers()` sample them into points for `points`.

## Flat elements

### `plane`

For a voxel at distance $d$ from the element plane, the points of the face
at distance $r$ from the voxel lie on a circle of radius
$\sqrt{r^2 - d^2}$. The signal in each time step is the area of the face
between two such circles, divided by $r$. The area of a disk inside a
rectangle has a closed form, built from the primitive
$G(a, x) = \tfrac{1}{2}\bigl(x\sqrt{a^2 - x^2} + a^2 \arcsin(x/a)\bigr)$,
so the only discretization is the time axis shared by all modes.

## Cylindrical elements

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

### `cylinder_trapezoidal`

This mode replaces the arc integral by the **trapezoidal rule**: when
$\alpha_l$ and $\beta_l$ are close, the curve $u \mapsto \Psi(u, c\tau_l)$ is
approximated by the chord joining its endpoints,

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

## Any geometry

### `points`

The element surface is replaced by $Q$ points with their areas; each point
contributes a $1/r$-weighted pressure sample, i.e. the surface integral is
replaced by a quadrature rule. This applies to any geometry, including
shapes without a closed form here, such as spherically focused or annular
elements: `locPoints` and `area` can come from any sampling of the surface.
For the two geometries above it converges to the closed-form modes as $Q$
grows, which makes it the reference for validating them. Its error carries
an extra $O(Q^{-1})$ quadrature term, so accuracy is governed by two
parameters ($Q$ and `upsample`) instead of one, and fine discretizations get
expensive.

## Choosing a mode for cylindrical elements

Accuracy is quoted as the reconstruction PSNR relative to
`cylinder_exact`, on the fine noiseless grid of Experiment 1 of the paper;
speed is relative to `cylinder_lut`.

| Mode | Accuracy | Speed | Use case |
| ---- | -------- | ----- | -------- |
| `cylinder_lut` | matches `cylinder_exact` | 1× | **default** for reconstruction |
| `cylinder_exact` | reference | ~10× slower | reference / validation |
| `cylinder_trapezoidal` | −0.3 dB | 1.2–1.5× faster | fastest; when the bias is acceptable |
| `points` | −1.2 dB | ~3× slower | validation; other element shapes |

With 1 % measurement noise these modes become quantitatively comparable;
the differences above are visible mainly in the noiseless fine-grid regime.
