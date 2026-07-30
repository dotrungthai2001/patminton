# Transducer models

## Element parametrization

Each cylindrical element is described by 12 parameters, one row of the
`infos_transducers` array:

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
`PAT`.

### `points`

The transducer surface is discretized into point sensors (built with
`discretize_cylindrical_transducers()`);
each point contributes a $1/r$-weighted pressure sample. Simple and exact
in the limit, but expensive for fine discretizations.

### `cylinder_exact`

The surface integral is evaluated analytically at each time step with
**Carlson symmetric elliptic integrals** computed on the GPU. Most accurate;
slower than the LUT variant because the Carlson iteration runs per thread at
every time step.

### `cylinder_lut`

Same formula as `cylinder_exact`, but the incomplete elliptic integrals
$E(\varphi, k)$ and $F(\varphi, k)$ are replaced by **bilinear
interpolation in a precomputed lookup table** (LUT). The LUT axes use a
non-uniform grid clustered near the singularity
$(\varphi, k) \to (\pi/2, 1)$:

$$
\sin\varphi = (1-\varepsilon)\bigl(1 - (1-x)^4\bigr), \qquad
k = (1-\varepsilon)\bigl(1 - (1-y)^4\bigr), \qquad x, y \in [0, 1].
$$

The LUT is built in Python with `scipy.special.ellipeinc` / `ellipkinc` and
transferred to the GPU at initialization. Roughly 10× faster than
`cylinder_exact` with sub-percent accuracy loss. **Recommended default.**

### `cylinder_far_field`

The elliptic integrals are replaced by a **planar approximation** treating
the element as a flat rectangle at each time step. Valid in the far field
($R \gg \lambda$); fastest cylindrical mode, less accurate near the
transducer or for small radii.

### `cylinder_arcs`

The cylinder is decomposed into a stack of **horizontal arcs** at fixed
heights, each integrated independently along the angle. Useful for
validation against the direct elliptic approach.

### `cylinder_planes`

The cylinder is decomposed into a stack of **planar slices** perpendicular
to the axis, each integrated independently. Complementary to
`cylinder_arcs`.

## Choosing a mode

| Mode | Accuracy | Speed | Use case |
| ---- | -------- | ----- | -------- |
| `cylinder_lut` | high | fast | default for reconstruction |
| `cylinder_exact` | highest | slow | reference / validation |
| `cylinder_far_field` | moderate | fastest | large $R$, quick tests |
| `points` | tunable | depends on sampling | arbitrary geometries |
| `cylinder_arcs` / `cylinder_planes` | high | moderate | validation studies |
