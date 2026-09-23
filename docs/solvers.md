# Reconstruction algorithms

All solvers minimize a regularized least-squares objective using only the
forward and adjoint products of a `PAT` operator, and
share the return signature `(u, F_list, SNR_list, SSIM_list, elapsed,
SNR_planes_history)`. Passing the ground truth via `ref` enables SNR/SSIM
tracking per iteration and SNR-based early stopping (`patience`).

| Function | Method | Constraint |
| -------- | ------ | ---------- |
| `least_squares_CG()` | CGLS (conjugate gradients on the normal equations) | none |
| `least_squares_LBFGSB()` | L-BFGS-B (quasi-Newton, scipy) | $u \ge 0$ |
| `least_squares_PGD()` | Projected gradient descent | $u \ge 0$ |
| `least_squares_CP_TV()` | Chambolle-Pock primal-dual with 3D TV | $u \ge 0$ |

!!! warning "Operator scaling"

    In raw physical units $\|A^\top A\| \sim 10^{-15}$, so a regularization
    weight like $\lambda = 10^{-4}$ makes the Tikhonov term dominate the data
    term by ~11 orders of magnitude (CGLS then "converges" in one iteration to
    a heavily damped image). Rescale first with `normalize_operator()`, which
    wraps the operator as $A/\|A\|$ and rescales the data accordingly — the
    minimizer is unchanged and $\lambda$, step sizes and tolerances live on
    the usual scale.

## CGLS

Minimizes $\tfrac{1}{2}\|Au - s\|^2 + \tfrac{\lambda}{2}\|u\|^2$ by
conjugate gradients on the normal equations
$(A^\top A + \lambda I)u = A^\top s$. Each iteration costs one forward and
one adjoint pass. No non-negativity constraint is enforced on the iterate
(the clamp applied for metric reporting does not affect the CG direction).
Supports an optional diagonal preconditioner `M_inv`.

## L-BFGS-B

Minimizes the same objective with $u \ge 0$ using scipy's `L-BFGS-B`. The
gradient $\nabla f = A^\top(Au - s) + \lambda u$ is computed on the GPU
and passed to the CPU optimizer. Supports **checkpoint/resume** (`ckpt_dir`,
`x0`, `iter_offset`, …) for long runs split across HPC walltime windows.

## Projected gradient descent

Gradient step followed by projection onto $u \ge 0$:

$$
u \leftarrow \max\Bigl(0,\; u - \tfrac{1}{L}\bigl[A^\top(Au - s) + \lambda u\bigr]\Bigr)
$$

The Lipschitz constant $L = \|A^\top A + \lambda I\|$ is estimated by
power iteration
(`estimate_lipschitz()`, 30 steps)
before the first iteration.

## Chambolle-Pock with total variation

Minimizes

$$
\min_{u \ge 0} \; \tfrac{1}{2}\|Au - s\|^2 + \lambda\,\|\nabla u\|_{2,1}
$$

with the primal-dual Chambolle-Pock algorithm, using an explicit scaling
`beta` on the TV block. `lam` is required (no sensible default). `beta`
changes the step sizes, not the minimizer; it defaults to
$\|A\|/\|\nabla\| = \sqrt{\|A^\top A\|/12}$, which balances the two blocks
($\|\nabla\|^2 \le 12$ for 3D forward differences). The operator norms are
estimated by power iteration at startup.

## Metrics

- `SNR()` — signal-to-noise ratio in dB, plus
  per-plane (`SNR_per_plane()`), masked
  (`SNR_masked()`) and per-region
  (`SNR_per_region()`) variants for local quality maps.
- `PSNR()` — peak signal-to-noise ratio in dB, with the range of the
  reference as peak value.
- `compute_ssim_3d()` — 3D SSIM with a
  Gaussian window.
