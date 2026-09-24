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

The metrics compare a reconstruction $u$ with the reference $\mathbf{p}_0$
(ground truth, `ref`), both with $N$ voxels.

**SNR.** `SNR()` returns, in dB,

$$
\mathrm{SNR}(u) = 10 \log_{10} \frac{\|\mathbf{p}_0\|_2^2}{\|u - \mathbf{p}_0\|_2^2}.
$$

`SNR_per_plane()`, `SNR_masked()` and `SNR_per_region()` apply the same
formula to each plane along one axis, to the voxels selected by a boolean
mask, and to each of several named regions, for local quality maps.

**PSNR.** `PSNR()` returns, in dB,

$$
\mathrm{PSNR}(u) = 10 \log_{10} \frac{d^2}{\mathrm{MSE}(u)},
\qquad
\mathrm{MSE}(u) = \frac{1}{N}\|u - \mathbf{p}_0\|_2^2,
$$

where the peak value $d$ (`data_range`) defaults to the range of the
reference, $\max_i \mathbf{p}_0[i] - \min_i \mathbf{p}_0[i]$.

**SSIM.** `compute_ssim_3d()` extends the structural similarity index of
[Wang et al. (2004)](https://doi.org/10.1109/TIP.2003.819861) to 3D. It
follows their reference implementation
[`ssim.m`](https://ece.uwaterloo.ca/~z70wang/research/ssim/): Gaussian
window of size 11 and standard deviation 1.5, constants $K_1 = 0.01$ and
$K_2 = 0.03$, and downsampling by a factor
$\max(1, \mathrm{round}(\min(N_x, N_y, N_z)/256))$. The dynamic range
(`data_range`) defaults to 1, which assumes volumes normalized to $[0, 1]$.
See the
[implementation](https://github.com/dotrungthai2001/patminton/blob/main/src/patminton/algorithms.py)
for details.
