# Key parameters

| Parameter | Where | Description |
| --------- | ----- | ----------- |
| `tStart`, `nT`, `dt` | `PAT.__init__` | Time window $[t_\text{start}, t_\text{start} + n_T\,\Delta t)$. It must contain every arrival time from the grid to the element surfaces: a surface patch whose arrival lies entirely outside the window is skipped, and the out-of-window part of a patch that straddles an edge is added to the edge sample. The convolution with $g(t)$ is circular, so signal at the end of the window also wraps onto its first samples. |
| `upsample` | `PAT.__init__` | Time-axis oversampling factor for the convolution. Larger values give a more accurate convolution with the system kernel $g(t)$ at the cost of GPU memory and longer FFTs. Typical values: 11–51. |
| `steps` | `PAT.__init__` | Coarse time-step stride in the smooth regions of the integration loop. Larger is faster but less accurate. Default: `upsample`. For a point source at the center of the test geometry (`upsample=5`), the default gives a 4 % relative error against a fine point quadrature of the element surfaces; `steps = steps_border = 1` gives 0.2 %. |
| `steps_border` | `PAT.__init__` | Fine time-step stride near the integration boundaries (the arc bounds $\alpha_l$, $\beta_l$ and the support edges). Default: `upsample`; 1 gives maximum accuracy. |
| `blockSize` | `PAT.__init__` | CUDA thread block edge size (3D blocks of `blockSize`³ threads). Tune for GPU occupancy. At most 8 for the `cylinder_*` modes, whose kernels are compiled for 512-thread blocks. |
| `laser_pulse_variance` | `PAT.__init__` | Standard deviation (s) of the Gaussian envelope modeling the laser pulse — despite the name, the value is used as $\sigma$, not $\sigma^2$ (`exp(-t²/2σ²)`). Default `5e-9` is a 5 ns pulse. 0 disables it. |
| `eir` | `PAT.__init__` | 1D array of the Electronic Impulse Response, sampled at `dt`, with the center of the EIR at index 0 (zero-phase convention). |
| `Nphi`, `Nk` | `PAT.__init__` | LUT resolution for `cylinder_lut` (default 1000×1000), along the $\sin\varphi$ and parameter $\nu$ axes respectively. Higher improves accuracy at the cost of GPU memory; at 1000×1000 the interpolation error is already below single precision. |
| `lam` | solvers | Tikhonov (or TV) regularization weight. |
| `patience` | solvers | Iterations without SNR improvement before early stopping (requires `ref`). |
| `M_inv` | solvers | Optional diagonal preconditioner applied to the normal equations. |
