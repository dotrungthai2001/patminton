# Key parameters

| Parameter | Where | Description |
| --------- | ----- | ----------- |
| `upsample` | `PAT.__init__` | Time-axis oversampling factor for the convolution. Larger values give a more accurate convolution with $g(t)$ at the cost of GPU memory and longer FFTs. Typical values: 11–51. |
| `steps` | `PAT.__init__` | Coarse time-step stride in the smooth regions of the integration loop. Larger is faster but less accurate. Default: `upsample`. |
| `steps_border` | `PAT.__init__` | Fine time-step stride near the integration boundaries (singularities at $\alpha$, $\beta$ and the support edges). Default: `upsample`; 1 gives maximum accuracy. |
| `blockSize` | `PAT.__init__` | CUDA thread block edge size (3D blocks of `blockSize`³ threads). Tune for GPU occupancy. |
| `laser_pulse_variance` | `PAT.__init__` | Variance (s²) of the Gaussian envelope modeling the laser pulse. 0 disables it. |
| `eir` | `PAT.__init__` | 1D array of the Electronic Impulse Response, sampled at `dt`, with the center of the EIR at index 0 (zero-phase convention). |
| `Nphi`, `Nk` | `PAT.__init__` | LUT resolution for `cylinder_lut` (default 1000×1000). Higher improves accuracy at the cost of GPU memory. |
| `lam` | solvers | Tikhonov (or TV) regularization weight. |
| `patience` | solvers | Iterations without SNR improvement before early stopping (requires `ref`). |
| `M_inv` | solvers | Optional diagonal preconditioner applied to the normal equations. |
