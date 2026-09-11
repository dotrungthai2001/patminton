# Key parameters

| Parameter | Where | Description |
| --------- | ----- | ----------- |
| `upsample` | `PAT.__init__` | Time-axis oversampling factor for the convolution. Larger values give a more accurate convolution with the system kernel $h(t)$ at the cost of GPU memory and longer FFTs. Typical values: 11–51. |
| `steps` | `PAT.__init__` | Coarse time-step stride in the smooth regions of the integration loop. Larger is faster but less accurate. Default: `upsample`. |
| `steps_border` | `PAT.__init__` | Fine time-step stride near the integration boundaries (the arc bounds $\alpha_l$, $\beta_l$ and the support edges). Default: `upsample`; 1 gives maximum accuracy. |
| `blockSize` | `PAT.__init__` | CUDA thread block edge size (3D blocks of `blockSize`³ threads). Tune for GPU occupancy. |
| `laser_pulse_variance` | `PAT.__init__` | Standard deviation (s) of the Gaussian envelope modeling the laser pulse — despite the name, the value is used as $\sigma$, not $\sigma^2$ (`exp(-t²/2σ²)`). Default `5e-9` is a 5 ns pulse. 0 disables it. |
| `eir` | `PAT.__init__` | 1D array of the Electronic Impulse Response, sampled at `dt`, with the center of the EIR at index 0 (zero-phase convention). |
| `Nphi`, `Nk` | `PAT.__init__` | LUT resolution for `cylinder_lut` (default 1000×1000), along the $\sin\varphi$ and parameter $\nu$ axes respectively. Higher improves accuracy at the cost of GPU memory; at 1000×1000 the interpolation error is already below single precision. |
| `lam` | solvers | Tikhonov (or TV) regularization weight. |
| `patience` | solvers | Iterations without SNR improvement before early stopping (requires `ref`). |
| `M_inv` | solvers | Optional diagonal preconditioner applied to the normal equations. |
