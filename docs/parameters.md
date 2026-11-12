# Key parameters

| Parameter | Where | Description |
| --------- | ----- | ----------- |
| `tStart`, `nT`, `dt` | `PAT.__init__` | Time window $[t_\text{start}, t_\text{start} + n_T\,\Delta t)$. It must contain every arrival time from the grid to the element surfaces: the out-of-window part of a patch that straddles an edge is added to the edge sample. The convolution with $g(t)$ is circular, so signal at the end of the window also wraps onto its first samples. |
| `upsample` | `PAT.__init__` | Time-axis oversampling factor for the convolution. Larger values give a more accurate convolution with the system kernel $g(t)$ at the cost of GPU memory and longer FFTs. Typical values: 11–51. |
| `steps` | `PAT.__init__` | Coarse time-step stride in the smooth regions of the integration loop. Larger is faster but less accurate. Default: `upsample`. For a point source at the center of the test geometry (`upsample=5`), the default gives a 3.3 % relative error against a point quadrature of the element surfaces (151 × 7 points per element, as in `test_models_match_point_quadrature`); `steps = steps_border = 1` gives 0.37 %. In the `cylinder_*` modes, the last time step of each surface patch is its own block, and `steps` can also be a 3D integer array: a map of steps at the nodes `steps_coords`, read in the frame of each element (see `PAT.set_steps_map`). |
| `steps_border` | `PAT.__init__` | Fine time-step stride near the integration boundaries (the arc bounds $\alpha_l$, $\beta_l$ and the support edges). Default: `upsample`, or the step of the voxel when `steps` is a map; 1 gives maximum accuracy. |
| `steps_coords` | `PAT.__init__` | `(x, y, z)` strictly increasing node coordinates (m) of a steps map, in the element frame: x along `e2`, y along `e1 × e2`, z along the cylinder axis `e1`, origin at the cylinder center. Each voxel takes the step of the nearest node, for every element; positions outside the map take the step of its border. |
| `blockSize` | `PAT.__init__` | CUDA thread block edge size (3D blocks of `blockSize`³ threads). Tune for GPU occupancy. |
| `laser_pulse_variance` | `PAT.__init__` | Standard deviation (s) of the Gaussian envelope modeling the laser pulse — despite the name, the value is used as $\sigma$, not $\sigma^2$ (`exp(-t²/2σ²)`). Default `5e-9` is a 5 ns pulse. 0 disables it. |
| `eir` | `PAT.__init__` | 1D array of the Electronic Impulse Response, sampled at `dt`, with the center of the EIR at index 0 (zero-phase convention). |
| `Nphi`, `Nk` | `PAT.__init__` | LUT resolution for `cylinder_lut` (default 1000×1000), along the $\sin\varphi$ and parameter $\nu$ axes respectively. Higher improves accuracy at the cost of GPU memory; at 1000×1000 the interpolation error is already below single precision. |
| `lam` | solvers | Tikhonov (or TV) regularization weight. |
| `patience` | solvers | Iterations without SNR improvement before early stopping (requires `ref`). |
| `M_inv` | solvers | Optional diagonal preconditioner applied to the normal equations. |
