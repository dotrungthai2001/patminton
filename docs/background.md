# Background

These pages describe the model implemented by `patminton` and how to use it:

- [Physical and mathematical model](theory.md) — the forward operator, its
  adjoint, and the elliptic-integral formulation for cylindrical elements.
- [Transducer models](transducer-models.md) — the six forward/adjoint
  implementations (point quadrature, exact elliptic, lookup table,
  trapezoidal, piecewise plane, arcs) and their accuracy/speed trade-offs.
- [Reconstruction algorithms](solvers.md) — CGLS, L-BFGS-B, PGD and
  Chambolle-Pock TV, with the operator-scaling caveat.
- [Key parameters](parameters.md) — accuracy/speed/memory knobs.
