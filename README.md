# Poisson MPI Benchmark
This directory contains code for a relatively simple finite difference 
fluid solver for exploring communication issues on modern architectures
(particularly GPUs). The main goal is to look at different neighbor collective
and GPU communication approaches.

Computationally, the main elements of the benchmark are:
  * Solution of the pressure gradient at each timestep to maintain 
    incompressibility. The benchmark has two initial implementations:
    (1) Calling a matrix-free solver in HYPRE to solve the problem or (2)
    running a local matrix-free preconditioned CG solver, in which different
    MPI approaches for the halo exchange are explored.
  * Either forward euler or 3rd-order Runge Kutta for time integration
  * Interpolation (either cubic splines or linear) for semi-Lagrangian 
    advection across timesteps.
