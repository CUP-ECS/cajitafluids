# Poisson MPI Benchmark
This directory contains code for a relatively simple finite difference 
fluid advection solver for exploring communication issues on modern architectures
(particularly GPUs). The main goal is to look at different neighbor collective
and GPU communication approaches.

Computationally, the benchmark advects a material feature (that doesn't otherwise effect
fluid flow, e.g. by changing pressite) using the incompressible Euler fluid flow equations. 
Consider, for example, something like a dye being carried through a tank of water or a 
fragrance wafting across a room. 

The main elements of the benchmark are:
  * Solution of the pressure gradient at each timestep to maintain 
    incompressibility. The benchmark has two initial implementations:
    (1) Calling a matrix-free solver in HYPRE to solve the problem or (2)
    running a local matrix-free preconditioned CG solver, in which different
    MPI approaches for the halo exchange are explored.
  * Interpolation (either cubic splines or linear) for semi-Lagrangian 
    advection of the material being advected across timesteps.
  * 3rd-order Runge Kutta for time integration

Sources:
  - Fluid Simulation for Comptuer Graphics by Bridson
  - Incremental Fluids in Kokkos (git@github.com:pkestene/incremental-fluids-kokkos.git)
