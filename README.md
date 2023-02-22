# INF560-Project
- Selected Project: Particle Interaction
- Main target: Data parallelism -> Separate data between processes in MPI, OpenMP, NVIDIA CUDA
- Main decomposition: START ALWAYS WITH MPI, then OpenMP, then GPU
---

### Discussion from Wednesday 18th Jan.
- Description of Project Files
  - `nbody.h`: holds info for the particle structure, and the node structure in the barnes-hut algorithm.
  - `nbody_alloc.c/.h`: memory allocator routines (init, allocate, free) used in the nbody algorithms.
  - `nbody_barnes_hut.c`: nbody simulation that implements the barnes-Hut algorithm (O(nlog(n))
  - `nbody_brute_force.c`: nbody simulation using the brute-force algorithm (O(nxn))
  - `nbody_tools.c/.h`: general helper functons for nbody simulation, as well as node manipulation functions for barnes-hut
  - `particles.log`: file containing a set of particles to simulate
  - `ui.c/.h` and `xstuff.c/.h`: X11-based display implementation.
- Parallelization approach (Brute Force)
  - MPI: No root process / Have multiple processes, with each process computing all forces applied to a set of particles / At each step, data will be broadcasted between all processes so that next step processing might begin.
- Parallelization approach (Barnes Hut)
  - MPI: No root process / Have multiple processes, with each process generating a tree-like structure associated with the Barnes Hut algorithm at the first step / At each following step, each process will calculate the force applied to its designated set of particles, update the tree structure accordingly with its calculated particles forces, then broadcast its tree and particles changes to other processes, and simultaneously receive changes from other processes on associated particles and update its tree.


BEST CUDA APPROACH: 1 MPI RANK PER COMPUTE NODE, USE STREAMS WITH OPENMP TO AVOID SERIALIZATION
ONLY WORTH IT TO USE CUDA PARALLELIZATION FOR A LARGE ENOUGH DATA SIZE
COULD SPLIT PART OF THE WORK BETWEEN CPU (OpenMP) AND GPU (CUDA)
