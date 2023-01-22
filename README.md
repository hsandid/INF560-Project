# INF560-Project
- Selected Project: Particle Interaction

---

### Discussion from Wednesday 18th Jan.
- Description of Project Files
  - `nbody.h`
  - `nbody_alloc.c/.h`
  - `nbody_barnes_hut.c`
  - `nbody_brute_force.c`
  - `nbody_tools.c/.h`
  - `particles.log`
  - `ui.c/.h`
  - `xstuff.c/.h`
- Parallelization approach (Brute Force)
  - MPI: No root process / Have multiple processes, with each process computing all forces applied to a set of particles / At each step, data will be broadcasted between all processes so that next step processing might begin.
- Parallelization approach (Barnes Hut)
  - MPI: No root process / Have multiple processes, with each process generating a tree-like structure associated with the Barnes Hut algorithm at the first step / At each following step, each process will calculate the force applied to its designated set of particles, update the tree structure accordingly with its calculated particles forces, then broadcast its tree and particles changes to other processes, and simultaneously receive changes from other processes on associated particles and update its tree.
