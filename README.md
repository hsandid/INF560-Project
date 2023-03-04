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
AVOID STREAMS!
- Main parallelism source: Processing of each particle.

### Algorithms
#### (1) Brute-Force
- The naive algorithm (in nbody_brute_force.c) computes, for each time step, the gravitational force applied to each particle, and computes its new position.
- The force applied to particle p1 is computed by summing the force that all the other particles apply to p1.
- The complexity of the force computation is thus O(n^2) for each time step. 

```c
for(i=0; i < nparticles; i++) {
    compute_force(p1, p[i]->x_pos, p[i]->y_pos, p[i]->mass);
}
```

- The complexity of the force computation is thus O(n^2) for each time step.
- Once the forces are computed, the new position of each particle can be computed:

```c
for(i=0; i < nparticles; i++) {
    move_particle(&particles[i], step);
}
```

#### (2) Barnes-Hut
![image](https://www.enseignement.polytechnique.fr/profs/informatique/Patrick.Carribault/INF560/TD/projects/decoupage_espace.png)
![image2](https://www.enseignement.polytechnique.fr/profs/informatique/Patrick.Carribault/INF560/TD/projects/quad_tree.png)
- Algorithm with a lower complexity. Particles are stored in a tree based on their position and groups of particles can be viewed as one big particle in order to reduce the computation cost (Reference: https://www.enseignement.polytechnique.fr/profs/informatique/Patrick.Carribault/INF560/TD/projects/barnes_86.pdf).
-  In order to reduce the complexity of the simulation, the Barnes-Hut algorithm (see nbody_barnes_hut.c) uses approximations during the phase that computes the forces applied to a particle.
- The main idea of this algorithm is to group particles that are close to each other into one big "virtual" particle. When computing the forces applied to a particle p1, if a group of particles is far enough from p1, the cumulated force of the particles is approximated by the force of the virtual particle whose position is at the barycenter of the particles and whose mass is the sum of the particles mass.
- To do that, the space (in this case, a 2 dimension space) is recursively split in 4, forming a quad-tree. Each node of the quad-tree corresponds to a region of the 2D space. Each node is split recursively into 4 sub node until the nodes contain at most one particle. The result is a tree where the leafs contain particles. Here is an example of a tree containing 8 particles: 
-  When creating the tree, the barycentre of each node is computed. Thus, the computation of the forces applied to a particle p1 is done recursively starting from the root of the tree. For each node n, the following algorithm is applied:
    - If n is a leaf, the force of n's particle (if any) on p1 is computed
    - Else, the ratio between the size of the node (size) and the distance between the barycentre and p1 (distance) is computed.
        - If size/distance < θ, the force of a virtual particle against p1 is computed. The virtual particle mass is the sum of the mass of the particles in the node, and its position is the barycentre of these particles.
        - Otherwise, the algorithm is repeated recursively for the 4 sub nodes of n

- The value of θ is chosen arbitrarily and defines the approximation of the algorithm. For this project, we assume that θ = 2.
- Once the computation of the force applied to the particles is complete, the new position of the particles is computed, and a new tree corresponding to the new position is created. 

### Setup
- `set_env.sh` file is present on root, and used to initialize the required environment variables on polytechnique lab computers.
    - Command: `source 'set_env.sh'`
- Makefile setup: Dump file activated + no display to avoid issues when programming remotely
- `srun`, `mpirun`,

### Correctness
- Brute Force: Can have 1:1 correctness, check if dumpfile generated is the same
- Barnes-Hut: Can have a factor of correctness, check if results generated by dumpfile has degree of accuracy (EDIT: If we are not going to split tree, we must expect a 1:1 degree of correctness).

### Performance Study
- Need script to generate graph of performance in sequential vs original parallel vs MPI vs OpenMP vs NVIDIA CUDA; maybe loop over different possible variables
- Report: Need to keep in mind speed-up, and notion of weak/strong scaling
- TODO: Use the command `gnuplot plot.gp > plot.png` with the `plot.gp` file in `/scripts` to generate a plot of performance for performance run
- Do we see the same performance relationship in parallel, as we have seen in sequential  (O(n^2) vs O(nlogn))?

### Step 1 - Pure MPI
#### Hints
- Particle interaction
    - Brute force implementation
        - Replicating the particles on all the MPI ranks is probably the easiest solution for parallelizing this implementation
        - The number of steps depends on the speed of the particles.
    - Barnes-Hut implementation
        - This implementation may be difficult to parallelize using MPI
        - You should replicate all the particles on all the MPI ranks
        - Balancing the processing load across the MPI ranks is the difficult part
#### Approach
- Modify makefile to support build with MPI
- Add SLURM batch script

### Step 2 - Pure OpenMP
- Brute force implementation
    - You can parallelize two loops: the computation of forces, and the loop that moves particles
- Barnes Hut implementation
    - You can parallelize the loop that computes the forces.
    - Since the algorithm is recursive, using OpenMP tasks or nested parallelism is probaby the most efficient
    - The loop that moves particles cannot be parallelized (at least not easily)

### Steps to Expand..
- Pure CUDA
- Mixing OpenMP, MPI and CUDA
- How will users interact with the application ? 
    - Modular Input: Particles number, steps number
    - Modular user setup: CPU cores, GPU availability
    - Modular ranks/threads: Number of MPI ranks and OpenMP threads
    - Which parallelization approach to take for each use case (or not) ?
- Debug print in code
