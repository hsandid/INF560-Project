#!/bin/bash

source set_env.sh
#lstopo-no-graphics
#lstopo
#sinfo
cd ../nbody/parallel
make all

#./nbody_brute_force 1000 2
#./nbody_barnes_hut 1000 2
# Sbatch for MPI only ?
# Think about srun/salloc/sbatch with mpirun ?
salloc -N 1 -n 1 mpirun ./nbody_brute_force 1000 2
#salloc -N 1 -n 1 mpirun ./nbody_barnes_hut 1000 2
