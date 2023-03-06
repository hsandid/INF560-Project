#!/bin/bash

source set_env.sh
#lstopo-no-graphics
#lstopo
#sinfo
cd ../nbody/sequential
make clean
make all
./nbody_brute_force 500 2
#./nbody_barnes_hut 1000 2

#for stepCount in 2 
#do 
#    for particlesCount in 10 50 100 250 500 1000 #5000 10000
#    do 
#        ./nbody_brute_force ${particlesCount} ${stepCount}
#        ./nbody_barnes_hut ${particlesCount} ${stepCount}
#    done
#done
