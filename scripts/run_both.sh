#!/bin/bash

mydir=${PWD}

for stepCount in {1..10};
do
    source run_parallel.sh
    cd $mydir
    source run_sequential.sh
    cd $mydir

    echo -e "=========================="
    if cmp  "../nbody/parallel/particles.log" "../nbody/sequential/particles.log"; then
    echo -e "BRUTE-FORCE VALID"
    else
    echo -e "BRUTE-FORCE INVALID"
    fi
    echo -e "=========================="
done
#do 
#    for particlesCount in 10 50 100 250 500 1000 #5000 10000
#    do 
#        ./nbody_brute_force ${particlesCount} ${stepCount}
#        ./nbody_barnes_hut ${particlesCount} ${stepCount}
#    done
#done


