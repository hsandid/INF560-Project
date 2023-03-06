#!/bin/bash

# Sourcing
source set_env.sh

cd ../nbody

# Clean + Make sequential
cd sequential
make clean > /dev/null
make all > /dev/null
cd ..

# Clean + Make parallel
cd parallel
make clean > /dev/null
make all > /dev/null
cd ..

# Run brute-force comparison
cd sequential
./nbody_brute_force 1300 2 > /dev/null
cd ..

cd parallel
salloc -N 1 -n 4 mpirun ./nbody_brute_force 1300 2 > /dev/null
cd ..

sleep 10

echo -e "=========================="
if cmp  "parallel/particles.log" "sequential/particles.log"; then
  echo -e "BRUTE-FORCE VALID"
else
  echo -e "BRUTE-FORCE INVALID"
fi
echo -e "=========================="

# Run barnes-hut comparison
cd sequential
./nbody_barnes_hut 1000 2 > /dev/null
cd ..

cd parallel
salloc -N 1 -n 1 mpirun ./nbody_barnes_hut 1000 2 > /dev/null
cd ..

#sleep 7

echo -e "=========================="
if cmp  "parallel/particles.log" "sequential/particles.log"; then
  echo -e "BARNES-HUT VALID"
else
  echo -e "BARNES-HUT INVALID"
fi
echo -e "==========================\n"




# add later for barnes hut
#./nbody_barnes_hut 1000 2


