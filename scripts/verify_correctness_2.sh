#!/bin/bash

echo -e "=========================="
if cmp  "../nbody/parallel/particles.log" "../nbody/sequential/particles.log"; then
  echo -e "BRUTE-FORCE VALID"
else
  echo -e "BRUTE-FORCE INVALID"
fi
echo -e "=========================="


