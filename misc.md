### Utilizing parallel ressources
- General Info: `lstopo`, `lstopo-no-graphics`, `sinfo`
- Batch script for SLURM, executed with `sbatch`:
```bash
#!/bin/bash -l

#SBATCH -n 4

srun hostname
```
### Hybrid approach
- It is very important to think about where you can apply the different programming models. Both paradigms (distributed memory and shared memory) may apply on the same level of parallelism (e.g., splitting an image into pieces in MPI and process each part with OpenMP directives) or may exploit different levels (e.g., split the patterns with MPI and process each pattern with OpenMP to traverse the DNA database).
- To summarize, you have to develop the right cost model that will adapt the parallelism method according to the input parameters (number of MPI ranks, number of OpenMP threads, data input sets...). 
- The main advice is related to the design of cost model for your parallel programming models. Indeed, parallelizing the application with MPI, OpenMP and CUDA may take some time (to code, to debug and to validate). But then it is necessary to choose which model to involve at some parallelism level. This choice will depend on the input parameters set by the end user (input file, number of MPI tasks, number of nodes, maybe number of OpenMP threads...).
