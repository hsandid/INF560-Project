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

---
- One node is not enough
  - Group of multiple nodes (**Cluster**)
  - Whole machine --> Distributed Memory System
  - Increase in number of nodes leads to stress on Network interface
- Parallel Programming Model (or paradigms)
  - Requires independent tasks, which can be executed in any order and/or simultaneously
    - Focus: Data parallelism
    - Repeat same actions on similar data
  - Distributed Memory: **MPI**
  - Shared memory: **OpenMP**
  - Heterogeneous/manycore parallelism: **CUDA**

- Shared Memory System
  - Compute units share memory
  - A **node** is the largest set of units sharing memory
  - Watch out for critical parts
  - Best used in a multi-threaded process, as threads have access to the same memory zone.
  - OpenMP approach: Compiler directives in C.
- Distributed Memory System
  - Each compute unit has its own memory space, with units not able to access each other's memories
  - Parallel tasks work on their own memory space, with data split among parallel tasks to enable parallel execution.
  - MPI Approach: Use MPI library and directives
- Mixed Systems
  - Mix of shared and distributed memory systems
  - Cluster of nodes within a network

---

- MPI Focus (DSM)
  - Header file, MPI_Init + MPI_Finalize
  - Need to handle the compilation process (Makefile modification + execution modification)
  - Communicator of all processes: MPI_Comm MPI_COMM_WORLD
  - Get number of processes in communicator: `int MPI_Comm_size( MPI_Comm comm, int *size);`
  - Get rank of MPI process within communicator: `int MPI_Comm_rank(MPI_Comm comm, int *rank);`
    - Rank must be used to determine data to work on + role of either master or slave.
  - Number of MPI processes may be different from number of available cores/processors + Execution of processes is not related to their rank
  - Do we care about point-to-point communications (One-to-One)?
    - For data, No! 
    - For control & end, maybe!
    - **PROJECT:** We will not have a master/slave model.
  - Do we care about collectives ?
    - For data, Yes!
    - **PROJECT:** We want to use All-to-All data passing.
  - Rules: All processes in the group identified by the communicator must call the collective routine, with the sequence of collective communication being respected.
  - Collectives (or global) synchronization:
    - Synchronize all processes belonging to target communicator: `int MPI_Barrier( MPI_Comm comm ) ;`
  - Collectives communication (non all-to-all): broadcast (1-all);  Scatter (1-to-all); Gather (All-to-1); Reduction (All-to-1; possibility to create custom reduction operation); 
  - Collectives communication (all-to-all): Allgather (gather + broadcast; all-to-all); Reduce-to-All (all-to-all);
  - Collective communications with different sizes (v-versions): gather - allgather - scatterv - alltoallv --> Not recommended as not performance efficient.
  - Non-blocking collectives: No! We MUST receive data from all other threads, then compute based on this data, then MUST share data computer to all, then proceed to next time step.
  - Domain decomposition for data parallelism exploitation: Split data into pieces, size and shape depend on input data set, computation, and number of MPI ranks. 
  - **PROJECT:** Every rank needs to share data with all other ranks (particles). Every rank will have copy of data, then computes part of data it is responsible of for that step, then communicates its computation to all other processes. Root will output final result at the end.

---

- OpenMP Focus (SMS)
  - Main goal: Enable parallelism for a whole node
  - API/Directives to augment existing code
  - Execution inside one process, which creates and activates threads based on fork/join model; with each thread having an implicit task and a specific rank
  - Compiler must be aware of OpenMP, with flags such as `-fopenmp` being used, and proper flags to function APIs. Need to modify makefile!
  - OpenMP runtime can be controlled with (1) directives and function calls inside program, (2) and environment variables outside program 
  - Number of threads can be specified outside program with `OMP_NUM_THREADS`, or inside program with `num_threads(int)`, although the number of threads is not guaranteed. 
  - C header file: `#include <omp.h>`
  - Declaring parallel region: `#pragma omp parallel`
  - Check if currently in a parallel region: `omp_in_parallel();`
  - Get thread number: `int omp_get_thread_num()`
  - Get number of threads: `int omp_get_num_threads() ;`
  - Implicit synchronization barrier at the end of parallel region.
  - By default, data is shared inside a parallel region; with variables declared before a parallel region being shared by all threads (equivalent to `shared(a,b)`). Default behavior can be changed with `default` clause. **Useful:** `default(none)` can be used to force declaration of the scope of each variable used from the master thread. Be careful of concurrent read-writes. `private(a)` can be used to have private data but with it initialized to nothing, and `firstprivate(a, b)` can be used to have existing data with private variables. The value inside these private variables will not be reflected into the previous existing variable in the master thread (for loops, the value will be before the loop itself). The `lastprivate(a,b)` clause can be used to have the value after the last iteration of the loop.
  - Local variables are private even inside other functions.
  - Global/Static variables (in C, those declared outside function blocks + static qualifier) are shared by every thread in the same process
  - Dynamic memory allocation is allowed inside parallel region, with pointer value able to be transmitted to other threads for concurrent access.
  - OpenMP worksharing directives include `for` in loops, or `sections` for blocks of instructions.
  - `for` directive can have added `schedule` clause for scheduling and have implicit synchronization barriers. Schedule can be `static`, which splits iteration domain into chunk with equal size when possible, or `dynamic` were chunks are distributed on-demands to threads. `guided` can also be used for dynamic scheduling, with the added benefit that it will vary chunk size. Chunk size selection is important. The `OMP_SCHEDULE` environment variable can also be used, as well as the `omp_set_schedule()` function call.
  - In case a reduction operation is used on a variable, the `reduction(<op>:a)` clause can be used. 
  - Clauses can be combined: `#pragma omp parallel for`
  - `collapse(int)` clause might be interesting to use over nested loops. However, watch out if the number of iterations on the nested loop is not fixed or depends on the first loop's index.
  - In case a barrier is needed (?), we can use the `#pragma omp barrier` directive
  - In case atomic operations are needed (?), we can use the `#pragma omp atomic` directive.
  - In case critical regions are needed (?), we can use the `#pragma omp critical` directive.
  - In case we only want one thread to execute part of a target block, we can either use the `#pragma omp master` directive (no synchronization) or the  `#pragma omp single` (implicit synchronization) directive.
  - Ignored locks and mutex usage for critical sections.
  - OpenMP Tasks: `#pragma omp task`. Scheduling can be forced using `if(int)` directive. Do not mix use workshare constructs within tasks! Task nesting is possible (but not our focus ?). Task binding might be a bit too much to tackle. We need to keep in mind the scope of variables accessed within a task, and if they are shared (i.e. have the same address across all threads). For variables declared inside task body, they are `private`. For variables declared outside task body, they can be `shared` (pre-determined or global variables) or otherwise `firstprivate`. Barrier directives can be used between tasks to ensure a set of tasks is done before moving on. To wait for the completion of tasks created by the current thread, the directive `taskwait` can be used. `taskgroup` can be used to wait for a set of tasks including nested tasks. Locks and atomic operation are also possible, but not considered here. It is possible to use tasks to parallelize loops (and not use workshare directives) but it might make it harder. Task dependencies might be too much.
  - Dealing with while loops is difficult in OpenMP, but tasks can be used to process it (see slides for example)
  - For performance profiling, we can use `double omp_get_wtime(void);` for time in seconds; or `double omp_get_wtick(void);` for counter accuracy.
  - Avoid having many disjoint parallel regions, as waking up threads causes overhead.

---

- Hybrid approach between OpenMP/MPI
  - Need to do some runtime stacking.
  - MPI advantages: Can exploit whole cluster - work on SMS and DMS.
  - MPI disadvantages: difficult data sharing and load balancing
  - OpenMP advantages: Incremental code changes - Data sharing - Nothing to duplicate - Easy load balancing within a node
  - OpenMP disadvantages: Not useable on multiple compute nodes.
  - Check slides for hybrid Hello World example and compilation options
  - Domain decomposition: In MPI, each rank owns a domain. In OpenMP, each thread owns a subset of data (use for construct, static scheduling and/or dynamic clauses on workshare loop). In MPI+OpenMP, first divide for MPI task, then divide for OpenMP task.
  - Taxonomy model to shape performance of hybrid model: two main parameters shaping performance are granularity and placement. (**IMPORTANT** Look at the slides for the Hybrid taxonomy). Either Fine Grain (loop level parallelism) or Coarse Grain (Single OpenMP parallel region, with critical regions inside synchronization constructs), with OpenMP first. Placement might take different approaches to deal with concurrency for resources.
  - Goal: design of OpenMP runtime fully integrated into MPI runtime dealing with Granularity and Placement
  - Granularity: OpenMP Fine Grain requirements: Optimization of launching/stopping a parallel region, and Optimization of performing loop scheduling vs OpenMP Coarse grain requirements:  Optimization of synchronization constructs (barrier, single, nowait…)
  - Placement: ?
  - For OpenMP, design without busy waiting if possible.

---

- CUDA
  - GPU/Accelerator seen as slave device (CPU is host, accelerator is slave)
  - Check number of available devices: `__host____device__cudaError_t cudaGetDeviceCount (int *count)`
  - Select device for CUDA operations (multi gpu ?): `cudaSetDevice(int)`
  - Requires Host programming and compute kernels, with data being transferred back and forth.
  - Programming process: (1) Initialization, (2) GPU memory allocation, (3) Host to device transfer, (4) Remote execution of compute kernel, (5) Device to host transfer.
  - Allocate memory space on device with `cudaMalloc`
  - Transfer data with device with `cudeMemcpy`
  - CUDA thread hierarchy: Blocks and grids to organize thread hierarchy. Grids contain blocks, which contain threads. Much choose a dimension of grid and blocks in grid when launching a CUDA kernel from host device. (`my_kernel<<<Dimgrid, Dimblock>>>(arg1, arg2, arg3 )`)
  - Each thread has a local memory, each block has a private shared memory, and each grid has a global memory.
  - Barrier synchronization with `cudaThreadSynchronize();` or `CUDA_LAUNCH_BLOCKING=1`
  - Performance handling, error handling, using nvprof and CUDA events `cudaEvent_t ` and errors `cudaError_t`(see slide).
  - CUDA code is contained in `.cu` file and contains CUDA and/or host code, might be separated from `.c` files containing only host code.
  - Possible to have different variable attributes: `_device_` is for global memory and by default (shared by all threads),  and there is also `__constant__` and `__shared__` which have limited application.
  - In case of concurrent accesses, there might be some synchronization done (?). Nothing done about variable restrictions (?)
  - Special data types are available: vectors, multi-dimension integers
  - Some pre-defined variables are accessible for CUDA:  Grid dimensions`dim3 gridDim`, Index of block in grid `dim3 blockIdx`, Block dimensions `dim3 blockDim`, Index of thread in block`uint3 threadIdx`, Warp size`int warpSize`
  - Memory barrier (?), Syncrhonization (?), Math operations (?), Atomic operations (?), Timing and I/O (?), register allocation and limiting number of threads and blocks (?)
  - Best practices for CUDA: Minimize data transfers between host and device, Access main device memory w/ coalesced operations
  - Multi-gpu stuff, context shifts, driver API and runtime API.

---

- MPI+OpenMP+CUDA
  - MPI+CUDA example (slides)
  - OpenMP+CUDA (slides)
  - MPI+OpenMP+CUDA: Best case for GPU is 1 MPI process per available GPU, and streams for threads (Not the required approach ?). Divide load between GPU and CPU-based ranks. GPU rank should have higher load.  All threads in same MPI process share same address space on target GPU, so there is a possibility to allocate and transfer memory before parallel region
    and launched kernels from OpenMP threads

---

TASKSLIST:

1. Modify makefiles for MPI, openMP, CUDA support. Ensure MPI has thread support.
2. Work being done is always the same or not ? Should we have static scheduling or dynamic scheduling in MPI/OpenMP ?
3. OpenMP tasks and Tree tasks in Barnes Hut (Check fibonacci example in the slides for cutoff).
