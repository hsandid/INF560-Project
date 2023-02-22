## Report Structure
- Sequential Program Analysis
  - Performance analysis with different input parameters (N Paricles, T time) 

- Parallel approach(es)
  - Data parallelism approach/correctness
  - General implementation details
  - Performance Analysis of parallel programs under most parameters

- Comparing parallel approaches
  - Performance comparison
  - Selecting one approach  

- Selected parallel approach
  - In-depth implementation details
  - Difficulties encountered
  - What could still be enhanced

- Decision Tree
  - Hardware resources availability (CPU cores, GPU)
  - Input Set (Particles)
  - Number of requested MPI ranks and OpenMP threads

- Conclusion
  - Speed-up obtained in sequential vs Parallel
  - Weak/Strong scaling observed in different use-cases 

---

Professor's Note:
-> Remember that everything must be done for both #1 Brute force algorithm and #2 Barnes implementation
-> Decision tree is nice, but do not focus on it for this project
-> Main focus should be performance: (1) How does performance differ IN SEQUENTIAL between brute force and barnes (O(n^2) vs O(nlogn)) ? (2) Do we still have that relationship when codes are made parallel ?
