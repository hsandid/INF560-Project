/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
**
**/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>
// Adding MPI header file
#include <mpi.h>
// Adding OpenMP header file
#include <omp.h>

#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

#include "ui.h"
#include "nbody.h"
#include "nbody_tools.h"

FILE *f_out = NULL;

int nparticles = 10; /* number of particles */
float T_FINAL = 1.0; /* simulation end time */
particle_t *particles;

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

/* MPI BUFFERS */
double *partsBuffer;
double *partsBufferRecv;


/* MPI Variables */
int N = 1;
int rank = 0;

int *sizeBuffers;
int commonVal;
int endIndex;
int numParticles;
int *displsArray;

/* CUDA variables */
int nbGPU;

extern void GPUComputeForce();

void init()
{
  /* Nothing to do */
}

#ifdef DISPLAY
extern Display *theDisplay; /* These three variables are required to open the */
extern GC theGC;            /* particle plotting window.  They are externally */
extern Window theMain;      /* declared in ui.h but are also required here.   */
#endif

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force_seq(particle_t*p, double x_pos, double y_pos, double mass) {
  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = x_pos - p->x_pos;
  y_sep = y_pos - p->y_pos;
  dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT*(p->mass)*(mass)/dist_sq;

  p->x_force += grav_base*x_sep;
  p->y_force += grav_base*y_sep;
}

/* compute the new position/velocity */
void move_particle_seq(particle_t*p, double step) {

  p->x_pos += (p->x_vel)*step;
  p->y_pos += (p->y_vel)*step;
  double x_acc = p->x_force/p->mass;
  double y_acc = p->y_force/p->mass;
  p->x_vel += x_acc*step;
  p->y_vel += y_acc*step;

  /* compute statistics */
  double cur_acc = (x_acc*x_acc + y_acc*y_acc);
  cur_acc = sqrt(cur_acc);
  double speed_sq = (p->x_vel)*(p->x_vel) + (p->y_vel)*(p->y_vel);
  double cur_speed = sqrt(speed_sq);

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);
}

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force(particle_t *p, double x_pos, double y_pos, double mass)
{
  // Parallelism Notes: No shared variable written to here
  // NVIDIA parallelism..
  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = x_pos - p->x_pos;
  y_sep = y_pos - p->y_pos;
  dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT * (p->mass) * (mass) / dist_sq;

  p->x_force += grav_base * x_sep;
  p->y_force += grav_base * y_sep;
}

/* compute the new position/velocity */
void move_particle(particle_t *p, double step, double* tempAcc, double* tempSpeed)
{
  // Parallelism Notes: Shared variable written to here
  // NVIDIA parallelism..

  p->x_pos += (p->x_vel) * step;
  p->y_pos += (p->y_vel) * step;
  double x_acc = p->x_force / p->mass;
  double y_acc = p->y_force / p->mass;
  p->x_vel += x_acc * step;
  p->y_vel += y_acc * step;

  /* compute statistics */
  double cur_acc = (x_acc * x_acc + y_acc * y_acc);
  cur_acc = sqrt(cur_acc);
  double speed_sq = (p->x_vel) * (p->x_vel) + (p->y_vel) * (p->y_vel);
  double cur_speed = sqrt(speed_sq);
  sum_speed_sq += speed_sq;

  *tempAcc = MAX(*tempAcc, cur_acc);
  *tempSpeed = MAX(*tempSpeed, cur_speed);

  // #pragma omp critical
  // {
    
  //   max_acc = MAX(max_acc, cur_acc);
  //   max_speed = MAX(max_speed, cur_speed);
  // }
  
}

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles(double step)
{
  /* First calculate force for particles. */

  if(nparticles<500)
  {
    /* First calculate force for particles. */
  int i;
  for(i=0; i<nparticles; i++) {
    int j;
    particles[i].x_force = 0;
    particles[i].y_force = 0;
    for(j=0; j<nparticles; j++) {
      particle_t*p = &particles[j];
      /* compute the force of particle j on particle i */
      compute_force_seq(&particles[i], p->x_pos, p->y_pos, p->mass);
    }
  }

  /* then move all particles and return statistics */
  for(i=0; i<nparticles; i++) {
    move_particle_seq(&particles[i], step);
  }
  }
  else
  {

  
  int i;
  // MPI-related variables
  
  // Help normalize data region worked on from [X..Y] to [0..X-Y]

  // Should allocate array for array sizes to be sent (do it once, watch for edge case)
  // Should allocate array for particles handled by each MPI rank (do once, watch for edge case)
  // Allocated particles buffer should be N particles, not (N/npartciles)*N
  // Have special case for last rank, if there are still particles, it should handle processing
  // Make it generalized I would say

  
  // Check there is at least one GPU
  // First: Do loop to compute the particles, should have it parallel to OpenMP
  // so execute it within single thread of OpenMP
  // if(nbGPU)
  // {
  //   //printf("Number of GPUs on node for this MPI Rank node:  %d\n ",nbGPU);
  //   //GPUComputeForce(particles, (nparticles / N) * (rank), (nparticles / N) * (rank)+0.8*numParticles,partsBuffer, 0);
  // }

  
  

#pragma omp parallel default(none) shared(particles, partsBuffer, nparticles, N, rank,commonVal,endIndex, nbGPU, numParticles) 
{
  if(nbGPU>0)
  {
    #pragma omp master
    {
      #pragma omp task
      {
        //GPUComputeForce(particles, (nparticles / N) * (rank), (nparticles / N) * (rank)+(numParticles*8)/10+1,partsBuffer, 0);
        GPUComputeForce(particles, nparticles, (nparticles / N) * (rank), (nparticles / N) * (rank)+(numParticles*8)/10, partsBuffer, commonVal);
        //printf("Hello World\n");
      }
    }
    #pragma omp for schedule(static) private(i) 
    for (i = (nparticles / N) * (rank)+(numParticles*8)/10; i < endIndex; i++)
    {
      
      int j;
      particles[i].x_force = 0;
      particles[i].y_force = 0;
      for (j = 0; j < nparticles; j++)
      {
        particle_t *p = &particles[j];
        /* compute the force of particle j on particle i */
        compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
      }
      partsBuffer[(i-commonVal)*2] = particles[i].x_force;
      partsBuffer[(i-commonVal)*2+1] = particles[i].y_force;
    }
  }
  else
  {
    #pragma omp for schedule(static) private(i) 
    for (i = (nparticles / N) * (rank); i < endIndex; i++)
    {
      
      int j;
      particles[i].x_force = 0;
      particles[i].y_force = 0;
      for (j = 0; j < nparticles; j++)
      {
        particle_t *p = &particles[j];
        /* compute the force of particle j on particle i */
        compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
      }
      partsBuffer[(i-commonVal)*2] = particles[i].x_force;
      partsBuffer[(i-commonVal)*2+1] = particles[i].y_force;
    }
  }
    
    
    
  }

// double* mybuffer;
// if(rank==(N-1))
//   { 
//     mybuffer = (double *)malloc(sizeof(double) * 2 * ((nparticles / N)+(nparticles%N)));
//   }
//   else
//   {
//     mybuffer = (double *)malloc(sizeof(double) * 2 * (nparticles / N));
//   }

//GPUComputeForce(particles, (nparticles / N) * (rank), endIndex,partsBuffer, 0);


// THE LOOP VALUE IS NOT PASSED TO THE KERNEL
// FUUUUUUU
// int h;
// for(h=0;h<numParticles;h++)
// {
//   printf(" %f %f\n",partsBuffer[h],mybuffer[h]);
// }

// free(mybuffer);
//cudaThreadSynchronize();

// printf("\n");
// int z = 0;
// for (z = 0;z<numParticles * 2;z+=2)
// {
//   printf("%f %f ",partsBuffer[z],partsBuffer[z+1]);
// }

// printf("\n");

  // for (i = (nparticles / N) * (rank); i < (nparticles / N) * (rank)+10; i++)
  //   {
  //     partsBuffer[(i-commonVal)*2] = particles[i].x_force;
  //     partsBuffer[(i-commonVal)*2+1] = particles[i].y_force;
  //   }

//GPUComputeForce(particles, (nparticles / N) * (rank), endIndex,partsBuffer, 0);

  // if (nparticles % N != 0)
  // {
  //   for (i = (nparticles / N) * (N); i < nparticles; i++)
  //   {
  //     int j;
  //     particles[i].x_force = 0;
  //     particles[i].y_force = 0;
  //     for (j = 0; j < nparticles; j++)
  //     {
  //       particle_t *p = &particles[j];
  //       /* compute the force of particle j on particle i */
  //       compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
  //     }
  //   }
  // }

  // Might be useful for CUDA
  

  // Here we'd like to use Allgatherv
  // What needs to be changed ?
   MPI_Allgatherv(partsBuffer, numParticles * 2, MPI_DOUBLE, partsBufferRecv, sizeBuffers, displsArray, MPI_DOUBLE, MPI_COMM_WORLD);

  double tempAcc = max_acc;
  double tempSpeed = max_speed;
 #pragma omp parallel default(none) reduction (max:tempAcc,tempSpeed) shared(particles, partsBufferRecv, nparticles, N, step,max_acc,max_speed,sum_speed_sq)
   {
    #pragma omp for private(i) schedule(static) 
    for (i = 0; i < nparticles; i++)
    {
      // if(i==nparticles-1 )
      // {
      //   printf("%f %f\n",partsBufferRecv[i * 2],partsBufferRecv[i * 2+1]);
      // }
      
      particles[i].x_force = partsBufferRecv[i * 2];
      particles[i].y_force = partsBufferRecv[i * 2 + 1];
    }

/* then move all particles and return statistics */

   #pragma omp  for  private(i) schedule(static) 
    for (i = 0; i < nparticles; i++)
    {
      move_particle(&particles[i], step, &tempAcc, &tempSpeed);
    }

    // #pragma omp  for  private(i)  schedule(static) 
    // for (i = 0; i < nparticles; i++)
    // {
    //   double x_acc = particles[i].x_force / particles[i].mass;
    //   double y_acc = particles[i].y_force / particles[i].mass;
    //   double cur_acc = (x_acc * x_acc + y_acc * y_acc);
    //   cur_acc = sqrt(cur_acc);
    //   double speed_sq = (particles[i].x_vel) * (particles[i].x_vel) + (particles[i].y_vel) * (particles[i].y_vel);
    //   double cur_speed = sqrt(speed_sq);
      
    // }
  }

  // Might be useful for CUDA
  // cudaThreadSynchronize();

  max_acc = tempAcc;
  max_speed = tempSpeed;

  // if(tempAcc!=max_acc)
  // {
  //   printf("TempAcc vs Acc\n%f %f\n",tempAcc,max_acc);
  // }
  }
  }

/* display all the particles */
void draw_all_particles()
{
  int i;
  for (i = 0; i < nparticles; i++)
  {
    int x = POS_TO_SCREEN(particles[i].x_pos);
    int y = POS_TO_SCREEN(particles[i].y_pos);
    draw_point(x, y);
  }
}

void print_all_particles(FILE *f)
{
  int i;
  for (i = 0; i < nparticles; i++)
  {
    particle_t *p = &particles[i];
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }
}

void run_simulation()
{

  
  
  
  nbGPU = 0;
  cudaGetDeviceCount(&nbGPU);
  cudaSetDevice (rank % nbGPU);
  
  // if(rank==0)
  // {
  //   GPUComputeForce(particles, 0, nparticles);
  // }

  commonVal = (nparticles / N) * (rank);
  endIndex = 0;
  numParticles = 0;

  if(rank==(N-1))
  { 
    partsBuffer = (double *)malloc(sizeof(double) * 2 * ((nparticles / N)+(nparticles%N)));
    endIndex = (nparticles / N) * (rank + 1) + nparticles%N;
    numParticles = (nparticles / N) + nparticles%N;
    //printf("Rank %d Buffer size %d\n",rank,2 * ((nparticles / N)+(nparticles%N)));
  }
  else
  {
    partsBuffer = (double *)malloc(sizeof(double) * 2 * (nparticles / N));
    endIndex = (nparticles / N) * (rank + 1);
    numParticles = (nparticles / N);
    //printf("Rank %d Buffer size %d\n",rank,2 * ((nparticles / N)));
  }
  
  
  partsBufferRecv = (double *)malloc(sizeof(double) * 2 * nparticles);

  //printf("Rank %d Big Buffer size %d\n",rank,2 * nparticles);

  sizeBuffers = malloc(sizeof(int)*N);
  displsArray = malloc(sizeof(int)*N);

  int k;
  for (k=0; k<N; k++)
  {
    // For last rank only
    if(k==N-1)
    {
      sizeBuffers[k] = ((nparticles / N)+ nparticles%N)*2;
    }
    else
    {
      sizeBuffers[k] = (nparticles / N)*2;
    }
    
    displsArray[k] = ((nparticles / N)*k)*2;
  }

  //    for (k=0; k<N; k++)
  // {
  //    printf("Rank %d sizebuf %d, displsArr %d\n", rank, sizeBuffers[k], displsArray[k]);
    
  // }

   //printf("Rank %d Numparticles %d endIndex %d\n", rank,numParticles, endIndex);
 

  double t = 0.0, dt = 0.01;
  while (t < T_FINAL && nparticles > 0)
  {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */
    all_move_particles(dt);

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */

    dt = 0.1 * max_speed / max_acc;

    /* Plot the movement of the particle */
#if DISPLAY
    clear_display();
    draw_all_particles();
    flush_display();
#endif
  }
  /* Free MPI particle buffers */
  free(partsBuffer);
  free(partsBufferRecv);
  free(sizeBuffers);
  free(displsArray);
}

/*
  Simulate the movement of nparticles particles.
*/
int main(int argc, char **argv)
{
  if (argc >= 2)
  {
    nparticles = atoi(argv[1]);
  }
  if (argc == 3)
  {
    T_FINAL = atof(argv[2]);
  }

  init();

  /* Allocate global shared arrays for the particles data set. */
  particles = malloc(sizeof(particle_t) * nparticles);
  all_init_particles(nparticles, particles);

  /* Initialize thread data structures */
#ifdef DISPLAY
  /* Open an X window to display the particles */
  simple_init(100, 100, DISPLAY_SIZE, DISPLAY_SIZE);
#endif

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  // Put MPI initialization and finalization
  // in-between main simulation function
  MPI_Init(&argc, &argv);

  /* Setting up MPI variables */
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Main thread starts simulation ... */
  run_simulation();

  gettimeofday(&t2, NULL);

  if (rank == 0)
  {
    double duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

#ifdef DUMP_RESULT
    FILE *f_out = fopen("particles.log", "w");
    assert(f_out);
    print_all_particles(f_out);
    fclose(f_out);
#endif

    printf("-----------------------------\n");
    printf("nparticles: %d\n", nparticles);
    printf("T_FINAL: %f\n", T_FINAL);
    printf("-----------------------------\n");
    printf("Simulation took %lf s to complete\n", duration);
    printf("-----------------------------\n");
  }

  MPI_Finalize();

#ifdef DISPLAY
  clear_display();
  draw_all_particles();
  flush_display();

  printf("Hit return to close the window.");

  getchar();
  /* Close the X window used to display the particles */
  XCloseDisplay(theDisplay);
#endif
  return 0;
}
