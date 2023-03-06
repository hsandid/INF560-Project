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
void compute_force(particle_t *p, double x_pos, double y_pos, double mass)
{
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
void move_particle(particle_t *p, double step)
{

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
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);
}


/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles(double step)
{
  /* First calculate force for particles. */

  int i;
  int k = 0;
  // START MPI
  int N;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  
  //struct Partstruct *partsBuffer = malloc(sizeof(struct Partstruct)*(nparticles/N));
  //struct Partstruct *partsBufferRecv = malloc(sizeof(struct Partstruct)*(nparticles));
  //particle_force partSend[nparticles];
  // END MPI

  for (i = (nparticles/N)*(rank); i < (nparticles/N)*(rank+1); i++) // Need to divide evenly between MPI ranks
  //for (i = 0; i < nparticles; i++)
  {
    int j;
    particles[i].x_force = 0; // Set force x and y applied on particles[i] to zero
    particles[i].y_force = 0; //
    for (j = 0; j < nparticles; j++) // Sum up all forces applied on particles[i]
    {
      particle_t *p = &particles[j];
      /* compute the force of particle j on particle i */
      compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
    }
    partsBuffer[k] = particles[i].x_force;
    partsBuffer[k+1] = particles[i].y_force;
    k+=2;
  }

  
  // for (i = (nparticles/N)*(rank); i < (nparticles/N)*(rank+1); i++)
  // {
  //   partsBuffer[k] = particles[i].x_force;
  //   partsBuffer[k+1] = particles[i].y_force;
  //   if(rank==3)
  //   {
  //     //printf("%f %f VS %f %f\n",particles[i].x_force,particles[i].y_force,partsBuffer[k],partsBuffer[k+1]);
  //   }
    
    
  // }

 // printf("K VALUE: %d\n", k);
  // for(k=nparticles/N-5;k<nparticles/N;k+=2)
  // {
  //  printf("R: %d, X: %f, Y: %f\n",rank, partsBuffer[k],partsBuffer[k+1]);
  // }
  

  if(nparticles%N!=0)
  {
    //printf("Range Imperfect: [%d,%d[ untouched\n", (nparticles/N)*(N) ,nparticles);
     for (i = (nparticles/N)*(N); i < nparticles; i++) // Need to divide evenly between MPI ranks
    {
      int j;
      particles[i].x_force = 0; // Set force x and y applied on particles[i] to zero
      particles[i].y_force = 0; //
      for (j = 0; j < nparticles; j++) // Sum up all forces applied on particles[i]
      {
        particle_t *p = &particles[j];
        /* compute the force of particle j on particle i */
        compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
      }
    }
  }

  // MPI allgather should be here, before the particles moves
  // This choice is done due to the data structure required to move data
  // Before move_particle: We just need to compute the force
  // After move_particle: We need force, velocity, node data structure..
  
  

  //printf("[]: %d to %d, with N: %d, and NDiv: %d\n",(nparticles/N)*(rank),(nparticles/N)*(rank+1)-1, 2*(nparticles/N), 2*((nparticles/N)*N));
  MPI_Allgather(partsBuffer, (nparticles/N)*2, MPI_DOUBLE, partsBufferRecv, (nparticles/N)*2, MPI_DOUBLE, MPI_COMM_WORLD);

  // /* then move all particles and return statistics */
  // for (i = 0; i < nparticles; i++)
  // {
  //   move_particle(&particles[i], step);
  // }

  // for (i = 0; i < (nparticles/N)*2; i++)
  // {
  //   //particles[i].x_force = partsBufferRecv[i*2];
  //   //particles[i].y_force = partsBufferRecv[i*2+1];
  //   //printf("%f \n",partsBuffer[i]);
  // }

  // for (i = 0; i < ((nparticles*N)/N)*2; i++)
  // {
  //   //particles[i].x_force = partsBufferRecv[i*2];
  //   //particles[i].y_force = partsBufferRecv[i*2+1];
  //   //if(rank==3)
  //   //{
  //    //printf("%f \n",partsBufferRecv[i]);
  //   //}
  //   //printf("%f \n",partsBufferRecv[i]);
  // }

  for (i = 0; i < (nparticles/N)*N; i++)
  {
    particles[i].x_force = partsBufferRecv[i*2];
    particles[i].y_force = partsBufferRecv[i*2+1];

    // if(particles[i].x_force == partsBufferRecv[i*2] && particles[i].y_force == partsBufferRecv[i*2+1])
    // {
    //   //printf("T\n");
    // }
    // else
    // {
    //   printf("F\n");
    // }
    
    //printf("%f %f VS %f %f\n",particles[i].x_force,particles[i].y_force,partsBufferRecv[i*2],partsBufferRecv[i*2+1]);
  }

  /* then move all particles and return statistics */
  for (i = 0; i < nparticles; i++)
  {
    //particles[i].x_force = partsBufferRecv[i*2];
    //particles[i].y_force = partsBufferRecv[i*2+1];
    move_particle(&particles[i], step);
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
  
  // DEBUG


  // printf("Rank: %d/%d\n", rank, N);
  // printf("Particles amount: %d\n", nparticles);

  /*
  int minParticlesPerRank = 50;
  
  if(nparticles<minParticlesPerRank)
  {
    printf("Less than %d particles overall!\n",minParticlesPerRank);
    printf("Using only 1 rank...\n")
  }
  else
  {
    if(nparticles<(N*minParticlesPerRank))
    {
      printf("Cannot assign at least %d particles per rank!\n",minParticlesPerRank);
      printf("Reducing number of ranks used to %d\n",nparticles/50)
      N = 
    }
  }
  */

  // printf("Range: [%d,%d]\n", (nparticles/N)*(rank),(nparticles/N)*(rank+1)-1);

  // if(nparticles%N!=0)
  // {
  //   printf("Range Imperfect: [%d,%d[ untouched\n", (nparticles/N)*(N) ,nparticles);
  // }
  // else
  // {
  //   printf("Range is a perfect Modulo!\n");
  // }

  
  
  
  
  //DEBUG
  // Creating particle MPI datastructure
  // double *partsBuffer = malloc(sizeof(double)*2*(nparticles/N));
  // double *partsBufferRecv = malloc(sizeof(double)*2*((nparticles/N)*N));
  int N;
  //int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  partsBuffer = (double *) malloc(sizeof(double)*2*(nparticles/N));
  partsBufferRecv = (double *) malloc(sizeof(double)*2*((nparticles/N)*N));

  double t = 0.0, dt = 0.01;
  while (t < T_FINAL && nparticles > 0)
  {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */
    all_move_particles(dt);

    //printf("R: %d, T: %f, DT: %f\n",rank,t,dt);
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
  /* Main thread starts simulation ... */
  run_simulation();
  

  gettimeofday(&t2, NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0)
  {
    double duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

  /* Are we sure that the result dump will occur in the main directory with rank zero only ? */
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
