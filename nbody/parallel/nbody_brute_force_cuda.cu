#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
//#include <cuda_runtime.h>

// includes
//#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
//#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))  /* utility function */

__global__ void vecAddThreeKernel(char *a, char *b, char *c, char *res, int N)
{
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i<N)
    {
        res[i] = a[i]+b[i] + c[i];
    }
}

__global__ void kernComputeForces(double*cuda_xpos_array, double*cuda_ypos_array, double*cuda_mass_array, double*cuda_xforce,double*cuda_yforce,int size)
{
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    //int j;
    //j = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (i<size)
    {
    int j;
    for (j=0;j<size;j++)
    {
        double x_sep, y_sep, dist_sq, grav_base;

        printf("LOOP on INDEX %d PASS %d\n",i,j);

        x_sep = cuda_xpos_array[j] - cuda_xpos_array[i];
        y_sep =  cuda_ypos_array[j] - cuda_ypos_array[i];
        dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

        /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
        grav_base =  0.01 * (cuda_mass_array[i]) * (cuda_mass_array[j]) / dist_sq;

        cuda_xforce[i] += grav_base * x_sep;
        cuda_yforce[i] += grav_base * y_sep;
    }

    //printf("X block id %d block dim %d threadID %d index %d val %f %f %f %f %f\n",blockIdx.x ,blockDim.x, threadIdx.x, i, cuda_xpos_array[i], cuda_ypos_array[i],cuda_mass_array[i], cuda_xforce[i], cuda_yforce[i]);
    // if (i<size)
    // {
    //     cuda_xforce[i] += cuda_xpos_array[i]+cuda_ypos_array[i] + cuda_mass_array[i];
    //     cuda_yforce[i] += cuda_xpos_array[i]+cuda_ypos_array[i] + cuda_mass_array[i];
    // }
    }
    
}

// Take a step back and compartementalize
// 1: Take pointer reference to array of particles, number of particles to treat
// start index and end index
// 2: Allocate arrays to store double of xpos, ypos, mass, xforce, yforce  on CPU side
// 3: Allocate 5 arrays on CUDA side to store double of xpos, ypos, mass; and xforce and yforce
// 4: OP

//5: Copy xforce and yforce results back into CPU memory
//6: Iterate to update particles forces as needed
//7: Free all arrays on NVIDIA side
//8: Return

// int startIndex, int endIndex, particle_t *particleArray
extern "C" void GPUComputeForce()
{
    // Save x coordinates int
    double *xpos_array   =(double *) malloc(sizeof(double)*(5));
    double * ypos_array = (double *)malloc(sizeof(double)*(5));
    double * mass_array = (double *)malloc(sizeof(double)*(5));
    double *xforce =  (double *)malloc(sizeof(double)*(5));
    double *yforce = (double *) malloc(sizeof(double)*(5));

    int i;
    for (i=0;i<5;i++)
    {
        xpos_array[i] = i;
        ypos_array[i] = i;
        mass_array[i] = i;
        xforce[i] = 0.0;
        yforce[i] = 0.0;
    }

    // Save x coordinates int
    double *cuda_xpos_array  ;
    double * cuda_ypos_array;
    double * cuda_mass_array;
    double *cuda_xforce ;
    double *cuda_yforce ;

    

    cudaMalloc((void **)&cuda_xpos_array, 5*sizeof(double));
    cudaMalloc((void **)&cuda_ypos_array, 5*sizeof(double));
    cudaMalloc((void **)&cuda_mass_array,5*sizeof(double));
    cudaMalloc((void **)&cuda_xforce, 5*sizeof(double));
    cudaMalloc((void **)&cuda_yforce, 5*sizeof(double));

    cudaMemcpy(cuda_xpos_array, xpos_array, 5*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_ypos_array, ypos_array, 5*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_mass_array, mass_array, 5*sizeof(double), cudaMemcpyHostToDevice);

    kernComputeForces<<<2 , 5 >>>(cuda_xpos_array, cuda_ypos_array, cuda_mass_array, cuda_xforce, cuda_yforce, 5);

    cudaMemcpy(xforce, cuda_xforce, 5*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(yforce, cuda_yforce, 5*sizeof(double), cudaMemcpyDeviceToHost);

    //Create loop to pass into particles array

    // test loop
    // for (i=0;i<50;i++)
    // {
    //     printf("%f %f \n",xforce[i], yforce[i]);
    // }
    // Save x coordinates int
    cudaFree(cuda_xpos_array);
    cudaFree( cuda_ypos_array);
    cudaFree( cuda_mass_array);
    cudaFree(cuda_xforce);
    cudaFree(cuda_yforce);
}

/* Function computing the final string to print */
void compute_string( char * res, char * a, char * b, char *c, int length ) 
{





char * d_a ;
char * d_b ;
char * d_c ;
char * d_res;

cudaMalloc((void **)&d_a, length*sizeof(char));
cudaMalloc((void **)&d_b, length*sizeof(char));
cudaMalloc((void **)&d_c, length*sizeof(char));
cudaMalloc((void **)&d_res, length*sizeof(char));

cudaMemcpy(d_a, a, length*sizeof(char), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, length*sizeof(char), cudaMemcpyHostToDevice);
cudaMemcpy(d_c, c, length*sizeof(char), cudaMemcpyHostToDevice);
//cudaMemcpy(d_res, res, length*sizeof(char), cudaMemcpyHostToDevice);

// Launch kernel
vecAddThreeKernel<<< 4, 20 >>>(d_a, d_b, d_c, d_res, length);
//vecAddThreeKernel(d_a, d_b, d_c, d_res, length);

cudaMemcpy(res, d_res, length*sizeof(char), cudaMemcpyDeviceToHost);

cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
cudaFree(d_res);

}
