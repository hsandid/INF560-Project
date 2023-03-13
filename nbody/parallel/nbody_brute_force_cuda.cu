#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
//#include <cuda_runtime.h>

// includes
//#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
//#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

__global__ void vecAddThreeKernel(char *a, char *b, char *c, char *res, int N)
{
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N)
    {
        res[i] = a[i]+b[i] + c[i];
    }
}

extern "C" void helloThere()
{
    printf("What!\n");
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
