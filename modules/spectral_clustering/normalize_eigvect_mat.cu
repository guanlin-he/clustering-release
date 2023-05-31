#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/spectral_clustering/normalize_eigvect_mat.h"


__global__ void kernel_normalize_eigenvector_matrix (int nbPoints, int nbClusters,  // input
                                                     T_real *GPU_eigVects)          // input & output
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nbPoints) {
        T_real accu = 0.0f;
        for (int j = 0; j < nbClusters; j++) {
            index_t eigvectIdx = ((index_t)j)*((index_t)nbPoints) + ((index_t)tid);
            accu += GPU_eigVects[eigvectIdx] * GPU_eigVects[eigvectIdx];
        }
        for (int j = 0; j < nbClusters; j++) {
            index_t eigvectIdx = ((index_t)j)*((index_t)nbPoints) + ((index_t)tid);
            GPU_eigVects[eigvectIdx] *= RSQRT(accu);
        }
    }
}


void normalize_eigenvector_matrix (int nbPoints, int nbClusters,  // input
                                   T_real *GPU_eigVects)          // input & output
{
    // Declaration
    dim3 Dg, Db;
    
    // Normalize each row of the eigenvector matrix to unit length
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    kernel_normalize_eigenvector_matrix<<<Dg,Db>>>(nbPoints, nbClusters,  // input
                                                   GPU_eigVects);         // input & output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
}