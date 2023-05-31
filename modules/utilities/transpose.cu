#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>  // CUBLAS_GEAM

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/transpose.h"

                  
void transpose_data (int nbPoints, int nbDims,
                     T_real *GPU_data,
                     T_real *GPU_dataT)
{
    T_real alpha = 1.0f;
    T_real beta = 0.0f;

    // Transpose GPU_data to GPU_dataT
    CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(handleCUBLAS,                
                         CUBLAS_OP_T, CUBLAS_OP_N,
                         nbPoints, nbDims,
                         &alpha, GPU_data, nbDims,
                         &beta, NULL, nbPoints,
                         GPU_dataT, nbPoints));
}