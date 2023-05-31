#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>     // CUBLAS_GEAM
#include <float.h>         // Library Macros (e.g. FLT_MAX, FLT_MIN)

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/feature_scaling_gpu.h"


// https://stackoverflow.com/a/51549250/1714410
// https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda/51549250#51549250
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}


__device__ __forceinline__ float atomicMinFloat (float * addr, float value) 
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}


__global__ void kernel_initialize_max_min (int nbDims,
                                           float *GPU_dimMax, float *GPU_dimMin)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nbDims) {
        GPU_dimMax[tid] = -FLT_MAX; //Alternative: -INFINITY;
        GPU_dimMin[tid] = FLT_MAX;  //Alternative: INFINITY;
    }
}


template <int BLOCK_SIZE_X>
__global__ void kernel_find_max_min (int nbPoints, int nbDims, T_real *GPU_dataT,
                                     float *GPU_dimMax, float *GPU_dimMin)
{
    // 2D block, 2D grid
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float *shMax = (float*)shBuff;                   // blockDim.y*blockDim.x  floats  in shared memory
    float *shMin = &shMax[blockDim.y * blockDim.x];  // blockDim.y*blockDim.x  floats  in shared memory

    if (col < nbPoints && row < nbDims) {
        index_t dataTIdx = ((index_t)row)*((index_t)nbPoints) + ((index_t)col);
        shMax[threadIdx.y*blockDim.x + threadIdx.x] = GPU_dataT[dataTIdx];
        shMin[threadIdx.y*blockDim.x + threadIdx.x] = shMax[threadIdx.y*blockDim.x + threadIdx.x];
    } else {
        shMax[threadIdx.y*blockDim.x + threadIdx.x] = -FLT_MAX; //Alternative: -INFINITY;
        shMin[threadIdx.y*blockDim.x + threadIdx.x] = FLT_MAX;  //Alternative: INFINITY;
    }
    
    if (BLOCK_SIZE_X > 512) {
        __syncthreads();
        if (threadIdx.x < 512) {
            shMax[threadIdx.y*blockDim.x + threadIdx.x] = (shMax[threadIdx.y*blockDim.x + threadIdx.x] > shMax[threadIdx.y*blockDim.x + threadIdx.x + 512]) ? shMax[threadIdx.y*blockDim.x + threadIdx.x] : shMax[threadIdx.y*blockDim.x + threadIdx.x + 512];
            shMin[threadIdx.y*blockDim.x + threadIdx.x] = (shMin[threadIdx.y*blockDim.x + threadIdx.x] < shMin[threadIdx.y*blockDim.x + threadIdx.x + 512]) ? shMin[threadIdx.y*blockDim.x + threadIdx.x] : shMin[threadIdx.y*blockDim.x + threadIdx.x + 512];
        } else
            return;
    }

    if (BLOCK_SIZE_X > 256) {
        __syncthreads();
        if (threadIdx.x < 256) {
            shMax[threadIdx.y*blockDim.x + threadIdx.x] = (shMax[threadIdx.y*blockDim.x + threadIdx.x] > shMax[threadIdx.y*blockDim.x + threadIdx.x + 256]) ? shMax[threadIdx.y*blockDim.x + threadIdx.x] : shMax[threadIdx.y*blockDim.x + threadIdx.x + 256];
            shMin[threadIdx.y*blockDim.x + threadIdx.x] = (shMin[threadIdx.y*blockDim.x + threadIdx.x] < shMin[threadIdx.y*blockDim.x + threadIdx.x + 256]) ? shMin[threadIdx.y*blockDim.x + threadIdx.x] : shMin[threadIdx.y*blockDim.x + threadIdx.x + 256];
        } else
            return;
    }

    if (BLOCK_SIZE_X > 128) {
        __syncthreads();
        if (threadIdx.x < 128) {
            shMax[threadIdx.y*blockDim.x + threadIdx.x] = (shMax[threadIdx.y*blockDim.x + threadIdx.x] > shMax[threadIdx.y*blockDim.x + threadIdx.x + 128]) ? shMax[threadIdx.y*blockDim.x + threadIdx.x] : shMax[threadIdx.y*blockDim.x + threadIdx.x + 128];
            shMin[threadIdx.y*blockDim.x + threadIdx.x] = (shMin[threadIdx.y*blockDim.x + threadIdx.x] < shMin[threadIdx.y*blockDim.x + threadIdx.x + 128]) ? shMin[threadIdx.y*blockDim.x + threadIdx.x] : shMin[threadIdx.y*blockDim.x + threadIdx.x + 128];
        } else
            return;
    }

    if (BLOCK_SIZE_X > 64) {
        __syncthreads();
        if (threadIdx.x < 64) {
            shMax[threadIdx.y*blockDim.x + threadIdx.x] = (shMax[threadIdx.y*blockDim.x + threadIdx.x] > shMax[threadIdx.y*blockDim.x + threadIdx.x + 64]) ? shMax[threadIdx.y*blockDim.x + threadIdx.x] : shMax[threadIdx.y*blockDim.x + threadIdx.x + 64];
            shMin[threadIdx.y*blockDim.x + threadIdx.x] = (shMin[threadIdx.y*blockDim.x + threadIdx.x] < shMin[threadIdx.y*blockDim.x + threadIdx.x + 64]) ? shMin[threadIdx.y*blockDim.x + threadIdx.x] : shMin[threadIdx.y*blockDim.x + threadIdx.x + 64];
        } else
            return;
    }

    if (BLOCK_SIZE_X > 32) {
        __syncthreads();
        if (threadIdx.x < 32) {
            shMax[threadIdx.y*blockDim.x + threadIdx.x] = (shMax[threadIdx.y*blockDim.x + threadIdx.x] > shMax[threadIdx.y*blockDim.x + threadIdx.x + 32]) ? shMax[threadIdx.y*blockDim.x + threadIdx.x] : shMax[threadIdx.y*blockDim.x + threadIdx.x + 32];
            shMin[threadIdx.y*blockDim.x + threadIdx.x] = (shMin[threadIdx.y*blockDim.x + threadIdx.x] < shMin[threadIdx.y*blockDim.x + threadIdx.x + 32]) ? shMin[threadIdx.y*blockDim.x + threadIdx.x] : shMin[threadIdx.y*blockDim.x + threadIdx.x + 32];
        } else
            return;
    }

    if (BLOCK_SIZE_X > 16) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 16) {
            shMax[threadIdx.y*blockDim.x + threadIdx.x] = (shMax[threadIdx.y*blockDim.x + threadIdx.x] > shMax[threadIdx.y*blockDim.x + threadIdx.x + 16]) ? shMax[threadIdx.y*blockDim.x + threadIdx.x] : shMax[threadIdx.y*blockDim.x + threadIdx.x + 16];
            shMin[threadIdx.y*blockDim.x + threadIdx.x] = (shMin[threadIdx.y*blockDim.x + threadIdx.x] < shMin[threadIdx.y*blockDim.x + threadIdx.x + 16]) ? shMin[threadIdx.y*blockDim.x + threadIdx.x] : shMin[threadIdx.y*blockDim.x + threadIdx.x + 16];
        }
    }

    if (BLOCK_SIZE_X > 8) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 8) {
            shMax[threadIdx.y*blockDim.x + threadIdx.x] = (shMax[threadIdx.y*blockDim.x + threadIdx.x] > shMax[threadIdx.y*blockDim.x + threadIdx.x + 8]) ? shMax[threadIdx.y*blockDim.x + threadIdx.x] : shMax[threadIdx.y*blockDim.x + threadIdx.x + 8];
            shMin[threadIdx.y*blockDim.x + threadIdx.x] = (shMin[threadIdx.y*blockDim.x + threadIdx.x] < shMin[threadIdx.y*blockDim.x + threadIdx.x + 8]) ? shMin[threadIdx.y*blockDim.x + threadIdx.x] : shMin[threadIdx.y*blockDim.x + threadIdx.x + 8];
        }
    }

    if (BLOCK_SIZE_X > 4) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 4) {
            shMax[threadIdx.y*blockDim.x + threadIdx.x] = (shMax[threadIdx.y*blockDim.x + threadIdx.x] > shMax[threadIdx.y*blockDim.x + threadIdx.x + 4]) ? shMax[threadIdx.y*blockDim.x + threadIdx.x] : shMax[threadIdx.y*blockDim.x + threadIdx.x + 4];
            shMin[threadIdx.y*blockDim.x + threadIdx.x] = (shMin[threadIdx.y*blockDim.x + threadIdx.x] < shMin[threadIdx.y*blockDim.x + threadIdx.x + 4]) ? shMin[threadIdx.y*blockDim.x + threadIdx.x] : shMin[threadIdx.y*blockDim.x + threadIdx.x + 4];
        } 
    }

    if (BLOCK_SIZE_X > 2) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 2) {
            shMax[threadIdx.y*blockDim.x + threadIdx.x] = (shMax[threadIdx.y*blockDim.x + threadIdx.x] > shMax[threadIdx.y*blockDim.x + threadIdx.x + 2]) ? shMax[threadIdx.y*blockDim.x + threadIdx.x] : shMax[threadIdx.y*blockDim.x + threadIdx.x + 2];
            shMin[threadIdx.y*blockDim.x + threadIdx.x] = (shMin[threadIdx.y*blockDim.x + threadIdx.x] < shMin[threadIdx.y*blockDim.x + threadIdx.x + 2]) ? shMin[threadIdx.y*blockDim.x + threadIdx.x] : shMin[threadIdx.y*blockDim.x + threadIdx.x + 2];
        }
    }

    if (BLOCK_SIZE_X > 1) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 1) {
            shMax[threadIdx.y*blockDim.x + threadIdx.x] = (shMax[threadIdx.y*blockDim.x + threadIdx.x] > shMax[threadIdx.y*blockDim.x + threadIdx.x + 1]) ? shMax[threadIdx.y*blockDim.x + threadIdx.x] : shMax[threadIdx.y*blockDim.x + threadIdx.x + 1];
            shMin[threadIdx.y*blockDim.x + threadIdx.x] = (shMin[threadIdx.y*blockDim.x + threadIdx.x] < shMin[threadIdx.y*blockDim.x + threadIdx.x + 1]) ? shMin[threadIdx.y*blockDim.x + threadIdx.x] : shMin[threadIdx.y*blockDim.x + threadIdx.x + 1];
        }
    }

    if (threadIdx.x == 0 && row < nbDims) {
        atomicMaxFloat(&GPU_dimMax[row], shMax[threadIdx.y*blockDim.x]);
        atomicMinFloat(&GPU_dimMin[row], shMin[threadIdx.y*blockDim.x]);
    }
}


inline void template_kernel_find_max_min (dim3 Dg, dim3 Db, size_t shMemSize,
                                          int nbPoints, int nbDims, T_real *GPU_dataT,
                                          float *GPU_dimMax, float *GPU_dimMin)
{
    switch(Db.x) {
        case 1024: kernel_find_max_min<1024><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        case 512:  kernel_find_max_min< 512><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        case 256:  kernel_find_max_min< 256><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        case 128:  kernel_find_max_min< 128><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        case 64:   kernel_find_max_min<  64><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        case 32:   kernel_find_max_min<  32><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        case 16:   kernel_find_max_min<  16><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        case 8:    kernel_find_max_min<   8><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        case 4:    kernel_find_max_min<   4><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        case 2:    kernel_find_max_min<   2><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        case 1:    kernel_find_max_min<   1><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT, GPU_dimMax, GPU_dimMin); break;
        default:   fprintf(stderr, "Unsupported value for Db.x of kernel_find_max_min kernel!\n"); exit(EXIT_FAILURE);
    }
}


__global__ void kernel_min_max_scaling (int nbPoints, int nbDims,
                                        float *GPU_dimMax, float *GPU_dimMin,
                                        T_real *GPU_dataT)
{
    // 2D block, 2D grid
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < nbPoints && row < nbDims) {
        T_real max = GPU_dimMax[row];
        T_real min = GPU_dimMin[row];
        T_real width = max - min;
        if (width != 0.0f) {
            index_t dataTIdx = ((index_t)row)*((index_t)nbPoints) + ((index_t)col);
            GPU_dataT[dataTIdx] = (GPU_dataT[dataTIdx] - min) / width;
        }
    }
}


__global__ void kernel_inverse_min_max_scaling (int nbPoints, int nbDims,
                                                float *GPU_dimMax, float *GPU_dimMin,
                                                T_real *GPU_dataT)
{
    // 2D block, 2D grid
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < nbPoints && row < nbDims) {
        T_real max = GPU_dimMax[row];
        T_real min = GPU_dimMin[row];
        T_real width = max - min;
        if (width != 0.0f) {
            index_t dataTIdx = ((index_t)row)*((index_t)nbPoints) + ((index_t)col);
            GPU_dataT[dataTIdx] = GPU_dataT[dataTIdx] * width + min;
        }
    }
}


void find_min_max (int nbDims, int nbPoints,
                   T_real *GPU_dataT,
                   float *GPU_dimMax, float *GPU_dimMin)
{
    // Declaration
    dim3 Dg, Db;
    size_t shMemSize;
    
    // Initialize the maximum and minimum values of each dimension
    Db.x = BsXD;
    Db.y = 1;
    Dg.x = nbDims/Db.x + (nbDims%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    kernel_initialize_max_min<<<Dg,Db>>>(nbDims, GPU_dimMax, GPU_dimMin);
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    
    // Find the maximum and minimum values of each dimension
    Db.x = BsXN;
    Db.y = BSYD;
    if (BsXN*BSYD > 1024) {
        printf("<-bsxn>*BSYD should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = nbDims/Db.y + (nbDims%Db.y > 0 ? 1 : 0);
    shMemSize = (sizeof(float)*Db.y)*Db.x +   // float shMax[blockDim.y * blockDim.x]
             (sizeof(float)*Db.y)*Db.x;       // float shMin[blockDim.y * blockDim.x]
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_find_max_min kernel needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    template_kernel_find_max_min(Dg, Db, shMemSize,
                                 nbPoints, nbDims, GPU_dataT, 
                                 GPU_dimMax, GPU_dimMin);
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
}


// Min-max scaling: rescale the numeric features to the range [0, 1]
// - advantage:    bounds the values to a specific range.
// - disadvantage: sensitive to outliers. 
void feature_scaling_on_gpu (int nbDims, int nbPoints,
                             T_real *GPU_dataT,
                             float *GPU_dimMax, float *GPU_dimMin)
{
    // Declaration
    dim3 Dg, Db;
    
    // Find the maximum and minimum values of each dimension
    find_min_max(nbDims, nbPoints,         // input
                 GPU_dataT,                // input
                 GPU_dimMax, GPU_dimMin);  // output
    
    // Min-max scaling
    Db.x = BsXN;
    Db.y = BSYD;
    if (BsXN*BSYD > 1024) {
        printf("<-bsxn>*BSYD should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = nbDims/Db.y + (nbDims%Db.y > 0 ? 1 : 0);
    kernel_min_max_scaling<<<Dg,Db>>>(nbPoints, nbDims,        // input
                                      GPU_dimMax, GPU_dimMin,  // input
                                      GPU_dataT);              // input & output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
}




void inverse_feature_scaling_on_gpu (int nbDims, int nbPoints,
                                     float *GPU_dimMax, float *GPU_dimMin,
                                     T_real *GPU_dataT)
{
    // Declaration
    dim3 Db, Dg;
    
    // Inverse min-max scaling
    Db.x = BsXN;
    Db.y = BSYD;
    if (BsXN*BSYD > 1024) {
        printf("<-bsxn>*BSYD should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = nbDims/Db.y + (nbDims%Db.y > 0 ? 1 : 0);
    kernel_inverse_min_max_scaling<<<Dg,Db>>>(nbPoints, nbDims,        // input
                                              GPU_dimMax, GPU_dimMin,  // input
                                              GPU_dataT);              // input & output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
}



void compute_unscaled_centroids (float *GPU_dimMax, float *GPU_dimMin,
                                 int nbClusters, int nbDims,
                                 T_real *GPU_centroids)
{
    T_real alpha = 1.0f;
    T_real beta = 0.0f;
    
    T_real *GPU_centroidsT;
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centroidsT, (sizeof(T_real)*nbDims)*nbClusters));
    
    // Transpose GPU_centroids to GPU_centroidsT
    CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(handleCUBLAS,
                         CUBLAS_OP_T, CUBLAS_OP_N,
                         nbClusters, nbDims,
                         &alpha, GPU_centroids, nbDims,
                         &beta, NULL, nbClusters,
                         GPU_centroidsT, nbClusters));
    
    inverse_feature_scaling_on_gpu(nbDims, nbClusters,      // input
                                   GPU_dimMax, GPU_dimMin,  // input
                                   GPU_centroidsT);         // input & output
    
    // Transpose GPU_centroidsT to GPU_centroids
    CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(handleCUBLAS,
                         CUBLAS_OP_T, CUBLAS_OP_N,
                         nbDims, nbClusters,
                         &alpha, GPU_centroidsT, nbClusters,
                         &beta, NULL, nbDims,
                         GPU_centroids, nbDims));

    CHECK_CUDA_SUCCESS(cudaFree(GPU_centroidsT));
}