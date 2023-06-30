#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>                // CUBLAS_GEAM
#include <curand_kernel.h>            // CURAND_UNIFORM
#include <thrust/device_vector.h>     // thrust::device_ptr
#include <thrust/execution_policy.h>  // thrust::host, thrust::device, thrust::cuda::par.on(stream)
#include <thrust/scan.h>              // thrust::exclusive_scan
#include <omp.h>                      // omp_get_wtime
#include <math.h>                     // Library functions (e.g. exp, expf, pow, powf, log, logf, sqrt, sqrtf, ceil)
#include <float.h>                    // Library Macros (e.g. FLT_MAX, FLT_MIN)

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/kmeans/kmeans_gpu.h"


/*-------------------------------------------------------------------------------*/
/* Select initial centroids                                                      */
/*-------------------------------------------------------------------------------*/
// Initialize centroids randomly
__global__ void kernel_setup_curand (int nbClusters, unsigned long long seed,
                                     curandState *state)  
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < nbClusters)
        curand_init(seed, tid, 0, &state[tid]);
}


__global__ void kernel_initialize_centroids_at_random (curandState *state, T_real *GPU_dataT,
                                                       int nbPoints, int nbDims, int nbClusters,
                                                       T_real *GPU_centroidsT)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < nbClusters) {
        curandState localState = state[tid];
        // curand_uniform() returns a pseudo-random float in the range (0.0, 1.0]
        int idx = (ceil(nbPoints * CURAND_UNIFORM(&localState))) - 1;  // Control idx in [0, nbPoints - 1]
        for (int j = 0; j < nbDims; j++) {
            index_t dataTIdx = ((index_t)j)*((index_t)nbPoints) + ((index_t)idx);
            GPU_centroidsT[j*nbClusters + tid] = GPU_dataT[dataTIdx];   // The right part is not coalesced
        }
    }
}



__global__ void kernel_calculate_sampling_probability_d2 (double *GPU_d2, double *GPU_d2_total,
                                                          int nbPoints,
                                                          double *GPU_sampProb)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double shD2;
    
    if (threadIdx.x == 0) {
        shD2 = *GPU_d2_total;
    }
    
    __syncthreads();
    
    if (tid < nbPoints) {
        GPU_sampProb[tid] = GPU_d2[tid] / shD2;
    }
}


__global__ void kernel_find_center_idx (int nbPoints, double randValue,
                                        double *GPU_inScanSum, double *GPU_exScanSum,
                                        int *GPU_centerIdx)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nbPoints) {
        double InScanSum = GPU_inScanSum[tid];
        double ExScanSum = GPU_exScanSum[tid];
        if (randValue >= ExScanSum && randValue < InScanSum)
            *GPU_centerIdx = tid;
    }
}


template <int BLOCK_SIZE_X>
__global__ void kernel_calculate_d2 (int nbPoints, int nbDims, int k, int idx,
                                     T_real *GPU_centroids, T_real *GPU_dataT,
                                     double *GPU_d2, double *GPU_d2_total)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dimIdx = threadIdx.x;
    T_real *shD2 = shBuff;                     // blockDim.x T_reals  in shared memory
    T_real *shCenterDim = &shD2[blockDim.x];   // nbDims     T_reals  in shared memory
    
    if (k > 0) {
        shD2[threadIdx.x] = (tid < nbPoints ? GPU_d2[tid] : 0.0f);
    } else {
        shD2[threadIdx.x] = (tid < nbPoints ? FLT_MAX : 0.0f);
    }

    while (dimIdx < nbDims) {
        index_t dataTIdx = ((index_t)dimIdx)*((index_t)nbPoints) + ((index_t)idx);
        shCenterDim[dimIdx] = GPU_dataT[dataTIdx];
        GPU_centroids[k*nbDims + dimIdx] = shCenterDim[dimIdx];
        dimIdx += blockDim.x;
    }
    
    __syncthreads();
    
    if (tid < nbPoints) {
        T_real diff, sqDist = 0.0f;
        for (int j = 0; j < nbDims; j++) {
            index_t dataTIdx = ((index_t)j)*((index_t)nbPoints) + ((index_t)tid);
            diff = GPU_dataT[dataTIdx] - shCenterDim[j];
            sqDist += diff*diff;
        }
        if (sqDist < shD2[threadIdx.x]) {
            shD2[threadIdx.x] = sqDist;
            GPU_d2[tid] = sqDist;
        }
    }
    
    if (BLOCK_SIZE_X > 512) {
        __syncthreads();
        if (threadIdx.x < 512)
            shD2[threadIdx.x] += shD2[threadIdx.x + 512];
        else
            return;
    }

    if (BLOCK_SIZE_X > 256) {
        __syncthreads();
        if (threadIdx.x < 256)
            shD2[threadIdx.x] += shD2[threadIdx.x + 256];
        else
            return;
    }

    if (BLOCK_SIZE_X > 128) {
        __syncthreads();
        if (threadIdx.x < 128)
            shD2[threadIdx.x] += shD2[threadIdx.x + 128];
        else
            return;
    }

    if (BLOCK_SIZE_X > 64) {
        __syncthreads();
        if (threadIdx.x < 64)
            shD2[threadIdx.x] += shD2[threadIdx.x + 64];
        else
            return;
    }

    if (BLOCK_SIZE_X > 32) {
        __syncthreads();
        if (threadIdx.x < 32)
            shD2[threadIdx.x] += shD2[threadIdx.x + 32];
        else
            return;
    }

    if (BLOCK_SIZE_X > 16) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 16)
            shD2[threadIdx.x] += shD2[threadIdx.x + 16];
    }

    if (BLOCK_SIZE_X > 8) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 8)
            shD2[threadIdx.x] += shD2[threadIdx.x + 8];
    }

    if (BLOCK_SIZE_X > 4) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 4)
            shD2[threadIdx.x] += shD2[threadIdx.x + 4];
    }

    if (BLOCK_SIZE_X > 2) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 2)
            shD2[threadIdx.x] += shD2[threadIdx.x + 2];
    }

    if (BLOCK_SIZE_X > 1) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 1)
            shD2[threadIdx.x] += shD2[threadIdx.x + 1];
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(GPU_d2_total, shD2[0]);
    }
}


inline void template_kernel_calculate_d2 (dim3 Dg, dim3 Db, size_t shMemSize,
                                          int nbPoints, int nbDims, int k, int idx,
                                          T_real *GPU_centroids, T_real *GPU_dataT,
                                          double *GPU_d2, double *GPU_d2_total)
{
    switch (Db.x) {
        case 1024: kernel_calculate_d2<1024><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        case 512:  kernel_calculate_d2< 512><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        case 256:  kernel_calculate_d2< 256><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        case 128:  kernel_calculate_d2< 128><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        case 64:   kernel_calculate_d2<  64><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        case 32:   kernel_calculate_d2<  32><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        case 16:   kernel_calculate_d2<  16><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        case 8:    kernel_calculate_d2<   8><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        case 4:    kernel_calculate_d2<   4><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        case 2:    kernel_calculate_d2<   2><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        case 1:    kernel_calculate_d2<   1><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, k, idx,  // input
                                                                    GPU_centroids, GPU_dataT,  // input
                                                                    GPU_d2, GPU_d2_total);     // output
                   break;
        default:   fprintf(stderr, "Unsupported value for Db.x of kernel_calculate_d2 kernel!\n");
                   exit(EXIT_FAILURE);
    }
}



/*-----------------------------------------------------------------------------------------*/
/* Compute point-centroid distances and assign each point to a cluter                      */
/*-----------------------------------------------------------------------------------------*/
extern __shared__ unsigned long long int shTrack[];
template <int BLOCK_SIZE_X>
__global__ void kernel_compute_assign (int nbPoints, int nbDims, int nbClusters,
                                       T_real *GPU_dataT, T_real *GPU_centroids,
                                       int *GPU_labels, unsigned long long int *GPU_trackTotal)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    shTrack[threadIdx.x] = 0;

    if (tid < nbPoints) {
        int min = 0;
        T_real diff, sqDist, minDistSq = FLT_MAX;
        for (int k = 0; k < nbClusters; k++) {
            sqDist = 0.0f;
            // Calculate the square of distance between instance tid and centroid k
            for(int j = 0; j < nbDims; j++) {
                index_t dataTIdx = ((index_t)j)*((index_t)nbPoints) + ((index_t)tid);
                diff = (GPU_dataT[dataTIdx] - GPU_centroids[k*nbDims + j]);
                sqDist += (diff*diff);
            }
            // Find and record the nearest centroid to instance tid
            if (sqDist < minDistSq) {
                minDistSq = sqDist;
                min = k;
            }
        }
        // Change the label if necessary
        if (GPU_labels[tid] != min) {
            shTrack[threadIdx.x] = 1;
            GPU_labels[tid] = min;
        }
    }

    // Count the changes of label into "track": two-part reduction
    // 1 - Parallel reduction of 1D block shared array shTrack[*] into shTrack[0],
    //     kill useless warps step by step, only the first warp survives at the end.
    if (BLOCK_SIZE_X > 512) {
        __syncthreads();
        if (threadIdx.x < 512)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 512];
        else
            return;
    }

    if (BLOCK_SIZE_X > 256) {
        __syncthreads();
        if (threadIdx.x < 256)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 256];
        else
            return;
    }

    if (BLOCK_SIZE_X > 128) {
        __syncthreads();
        if (threadIdx.x < 128)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 128];
        else
            return;
    }

    if (BLOCK_SIZE_X > 64) {
        __syncthreads();
        if (threadIdx.x < 64)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 64];
        else
            return;
    }

    if (BLOCK_SIZE_X > 32) {
        __syncthreads();
        if (threadIdx.x < 32)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 32];
        else
            return;
    }

    if (BLOCK_SIZE_X > 16) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 16)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 16];
    }

    if (BLOCK_SIZE_X > 8) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 8)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 8];
    }

    if (BLOCK_SIZE_X > 4) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 4)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 4];
    }

    if (BLOCK_SIZE_X > 2) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 2)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 2];
    }

    if (BLOCK_SIZE_X > 1) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 1)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 1];
    }
    
    // 2 - Final reduction into a global array
    if (threadIdx.x == 0) {
        if (shTrack[0] > 0)
            atomicAdd(GPU_trackTotal, shTrack[0]);
    }
}


inline void template_kernel_compute_assign (dim3 Dg, dim3 Db, size_t shMemSize,
                                            int nbPoints, int nbDims, int nbClusters,
                                            T_real *GPU_dataT, T_real *GPU_centroids,
                                            int *GPU_labels, unsigned long long int *GPU_trackTotal)
{
    switch (Db.x) {
        case 1024: kernel_compute_assign<1024><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        case 512:  kernel_compute_assign< 512><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        case 256:  kernel_compute_assign< 256><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        case 128:  kernel_compute_assign< 128><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        case 64:   kernel_compute_assign<  64><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        case 32:   kernel_compute_assign<  32><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        case 16:   kernel_compute_assign<  16><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        case 8:    kernel_compute_assign<   8><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        case 4:    kernel_compute_assign<   4><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        case 2:    kernel_compute_assign<   2><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        case 1:    kernel_compute_assign<   1><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters, // input
                                                                      GPU_dataT, GPU_centroids,     // input
                                                                      GPU_labels, GPU_trackTotal);  // output
                   break;
        default:   fprintf(stderr, "Unsupported value for Db.x of kernel_compute_assign kernel!\n"); 
                   exit(EXIT_FAILURE);
    }
}



/*-------------------------------------------------------------------------------*/
/* Update centroids                                                              */
/*-------------------------------------------------------------------------------*/
// If nbPoints/nbClusters <= tholdUsePackages (flagUsePackages = false)
__global__ void kernel_update_centroids_step1 (int nbPoints, int nbDims, int nbClusters,
                                               T_real *GPU_dataT, int *GPU_labels,
                                               int *GPU_count, T_real *GPU_centroidsT)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nbPoints) {
        int k = GPU_labels[tid];
        for (int j = 0; j < nbDims; j++) {
            index_t dataTIdx = ((index_t)j)*((index_t)nbPoints) + ((index_t)tid);
            atomicAdd(&GPU_centroidsT[j*nbClusters + k], GPU_dataT[dataTIdx]);
        }
        atomicAdd(&GPU_count[k], 1);
    }
}


__global__ void kernel_update_centroids_step1_using_shared_memory (int nbPoints, int nbDims, int nbClusters,
                                                                   T_real *GPU_dataT, int *GPU_labels,
                                                                   int *GPU_count, T_real *GPU_centroidsT)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    T_real *shCentroidT = shBuff;                              // nbDims*nbClusters  T_reals  in shared memory
    int    *shCount = (int*)&shCentroidT[nbDims*nbClusters];   // nbClusters         ints     in shared memory
    
    int cltIdx = threadIdx.x;       
    while (cltIdx < nbClusters) {
        for (int j = 0; j < nbDims; j++) {
            shCentroidT[j*nbClusters + cltIdx] = 0.0f;
        }
        shCount[threadIdx.x] = 0;
        cltIdx += blockDim.x;
    }
    
    __syncthreads();
    
    if(tid < nbPoints){
        int k = GPU_labels[tid];
        for (int j = 0; j < nbDims; j++) {
            index_t dataTIdx = ((index_t)j)*((index_t)nbPoints) + ((index_t)tid);
            atomicAdd_block(&shCentroidT[j*nbClusters + k], GPU_dataT[dataTIdx]);  
            // atomicAdd_block is faster than atomicAdd, but take care when atomicAdd_block can be used.
            // atomicAdd(&shCentroidT[j*nbClusters + k], GPU_dataT[j*nbPoints + tid]);
        }
        atomicAdd_block(&shCount[k], 1);
    }
    
    __syncthreads();
    
    cltIdx = threadIdx.x;
    while (cltIdx < nbClusters) {
        if (shCount[cltIdx] > 0) {
            for (int j = 0; j < nbDims; j++) {
                atomicAdd(&GPU_centroidsT[j*nbClusters + cltIdx], shCentroidT[j*nbClusters + cltIdx]);
            }
            atomicAdd(&GPU_count[cltIdx], shCount[cltIdx]);
        }
        cltIdx += blockDim.x;
    }
}


// Method from a student at CentraleSupelec
__global__ void kernel_update_centroids_step1_ref (int nbPoints, int nbDims, int nbClusters,
                                                   T_real *GPU_dataT, int *GPU_labels,
                                                   int *GPU_count, T_real *GPU_centroidsT)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    T_real *shCentroidT = shBuff;                              // nbDims*nbClusters  T_reals  in shared memory
    int    *shCount = (int*)&shCentroidT[nbDims*nbClusters];   // nbClusters         ints     in shared memory
    
    int cltIdx = threadIdx.x;       
    while (cltIdx < nbClusters) {
        for (int j = 0; j < nbDims; j++) {
            shCentroidT[j*nbClusters + cltIdx] = 0.0f;
        }
        shCount[threadIdx.x] = 0;
        cltIdx += blockDim.x;
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            int k = GPU_labels[tid + i];
            for (int j = 0; j < nbDims; j++) {
                index_t dataTIdx = ((index_t)j)*((index_t)nbPoints) + ((index_t)(tid + i));
                shCentroidT[j*nbClusters + k] += GPU_dataT[dataTIdx];
            }
            shCount[k]++;
        }
    }
    
    __syncthreads();
    
    cltIdx = threadIdx.x;
    while (cltIdx < nbClusters) {
        if (shCount[cltIdx] > 0) {
            for (int j = 0; j < nbDims; j++) {
                atomicAdd(&GPU_centroidsT[j*nbClusters + cltIdx], shCentroidT[j*nbClusters + cltIdx]);
            }
            atomicAdd(&GPU_count[cltIdx], shCount[cltIdx]);
        }
        cltIdx += blockDim.x;
    }
}


__global__ void kernel_update_centroids_step2 (int nbDims, int nbClusters,
                                               int *GPU_count, 
                                               T_real *GPU_centroidsT)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < nbClusters){
        int count = GPU_count[tid];
        if (count != 0) {
            for (int j = 0; j < nbDims; j++)
                GPU_centroidsT[j*nbClusters + tid] = GPU_centroidsT[j*nbClusters + tid] / count;
        }
    }
}


template <int nbDims>
__global__ void kernel_update_centroids_step1_child (int nbPoints, int nbClusters, int nbPackages,
                                                     T_real *GPU_dataT, int *GPU_labels,
                                                     int pid, int offset, int length,
                                                     int *GPU_count, T_real *GPU_packages)
{
    // To align the access of shared memory, the ordering of different types of variable should be 
    // 1) double (multiple of 8 bytes) 2) float / int (multiple of 4 bytes) 3) char (multiple of 1 bytes)
    T_real *shTabV = shBuff;                               // blockDim.y*blockDim.x  T_reals   Tab of data values          in shared memory
    int    *shTabL = (int*)&shTabV[blockDim.y*blockDim.x]; // blockDim.x             ints      Tab of labels (cluster Id)  in shared memory

    // 2D block, 2D grid
    int baseRow = blockIdx.y * blockDim.y;                 // Base row of the block
    int row = baseRow + threadIdx.y;                       // row of child threads
    int baseCol = blockIdx.x * blockDim.x + offset;        // Base column of the block
    int col = baseCol + threadIdx.x;                       // col of child threads
    int cltIdx = threadIdx.y * blockDim.x + threadIdx.x;   // 1D cluster index

    // Load the values and cluster labels of instances into sh mem tables
    if (col < (offset + length) && row < nbDims) {
        index_t dataTIdx = ((index_t)row)*((index_t)nbPoints) + ((index_t)col);
        shTabV[cltIdx] = GPU_dataT[dataTIdx];
        if (threadIdx.y == 0)
            shTabL[threadIdx.x] = GPU_labels[col];
    }

    __syncthreads();  // Wait for all data loaded into the sh mem

    // Compute partial evolution of centroid related to cluster number 'cltIdx'
    if (cltIdx < nbClusters) {   // !!!Attention: blockSize must be no less than nbClusters
        #define BlND (nbDims < BSYD ? nbDims : BSYD) // BlND: nb of dims stored by block
        T_real Sv[BlND];                   // Sum of values in BlND dimensions
        for (int j = 0; j < BlND; j++) {
            Sv[j] = 0.0f;                  // Init the tab Sv to zeros
        }
        int count = 0;                     // Init the counter of instances
        
        // Accumulate contributions to cluster number 'cltIdx'
        for (int x = 0; x < blockDim.x && (baseCol + x) < (offset + length); x++) {
            if (shTabL[x] == cltIdx) {
                count++;
                for (int y = 0; y < blockDim.y && (baseRow + y) < nbDims; y++)
                    Sv[y] += shTabV[y*blockDim.x + x];
            }
        }
        
        // - Save the contribution of block into global contribution of the packages
        if (count != 0) {
            if (blockIdx.y == 0) {
                atomicAdd(&GPU_count[cltIdx], count);
            }
            int dMax = (blockIdx.y == nbDims/blockDim.y ? nbDims%blockDim.y : blockDim.y);
            for (int j = 0; j < dMax; j++) { // BlND_max: nb of dims managed by blk
                index_t pkgIdx = ((index_t)(baseRow + j))*((index_t)nbClusters)*((index_t)nbPackages) + ((index_t)nbClusters)*((index_t)pid) + ((index_t)cltIdx);
                atomicAdd(&GPU_packages[pkgIdx], Sv[j]);
            }
        }
    }
}


template <int BLOCK_SIZE_X>
__global__ void kernel_update_centroids_step1_parent (int nbPoints, int nbDims, int nbClusters, int nbPackages,
                                                      T_real *GPU_dataT, int *GPU_labels, int clustAlgo,
                                                      int *GPU_count, T_real *GPU_packages)
{
    // 1D block
    int tid = threadIdx.x;    // Thread id

    if (tid < nbPackages) {
        int offset, length, quotient, remainder;
        int ns = blockDim.x;  // Nb of streams
        int np = nbPackages/ns + (nbPackages%ns > 0 ? 1 : 0);  // Nb of packages for each stream
        int pid;              // Id of packages
        cudaStream_t stream;
        dim3 Dg, Db;

        cudaStreamCreateWithFlags(&stream, cudaStreamDefault); 

        quotient = nbPoints/nbPackages;
        remainder = nbPoints%nbPackages;

        Db.x = BLOCK_SIZE_X;
        Db.y = BSYD;
        Dg.y = nbDims/Db.y + (nbDims%Db.y > 0 ? 1 : 0);

        for (int i = 0; i < np; i++) {
            pid = i*ns + tid;           // Calculate the id of packages
            if (pid < nbPackages) {
                offset = (pid < remainder ? ((quotient + 1) * pid) : (quotient * pid + remainder));
                length = (pid < remainder ? (quotient + 1) : quotient);
                Dg.x = length/Db.x + (length%Db.x > 0 ? 1 : 0);
                // Launch a child kernel on a stream to process a packages
                if (clustAlgo == KM_GPU) {
                    kernel_update_centroids_step1_child<       NB_DIMS><<<Dg,Db,((sizeof(T_real)*Db.y)*Db.x + sizeof(int)*Db.x),stream>>>(nbPoints, nbClusters, nbPackages, GPU_dataT, GPU_labels, pid, offset, length, GPU_count, GPU_packages);
                } else {
                    kernel_update_centroids_step1_child<MAX_NB_CLUSTERS><<<Dg,Db,((sizeof(T_real)*Db.y)*Db.x + sizeof(int)*Db.x),stream>>>(nbPoints, nbClusters, nbPackages, GPU_dataT, GPU_labels, pid, offset, length, GPU_count, GPU_packages);
                }
            }
        }
        cudaStreamDestroy(stream); 
    }
}


inline void template_kernel_update_centroids_step1_parent (int BsXP, int nbStreamsStep1,
                                                           int nbPoints, int nbDims, int nbClusters, int nbPackages,
                                                           T_real *GPU_dataT, int *GPU_labels, int clustAlgo,
                                                           int *GPU_count, T_real *GPU_packages)
{
    switch (BsXP) {
        case 1024: kernel_update_centroids_step1_parent<1024><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,
                                                                                    GPU_dataT, GPU_labels, clustAlgo,
                                                                                    GPU_count, GPU_packages);
                   break;
        case 512:  kernel_update_centroids_step1_parent< 512><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,
                                                                                    GPU_dataT, GPU_labels, clustAlgo,
                                                                                    GPU_count, GPU_packages);
                   break;
        case 256:  kernel_update_centroids_step1_parent< 256><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,
                                                                                    GPU_dataT, GPU_labels, clustAlgo,
                                                                                    GPU_count, GPU_packages);
                   break;
        case 128:  kernel_update_centroids_step1_parent< 128><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,   // input
                                                                                    GPU_dataT, GPU_labels, clustAlgo,           // input
                                                                                    GPU_count, GPU_packages);                   // output
                   break;
        case 64:   kernel_update_centroids_step1_parent<  64><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,   // input
                                                                                    GPU_dataT, GPU_labels, clustAlgo,           // input
                                                                                    GPU_count, GPU_packages);                   // output
                   break;
        case 32:   kernel_update_centroids_step1_parent<  32><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,   // input
                                                                                    GPU_dataT, GPU_labels, clustAlgo,           // input
                                                                                    GPU_count, GPU_packages);                   // output
                   break;
        case 16:   kernel_update_centroids_step1_parent<  16><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,   // input
                                                                                    GPU_dataT, GPU_labels, clustAlgo,           // input
                                                                                    GPU_count, GPU_packages);                   // output
                   break;
        case 8:    kernel_update_centroids_step1_parent<   8><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,   // input
                                                                                    GPU_dataT, GPU_labels, clustAlgo,           // input
                                                                                    GPU_count, GPU_packages);                   // output
                   break;
        case 4:    kernel_update_centroids_step1_parent<   4><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,   // input
                                                                                    GPU_dataT, GPU_labels, clustAlgo,           // input
                                                                                    GPU_count, GPU_packages);                   // output
                   break;
        case 2:    kernel_update_centroids_step1_parent<   2><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,   // input
                                                                                    GPU_dataT, GPU_labels, clustAlgo,           // input
                                                                                    GPU_count, GPU_packages);                   // output
                   break;
        case 1:    kernel_update_centroids_step1_parent<   1><<<1,nbStreamsStep1>>>(nbPoints, nbDims, nbClusters, nbPackages,   // input
                                                                                    GPU_dataT, GPU_labels, clustAlgo,           // input
                                                                                    GPU_count, GPU_packages);                   // output
                   break;
        default:   fprintf(stderr, "Unsupported value for BsXP of kernel_update_centroids_step1_parent kernel!\n"); 
                   exit(EXIT_FAILURE);
    }
}


__global__ void kernel_update_centroids_step2_child (int nbDims, int nbClusters, int nbPackages, int pid,
                                                     T_real *GPU_packages, int *GPU_count,
                                                     T_real *GPU_centroidsT)
{
    // 1D block in x-axis, 2D grid
    int rowC = blockIdx.y;                                 // Row of child thread
    int colC = blockIdx.x * blockDim.x + threadIdx.x;      // Col of child thread

    if (colC < nbClusters && rowC < nbDims) {
        if (GPU_count[colC] != 0) {
            index_t pkgIdx = ((index_t)rowC)*((index_t)nbClusters)*((index_t)nbPackages) + ((index_t)nbClusters)*((index_t)pid) + ((index_t)colC);
            atomicAdd(&GPU_centroidsT[rowC*nbClusters + colC], GPU_packages[pkgIdx] / GPU_count[colC]);
        }
    }
}


template <int BLOCK_SIZE_X>
__global__ void kernel_update_centroids_step2_parent (int nbDims, int nbClusters, int nbPackages,
                                                      T_real *GPU_packages, int *GPU_count,
                                                      T_real *GPU_centroidsT)
{
    // 1D block
    int tid = threadIdx.x;

    if (tid < nbPackages) {
        int ns = blockDim.x;   // Nb of streams
        int np = nbPackages/ns + (nbPackages%ns > 0 ? 1 : 0); // Nb of packages for each stream
        int pid;               // Id of packages
        cudaStream_t stream;
        dim3 Dg, Db;

        cudaStreamCreateWithFlags(&stream, cudaStreamDefault); 

        Db.x = BLOCK_SIZE_X;
        Db.y = 1;
        Dg.x = nbClusters/Db.x + (nbClusters%Db.x > 0 ? 1 : 0);
        Dg.y = nbDims;

        for (int i = 0; i < np; i++) {
            pid = i*ns + tid;   // Calculate the id of packages
            if (pid < nbPackages) 
                kernel_update_centroids_step2_child<<<Dg,Db,0,stream>>>(nbDims, nbClusters, nbPackages, pid,
                                                                        GPU_packages, GPU_count,
                                                                        GPU_centroidsT);
        }
        cudaStreamDestroy(stream); 
    }
}


inline void template_kernel_update_centroids_step2_parent (int BsXC, int nbStreamsStep2,
                                                           int nbDims, int nbClusters, int nbPackages,
                                                           T_real *GPU_packages, int *GPU_count,
                                                           T_real *GPU_centroidsT)
{
    switch (BsXC) {
        case 1024: kernel_update_centroids_step2_parent<1024><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        case 512:  kernel_update_centroids_step2_parent< 512><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        case 256:  kernel_update_centroids_step2_parent< 256><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        case 128:  kernel_update_centroids_step2_parent< 128><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        case 64:   kernel_update_centroids_step2_parent<  64><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        case 32:   kernel_update_centroids_step2_parent<  32><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        case 16:   kernel_update_centroids_step2_parent<  16><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        case 8:    kernel_update_centroids_step2_parent<   8><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        case 4:    kernel_update_centroids_step2_parent<   4><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        case 2:    kernel_update_centroids_step2_parent<   2><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        case 1:    kernel_update_centroids_step2_parent<   1><<<1,nbStreamsStep2>>>(nbDims, nbClusters, nbPackages,  // input
                                                                                    GPU_packages, GPU_count,         // input
                                                                                    GPU_centroidsT);                 // output
                   break;
        default:   fprintf(stderr, "Unsupported value for BsXC of kernel_update_centroids_step2_parent kernel!\n");
                   exit(EXIT_FAILURE);
    }
}


void seeding (int nbPoints, int nbDims, int nbClusters,
              T_real *GPU_dataT, int *GPU_labels,
              int seedingMethod, unsigned int seedBase,
              bool flagUsePackages, int nbPackages,
              int nbStreamsStep1, int nbStreamsStep2,
              T_real *GPU_centroids)
{
    // General variables
    dim3 Dg, Db;
    size_t shMemSize;
    
    if (seedingMethod == 1) {  // Select nbClusters centroids from nbPoints instances using random sampling
        // Declaration
        unsigned long long seed = (unsigned long long)seedBase;
        curandState *GPU_devStates;  // States for using cuRAND library
        T_real *GPU_centroidsT;
        T_real alpha;
        T_real beta;
        
        // Memory allocation
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centroidsT, (sizeof(T_real)*nbDims)*nbClusters));
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_devStates, sizeof(curandState)*nbClusters));
        
        // Initialize centroids at random
        Db.x = BsXC;
        Db.y = 1;
        Dg.x = nbClusters/Db.x + (nbClusters%Db.x > 0 ? 1 : 0);
        Dg.y = 1;
        kernel_setup_curand<<<Dg,Db>>>(nbClusters, seed,  // input
                                       GPU_devStates);    // output
        CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
        kernel_initialize_centroids_at_random<<<Dg,Db>>>(GPU_devStates, GPU_dataT,      // input
                                                         nbPoints, nbDims, nbClusters,  // input
                                                         GPU_centroidsT);               // output
        CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
        
        // Transpose GPU_centroidsT to GPU_centroids
        alpha = 1.0f;
        beta = 0.0f;
        CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(handleCUBLAS,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             nbDims, nbClusters,
                             &alpha, GPU_centroidsT, nbClusters,
                             &beta, NULL, nbDims,
                             GPU_centroids, nbDims));
        
        // Memory deallocation
        CHECK_CUDA_SUCCESS(cudaFree(GPU_devStates));
        CHECK_CUDA_SUCCESS(cudaFree(GPU_centroidsT));
    }
    
    
    if (seedingMethod == 2) {   // Select nbClusters centroids from nbPoints instances using D² sampling
            // Declaration
            unsigned int seed;  // time(NULL) will be different each time you launch the program
            int centerIdx;
            double randValue;
            double *GPU_d2;         // Array for D^2 defined in k-means++
            double *GPU_d2_total;   // Variable for the sum of D^2
            double *GPU_sampProb;   // Array for the sampling probability
            double *GPU_inScanSum;  // Array for the accumulated sampling probability by inclusive scan
            double *GPU_exScanSum;  // Array for the accumulated sampling probability by exclusive scan
            int    *GPU_centerIdx;  // Variable for the index of sampled centroid
            
            // Memory allocation
            CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_d2, sizeof(double)*nbPoints));
            CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_d2_total, sizeof(double)));
            CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_sampProb, sizeof(double)*nbPoints));
            CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_inScanSum, sizeof(double)*nbPoints));
            CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_exScanSum, sizeof(double)*nbPoints));
            CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centerIdx, sizeof(int)));
            
            // D² sampling from nbPoints instances
            seed = seedBase;
            randValue = 0.0;
            Db.x = BsXN;
            Db.y = 1;
            Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
            Dg.y = 1;
            shMemSize = sizeof(T_real)*Db.x +   // T_real shD2[blockDim.x] or T_real shWD2[blockDim.x]
                     sizeof(T_real)*nbDims;     // T_real shCenterDim[nbDims]
            if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
                printf("The kernel_calculate_d2 kernel needs too much shared memory per block (%lu bytes)!\n", shMemSize);
                exit(EXIT_FAILURE);
            }
            for (int k = 0; k < nbClusters; k++) {
                if (k == 0) {
                    centerIdx = rand_r(&seed)/(double)RAND_MAX * nbPoints;  // rand_r() returns a pseudo-random integer in the range [0, RAND_MAX]
                } else {
                    randValue = rand_r(&seed)/(double)RAND_MAX;
                    kernel_find_center_idx<<<Dg,Db>>>(nbPoints, randValue,           // input
                                                      GPU_inScanSum, GPU_exScanSum,  // input
                                                      GPU_centerIdx);                // output
                    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
                    CHECK_CUDA_SUCCESS(cudaMemcpy(&centerIdx, GPU_centerIdx, sizeof(int), cudaMemcpyDeviceToHost));
                }
                
                // Initialization
                CHECK_CUDA_SUCCESS(cudaMemset(GPU_d2_total, 0, sizeof(double)));
                // Calculate D²(x): the shortest distance from a data point to the closest center that have already been chosen
                template_kernel_calculate_d2(Dg, Db, shMemSize,
                                             nbPoints, nbDims, k, centerIdx,  // input
                                             GPU_centroids, GPU_dataT,        // input
                                             GPU_d2, GPU_d2_total);           // output
                CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
                
                // Calculate sampling probability
                kernel_calculate_sampling_probability_d2<<<Dg,Db>>>(GPU_d2, GPU_d2_total,  // input
                                                                    nbPoints,              // input
                                                                    GPU_sampProb);         // output
                CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
                
                // Calculate inclusive scan & exclusive scan results of sampling probability
                thrust::device_ptr<double> d_SampProb(GPU_sampProb);
                thrust::device_ptr<double> d_inScanSum(GPU_inScanSum);
                thrust::device_ptr<double> d_exScanSum(GPU_exScanSum);
                thrust::inclusive_scan(thrust::device, d_SampProb, d_SampProb + nbPoints, d_inScanSum);
                thrust::exclusive_scan(thrust::device, d_SampProb, d_SampProb + nbPoints, d_exScanSum, 0.0);
            }
            
            // Memory deallocation
            CHECK_CUDA_SUCCESS(cudaFree(GPU_d2));
            CHECK_CUDA_SUCCESS(cudaFree(GPU_d2_total));
            CHECK_CUDA_SUCCESS(cudaFree(GPU_sampProb));
            CHECK_CUDA_SUCCESS(cudaFree(GPU_inScanSum));
            CHECK_CUDA_SUCCESS(cudaFree(GPU_exScanSum));
            CHECK_CUDA_SUCCESS(cudaFree(GPU_centerIdx));
    }
}


void compute_assign (int nbPoints, int nbDims, int nbClusters,
                     T_real *GPU_dataT, T_real *GPU_centroids,
                     int *GPU_labels, unsigned long long int *track)
{
    // Declaration
    dim3 Dg, Db;
    size_t shMemSize;
    unsigned long long int *GPU_trackTotal; // Sum of label changes in two consecutive iterations
    
    // Memory allocation & initialization
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_trackTotal, sizeof(unsigned long long int)));
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_trackTotal, 0, sizeof(unsigned long long int)));
    
    // Compute distance & Assign points to clusters 
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    shMemSize = sizeof(unsigned long long int)*Db.x;    // unsigned long long int shTrack[blockDim.x]
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_compute_assign kernel needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    template_kernel_compute_assign(Dg, Db, shMemSize,
                                   nbPoints, nbDims, nbClusters, // input
                                   GPU_dataT, GPU_centroids,     // input
                                   GPU_labels, GPU_trackTotal);  // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    
    // Memory copy from device to host
    CHECK_CUDA_SUCCESS(cudaMemcpy(track, GPU_trackTotal, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
    
    // Memory deallocation
    CHECK_CUDA_SUCCESS(cudaFree(GPU_trackTotal));
}


void update_centroids (int clustAlgo,
                       int nbPoints, int nbDims, int nbClusters,
                       T_real *GPU_dataT, int *GPU_labels,
                       bool flagUsePackages, int nbPackages, T_real *GPU_packages,
                       int nbStreamsStep1, int nbStreamsStep2,
                       int *GPU_count, T_real *GPU_centroidsT)
{
    // Declaration
    dim3 Dg, Db;
    size_t shMemSize;
    
    // Initialization
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_count, 0, sizeof(int)*nbClusters));
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_centroidsT, 0, (sizeof(T_real)*nbDims)*nbClusters));
    
    // Two methods for updating centroids
    if (!flagUsePackages) {  // Traditional method (no use of packages)
        // Step 1 of updating centroids
        Db.x = BsXN;
        Db.y = 1;
        Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
        Dg.y = 1;
        shMemSize = (sizeof(T_real)*nbDims)*nbClusters +   // T_real shCentroidT[nbDims * nbClusters]
                 sizeof(int)*nbClusters;                   // int    shCount[nbClusters]
        if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
            printf("The kernel_update_centroids_step1_using_shared_memory kernel needs too much shared memory per block (%lu bytes)!\n", shMemSize);
            exit(EXIT_FAILURE);
        }
        // kernel_update_centroids_step1<<<Dg,Db>>>(nbPoints, nbDims, nbClusters,  // input
                                                 // GPU_dataT, GPU_labels,         // input
                                                 // GPU_count, GPU_centroidsT);    // output
        kernel_update_centroids_step1_using_shared_memory<<<Dg, Db, shMemSize>>>(nbPoints, nbDims, nbClusters,  // input
                                                                                 GPU_dataT, GPU_labels,         // input
                                                                                 GPU_count, GPU_centroidsT);    // output
        // kernel_update_centroids_step1_ref<<<Dg, Db, (sizeof(T_real)*nbDims)*nbClusters + sizeof(int)*nbClusters>>>(nbPoints, nbDims, nbClusters,  // input
                                                                                                                   // GPU_dataT, GPU_labels,         // input
                                                                                                                   // GPU_count, GPU_centroidsT);    // output
        CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
        
        // Step 2 of updating centroids
        Db.x = BsXC;
        Db.y = 1;
        Dg.x = nbClusters/Db.x + (nbClusters%Db.x > 0 ? 1 : 0);
        Dg.y = 1;
        kernel_update_centroids_step2<<<Dg,Db>>>(nbDims, nbClusters,  // input
                                                 GPU_count,           // input
                                                 GPU_centroidsT);     // input & output
        CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    
    } else {  // Two-step summation method (use of packages)
        // Initialization
        CHECK_CUDA_SUCCESS(cudaMemset(GPU_packages, 0, ((sizeof(T_real)*nbDims)*nbClusters)*nbPackages));
        
        // Step 1 of updating centroids
        if (BsXP*BSYD < nbClusters || BsXP*BSYD > 1024) {
            printf("<-bsxp>*BSYD has to be in [NB_CLUSTERS, 1024]!\n");
            exit(EXIT_FAILURE);
        }
        template_kernel_update_centroids_step1_parent(BsXP, nbStreamsStep1,
                                                      nbPoints, nbDims, nbClusters, nbPackages,  // input
                                                      GPU_dataT, GPU_labels, clustAlgo,          // input
                                                      GPU_count, GPU_packages);                  // output
        CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
        
        // Step 2 of updating centroids
        template_kernel_update_centroids_step2_parent(BsXC, nbStreamsStep2,
                                                      nbDims, nbClusters, nbPackages,  // input
                                                      GPU_packages, GPU_count,         // input
                                                      GPU_centroidsT);                 // output
        CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    }
}



void kmeans_gpu (int clustAlgo,
                 int nbPoints, int nbDims, int nbClusters, T_real *GPU_dataT,
                 int seedingMethod, unsigned int seedBase, T_real tolKMGPU, int maxNbItersKM,
                 int tholdUsePackages, int nbPackages, int nbStreamsStep1, int nbStreamsStep2,
                 int *nbIters, int *GPU_count, T_real *GPU_centroids, int *GPU_labels)
{
    // Declaration
    double begin, finish;
    unsigned long long int track;   // Number of label changes in two consecutive iterations
    T_real changeRatio;
    T_real *GPU_centroidsT;         // Array for the transposed matrix of centroids
    T_real *GPU_packages;           // Array for the packages used in UpdateCentroids
    bool flagUsePackages;
    T_real alpha, beta;
    
    // Initialization
    *nbIters = 0;
    flagUsePackages = (nbPoints/nbClusters > tholdUsePackages ? true: false);
    CHECK_CUDA_SUCCESS(cudaHostRegister(&track, sizeof(unsigned long long int), cudaHostRegisterPortable));
    //CHECK_CUDA_SUCCESS(cudaMemset(GPU_labels, 0, sizeof(int)*nbPoints));
    alpha = 1.0f;
    beta = 0.0f;
    
    // Memory allocation
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centroidsT, (sizeof(T_real)*nbDims)*nbClusters));
    if (flagUsePackages) {
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_packages, ((sizeof(T_real)*nbDims)*nbClusters)*nbPackages));
    }

    // Seeding: initialize centroids
    begin = omp_get_wtime(); 
    if (INPUT_INITIAL_CENTROIDS == "") {
        seeding(nbPoints, nbDims, nbClusters,   // input
                GPU_dataT, GPU_labels,                         // input
                seedingMethod, seedBase,                       // input
                flagUsePackages, nbPackages,                   // input
                nbStreamsStep1, nbStreamsStep2,                // input
                GPU_centroids);                                // output
    } else {
        T_real *centroids;
        centroids = (T_real *) malloc((sizeof(T_real)*nbClusters)*nbDims);
        // read_file_real(centroids, nbClusters, nbDims, INPUT_INITIAL_CENTROIDS, "\t", 0);
        read_file_real(centroids, nbClusters, nbDims, INPUT_INITIAL_CENTROIDS, " ", 0);  // " " delimter for InitialCentroids_InputDataset-50million.txt
        CHECK_CUDA_SUCCESS(cudaMemcpy(GPU_centroids, centroids, (sizeof(T_real)*nbClusters)*nbDims, cudaMemcpyHostToDevice));
        free(centroids);
    }
    finish = omp_get_wtime();
    Tomp_gpu_seeding += (finish - begin);
    
    do {
        // Compute distance & Assign points to clusters 
        begin = omp_get_wtime();
        compute_assign(nbPoints, nbDims, nbClusters,  // input
                       GPU_dataT, GPU_centroids,      // input
                       GPU_labels, &track);           // output
        finish = omp_get_wtime();
        Tomp_gpu_computeAssign += (finish - begin);
        
        // Update centroids
        begin = omp_get_wtime();
        update_centroids(clustAlgo,                                 // input
                         nbPoints, nbDims, nbClusters,              // input
                         GPU_dataT, GPU_labels,                     // input
                         flagUsePackages, nbPackages, GPU_packages, // input
                         nbStreamsStep1, nbStreamsStep2,            // input
                         GPU_count, GPU_centroidsT);                // output
        CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(handleCUBLAS,  // Transpose GPU_centroidsT to GPU_centroids
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             nbDims, nbClusters,
                             &alpha, GPU_centroidsT, nbClusters,
                             &beta, NULL, nbDims,
                             GPU_centroids, nbDims));
        finish = omp_get_wtime();
        Tomp_gpu_updateCentroids += (finish - begin);
        
        // Update parameters of iteration control
        (*nbIters)++;
        changeRatio = (T_real)track / (T_real)nbPoints;
        // printf("        track = %llu\tchangeRatio = %f\n", track, changeRatio); 
        
    } while (changeRatio > tolKMGPU && (*nbIters) < maxNbItersKM);   // Check stopping criteria
        
    // Memory deallocation
    CHECK_CUDA_SUCCESS(cudaFree(GPU_centroidsT));
    if (flagUsePackages) {
        CHECK_CUDA_SUCCESS(cudaFree(GPU_packages));
    }
    
    // Destroy handle & Unregister host memory
    CHECK_CUDA_SUCCESS(cudaHostUnregister(&track));
}


__global__ void kernel_attach_to_reps (int nbPoints, int nbDims, int nbReps,
                                       T_real *GPU_dataT, T_real *GPU_reps,
                                       int *GPU_labels)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nbPoints) {
        int min = 0;
        T_real diff, sqDist, minDistSq = FLT_MAX;
        for (int k = 0; k < nbReps; k++) {
            sqDist = 0.0f;
            // Calculate the square of distance between instance tid and centroid k
            for(int j = 0; j < nbDims; j++) {
                index_t dataTIdx = ((index_t)j)*((index_t)nbPoints) + ((index_t)tid);
                diff = (GPU_dataT[dataTIdx] - GPU_reps[k*nbDims + j]);
                sqDist += (diff*diff);
            }
            // Find and record the nearest centroid to instance tid
            if (sqDist < minDistSq) {
                minDistSq = sqDist;
                min = k;
            }
        }
        GPU_labels[tid] = min;
    }
}


void gpu_attach_to_representative(int nbPoints, int nbDims, int nbReps,
                                  T_real *GPU_dataT, T_real *GPU_reps, 
                                  int *GPU_labels)
{
    // float elapsed = 0.0f;
    dim3 Dg, Db;
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    kernel_attach_to_reps<<<Dg, Db>>>(nbPoints, nbDims, nbReps, // input
                                   GPU_dataT, GPU_reps,     // input
                                   GPU_labels);  // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
}


__global__ void kernel_input_data_attachment (int nbPoints, int *GPU_labelsReps,
                                              int *GPU_labels)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nbPoints) {
        int rid = GPU_labels[tid];
        GPU_labels[tid] = GPU_labelsReps[rid];
    }
}

void gpu_membership_attachment(int nbPoints, int *GPU_labelsReps,
                               int *GPU_labels)
{
    dim3 Dg, Db;
    
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    kernel_input_data_attachment<<<Dg, Db>>>(nbPoints, GPU_labelsReps,
                                             GPU_labels);
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
}