#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/spectral_clustering/constr_sim_lap_matrix_in_dense.h"


template <int BLOCK_SIZE_X>
__global__ void kernel_construct_similarity_degree_matrix (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                           T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                           T_real *GPU_sim, T_real *GPU_deg)
{
    // 2D block, 2D grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T_real *shTabSim = shBuff;       // blockDim.y*blockDim.x  T_reals  in shared memory
    
    shTabSim[threadIdx.y*blockDim.x + threadIdx.x] = 0.0f;
    //__syncthreads();
    
    if (row < nbPoints && col < nbPoints) {
        #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
            T_real diff, sqDist = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                sqDist += diff*diff;
            }
            if (sqDist < tholdSqDist && row != col) {
                shTabSim[threadIdx.y*blockDim.x + threadIdx.x] = 1.0f;
            }
            size_t idx = ((size_t)row)*((size_t)nbPoints) + ((size_t)col);      // Avoid integer overflow
            GPU_sim[idx] = shTabSim[threadIdx.y*blockDim.x + threadIdx.x];
        #endif
        
        #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
            T_real diff, sqDist = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                sqDist += diff*diff;
            }
            if (sqDist < tholdSqDist && row != col) {
                shTabSim[threadIdx.y*blockDim.x + threadIdx.x] = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));  // possible underflow of shTabSim
            }
            size_t idx = ((size_t)row)*((size_t)nbPoints) + ((size_t)col);      // Avoid integer overflow
            GPU_sim[idx] = shTabSim[threadIdx.y*blockDim.x + threadIdx.x];
        #endif
        
        #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
            T_real diff, sqDist = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                sqDist += diff*diff;
            }
            T_real sim = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
            if (sim > tholdSim && row != col) {
                shTabSim[threadIdx.y*blockDim.x + threadIdx.x] = sim;
            }
            size_t idx = ((size_t)row)*((size_t)nbPoints) + ((size_t)col);      // Avoid integer overflow
            GPU_sim[idx] = shTabSim[threadIdx.y*blockDim.x + threadIdx.x];
        #endif
        
        #ifdef COS_SIM_WITH_THOLD // Cosine similarity with threshold
            T_real elm1, elm2, dot = 0.0f, sq1 = 0.0f, sq2 = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                elm1 = GPU_dataT[idxOffset + (index_t)row];
                elm2 = GPU_dataT[idxOffset + (index_t)col];
                dot += elm1*elm2;
                sq1 += elm1*elm1;
                sq2 += elm2*elm2;
            }
            T_real sqSim = (dot*dot)/(sq1*sq2);
            if (sqSim > tholdSim*tholdSim && row != col) {
                shTabSim[threadIdx.y*blockDim.x + threadIdx.x] = SQRT(sqSim);
            }
            size_t idx = ((size_t)row)*((size_t)nbPoints) + ((size_t)col);      // Avoid integer overflow
            GPU_sim[idx] = shTabSim[threadIdx.y*blockDim.x + threadIdx.x];
        #endif
    }
    
    // 1 - Classic reduction of the block shared array
    //     shTabSim[tidPerBlock] into shTabSim[threadIdx.y*blockDim.x],
    //     kill useless warps step by step, 
    //     only the first warp survives at the end.
    if (BLOCK_SIZE_X > 512) {
        __syncthreads();
        if (threadIdx.x < 512)
            shTabSim[threadIdx.y*blockDim.x + threadIdx.x] += shTabSim[threadIdx.y*blockDim.x + threadIdx.x + 512];
        else
            return;
    }

    if (BLOCK_SIZE_X > 256) {
        __syncthreads();
        if (threadIdx.x < 256)
            shTabSim[threadIdx.y*blockDim.x + threadIdx.x] += shTabSim[threadIdx.y*blockDim.x + threadIdx.x + 256];
        else
            return;
    }

    if (BLOCK_SIZE_X > 128) {
        __syncthreads();
        if (threadIdx.x < 128)
            shTabSim[threadIdx.y*blockDim.x + threadIdx.x] += shTabSim[threadIdx.y*blockDim.x + threadIdx.x + 128];
        else
            return;
    }

    if (BLOCK_SIZE_X > 64) {
        __syncthreads();
        if (threadIdx.x < 64)
            shTabSim[threadIdx.y*blockDim.x + threadIdx.x] += shTabSim[threadIdx.y*blockDim.x + threadIdx.x + 64];
        else
            return;
    }

    if (BLOCK_SIZE_X > 32) {
        __syncthreads();
        if (threadIdx.x < 32)
            shTabSim[threadIdx.y*blockDim.x + threadIdx.x] += shTabSim[threadIdx.y*blockDim.x + threadIdx.x + 32];
        else
            return;
    }

    if (BLOCK_SIZE_X > 16) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 16)
            shTabSim[threadIdx.y*blockDim.x + threadIdx.x] += shTabSim[threadIdx.y*blockDim.x + threadIdx.x + 16];
    }

    if (BLOCK_SIZE_X > 8) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 8)
            shTabSim[threadIdx.y*blockDim.x + threadIdx.x] += shTabSim[threadIdx.y*blockDim.x + threadIdx.x + 8];
    }

    if (BLOCK_SIZE_X > 4) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 4)
            shTabSim[threadIdx.y*blockDim.x + threadIdx.x] += shTabSim[threadIdx.y*blockDim.x + threadIdx.x + 4];
    }

    if (BLOCK_SIZE_X > 2) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 2)
            shTabSim[threadIdx.y*blockDim.x + threadIdx.x] += shTabSim[threadIdx.y*blockDim.x + threadIdx.x + 2];
    }

    if (BLOCK_SIZE_X > 1) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 1)
            shTabSim[threadIdx.y*blockDim.x + threadIdx.x] += shTabSim[threadIdx.y*blockDim.x + threadIdx.x + 1];
    }

    // 2 - Final reduction into the global array
    if (threadIdx.x == 0 && row < nbPoints) {
        if (shTabSim[threadIdx.y*blockDim.x] > 0.0f) { // Error risk due to possible underflow of shTabSim, but unimportant here
            atomicAdd(&GPU_deg[row], shTabSim[threadIdx.y*blockDim.x]);
        }
    }
}


inline void template_kernel_construct_similarity_degree_matrix (dim3 Dg, dim3 Db, size_t shMemSize,
                                                                int nbPoints, int nbDims, T_real *GPU_dataT,
                                                                T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                                T_real *GPU_sim, T_real *GPU_deg)
{
    switch (Db.x) {
        case 1024: kernel_construct_similarity_degree_matrix<1024><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        case 512:  kernel_construct_similarity_degree_matrix< 512><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        case 256:  kernel_construct_similarity_degree_matrix< 256><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        case 128:  kernel_construct_similarity_degree_matrix< 128><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        case 64:   kernel_construct_similarity_degree_matrix<  64><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        case 32:   kernel_construct_similarity_degree_matrix<  32><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        case 16:   kernel_construct_similarity_degree_matrix<  16><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        case 8:    kernel_construct_similarity_degree_matrix<   8><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        case 4:    kernel_construct_similarity_degree_matrix<   4><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        case 2:    kernel_construct_similarity_degree_matrix<   2><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        case 1:    kernel_construct_similarity_degree_matrix<   1><<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                          sigma, tholdSim, tholdSqDist, // input
                                                                                          GPU_sim, GPU_deg);            // output
                   break;
        default:   fprintf(stderr, "Unsupported value for Db.x of kernel_construct_similarity_degree_matrix kernel!\n"); 
                   exit(EXIT_FAILURE);
    }
}


__global__ void kernel_compute_laplacian_matrix (int nbPoints, T_real *GPU_sim, T_real *GPU_deg,
                                                T_real *GPU_lap)
{
    // 2D block, 2D grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < nbPoints && col < nbPoints) {
        size_t idx = ((size_t)row)*((size_t)nbPoints) + ((size_t)col);  // Avoid integer overflow
        T_real deg = GPU_deg[col];
        T_real degL = RSQRT(GPU_deg[row]);
        T_real degR = RSQRT(deg);
        T_real sim = GPU_sim[idx];
        T_real lap;
        if (row != col) {
            lap =  - sim;
        } else {
            lap = deg - sim;
        }
        GPU_lap[idx] = degL * lap * degR;
    }
}



void construct_similarity_degree_matrix (int nbPoints, int nbDims, T_real *GPU_dataT,
                                         T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                         T_real *GPU_sim, T_real *GPU_deg)
{   
    // Declaration
    dim3 Dg, Db;
    size_t shMemSize;
    
    // Compute similarity matrix & degree matrix
    Db.x = BsXN;
    Db.y = BsYN;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = nbPoints/Db.y + (nbPoints%Db.y > 0 ? 1 : 0);
    if (BsXN*BsYN > 1024) {
        printf("<-bsxn>*<-bsyn> should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    shMemSize = (sizeof(T_real)*Db.y)*Db.x;    // T_real shTabSim[blockDim.y*blockDim.x]
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_construct_similarity_degree_matrix kernel needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    template_kernel_construct_similarity_degree_matrix(Dg, Db, shMemSize,
                                                       nbPoints, nbDims, GPU_dataT,   // input
                                                       sigma, tholdSim, tholdSqDist,  // input
                                                       GPU_sim, GPU_deg);             // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    
        // Save similarity matrix and degree matrix
        // T_real *sim;
        // T_real *deg;
        // sim = (T_real *) malloc((sizeof(T_real)*nbPoints)*nbPoints);
        // deg = (T_real *) malloc(sizeof(T_real)*nbPoints);
        // CHECK_CUDA_SUCCESS(cudaMemcpy(sim, GPU_sim, (sizeof(T_real)*nbPoints)*nbPoints, cudaMemcpyDeviceToHost));
        // CHECK_CUDA_SUCCESS(cudaMemcpy(deg, GPU_deg, sizeof(T_real)*nbPoints, cudaMemcpyDeviceToHost));
        // save_file_real(sim, nbPoints, nbPoints, "output/SimilarityMatrix.txt", "\t", 0);
        // save_file_real(deg, nbPoints, 1, "output/DegreeMatrix.txt", "", 0);
        // free(sim);
        // free(deg);
}



void compute_laplacian_matrix (int nbPoints, T_real *GPU_sim, T_real *GPU_deg,
                               T_real *GPU_lap)
{   
    // Declaration
    dim3 Dg, Db;
    
    // Compute normalized symmetric Laplacian matrix
    Db.x = BsXN;
    Db.y = BsYN;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = nbPoints/Db.y + (nbPoints%Db.y > 0 ? 1 : 0);
    if (BsXN*BsYN > 1024) {
        printf("<-bsxn>*<-bsyn> should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    kernel_compute_laplacian_matrix<<<Dg,Db>>>(nbPoints, GPU_sim, GPU_deg,  // input
                                               GPU_lap);                    // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    
        // Save Laplacian matrix
        // T_real *lap;
        // lap = (T_real *) malloc((sizeof(T_real)*nbPoints)*nbPoints);
        // CHECK_CUDA_SUCCESS(cudaMemcpy(lap, GPU_lap, (sizeof(T_real)*nbPoints)*nbPoints, cudaMemcpyDeviceToHost));
        // save_file_real(lap, nbPoints, nbPoints, "output/LaplacianMatrix.txt", "\t", 0);
        // free(lap);
}
