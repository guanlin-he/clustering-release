#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <float.h>   // Library Macros (e.g. FLT_MAX, FLT_MIN)
#include <omp.h>     // omp_get_wtime
#include <cuda.h> 
#include <cuda_runtime.h>
#include <thrust/device_vector.h>       // thrust::device_ptr
#include <thrust/execution_policy.h>    // thrust::host, thrust::device, thrust::cuda::par.on(stream)
#include <thrust/scan.h>                // thrust::exclusive_scan
#include <thrust/extrema.h>             // thrust::max_element, thrust::min_element, thrust::minmax_element
// #include <thrust/pair.h>                // thrust::pair

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/spectral_clustering/constr_epsilon_sim_matrix_in_csr.h"


__global__ void kernel_first_pass_CSR1_1D_grid_1D_blocks (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                          T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                          int *GPU_nnzRow)
{
    // 1D block in x-axis, 1D grid in x-axis but regarded as in y-axis
    int row = blockIdx.x;
    int col = threadIdx.x;
    int nnzThread = 0;
    __shared__ int shNnzRow;

    if (threadIdx.x == 0) {
        shNnzRow  = 0;
    }
    
    __syncthreads();
    
    while (col < nbPoints) {
        // Uniform similarity with threshold for squared distance
        #if defined(UNI_SIM_WITH_SQDIST_THOLD) || defined(GAUSS_SIM_WITH_SQDIST_THOLD)
            T_real diff, sqDist = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                sqDist += diff*diff;
            }
            if (sqDist < tholdSqDist && row != col) {
                nnzThread++;
            }
        #endif
        
        // Gaussian similarity with threshold
        #ifdef GAUSS_SIM_WITH_THOLD
            T_real diff, sqDist = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                sqDist += diff*diff;
            }
            T_real sim = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
            if (sim > tholdSim && row != col) {
                nnzThread++;
            }
        #endif

        // Cosine similarity with threshold
        #ifdef COS_SIM_WITH_THOLD
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
                nnzThread++;
            }
        #endif
        
        col += blockDim.x;
    }   // end of while
    
    if (nnzThread > 0) {
        atomicAdd_block(&shNnzRow, nnzThread);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {  // shNnzRow > 0 is removed because it is expensive
        GPU_nnzRow[row] = shNnzRow;
    }
}


__global__ void kernel_first_pass_CSR1_2D_grid_2D_blocks (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                          int offset, int length,
                                                          T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                          int *GPU_nnzRow)
{
    // 2D blocks, 2D grid
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y + offset;
    int nnzThread = 0;
    int *shNnz = (int*)shBuff;  // shNnz[blockDim.y]
                                                                      
    if (threadIdx.x == 0) {
        shNnz[threadIdx.y] = 0;
    }
    
    __syncthreads();
    
    if (col < nbPoints && row < offset + length) {
        // Uniform or Gaussian similarity with threshold for squared distance
        #if defined(UNI_SIM_WITH_SQDIST_THOLD) || defined(GAUSS_SIM_WITH_SQDIST_THOLD)
            T_real diff, sqDist = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                sqDist += diff*diff;
            }
            if (sqDist < tholdSqDist && row != col) {
                nnzThread++;
            }
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
                nnzThread++;
            }
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
                nnzThread++;
            }
        #endif
        
        if (nnzThread > 0) {
            atomicAdd_block(&shNnz[threadIdx.y], nnzThread);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0 && row < offset + length) {
        if (shNnz[threadIdx.y] > 0) {
            atomicAdd(&GPU_nnzRow[row], shNnz[threadIdx.y]);
        }
    }
}


__global__ void kernel_second_pass_CSR1_1D_grid_1D_blocks (int nbPoints, int nbDims,
                                                           T_real *GPU_dataT, int *GPU_csrRow,
                                                           T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                           T_real *GPU_csrVal, int *GPU_csrCol)
{
    // 1D block in x-axis, 1D grid in x-axis but regarded as in y-axis
    int row = blockIdx.x;
    int col = threadIdx.x;
    int offset = GPU_csrRow[row];
    int maxCol = ((nbPoints - 1)/blockDim.x + 1) * blockDim.x;
    
    // To align the access of shared memory, the ordering of different types of variable should be 
    // 1) double (multiple of 8 bytes) 2) float / int (multiple of 4 bytes) 3) char (multiple of 1 bytes)
    T_real *shValIter = shBuff;                 // blockDim.x  T_reals  in shared memory
    int    *shColIter = (int*)&shValIter[blockDim.x];  // blockDim.x  ints     in shared memory
    int    *shNnzIter = &shColIter[blockDim.x];        // 1  int      in shared memory
    
    if (threadIdx.x == 0) {
        *shNnzIter = 0;
    }
    
    __syncthreads();
    
    while (col < maxCol) {
        if (col < nbPoints) {
            #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[threadIdx.x] = 1.0f;
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[threadIdx.x] = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                T_real sim = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
                shColIter[threadIdx.x] = -1;
                if (sim > tholdSim && row != col) {
                    shValIter[threadIdx.x] = sim;
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
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
                shColIter[threadIdx.x] = -1;
                T_real sqSim = (dot*dot)/(sq1*sq2);
                if (sqSim > tholdSim*tholdSim && row != col) {
                    shValIter[threadIdx.x] = SQRT(sqSim);
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
            #endif
        }
        
        __syncthreads();
        
        if (*shNnzIter > 0 && threadIdx.x == 0) {  // The order of conditions makes a difference!
            for (int i = 0; i < blockDim.x && col + i < nbPoints; i++) {
                if (shColIter[i] != -1) {
                    GPU_csrVal[offset] = shValIter[i];
                    GPU_csrCol[offset] = shColIter[i];
                    offset++;
                }
            }
            *shNnzIter = 0;
        }
        
        __syncthreads();
        
        col += blockDim.x; 
    }   // end of while
}


__global__ void kernel_second_pass_CSR1_1D_grid_2D_blocks (int nbPoints, int nbDims,
                                                           T_real *GPU_dataT, int *GPU_csrRow,
                                                           T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                           T_real *GPU_csrVal, int *GPU_csrCol)
{
    // 2D block, 1D grid in x-axis but regarded as in y-axis
    int row = blockDim.y * blockIdx.x + threadIdx.y;
    int col = threadIdx.x;
    int offset = 0;
    int maxCol = ((nbPoints - 1)/blockDim.x + 1) * blockDim.x;
    int yofs = threadIdx.y * blockDim.x;
    
    // To align the access of shared memory, the ordering of different types of variable should be 
    // 1) double (multiple of 8 bytes) 2) float / int (multiple of 4 bytes) 3) char (multiple of 1 bytes)
    T_real *shValIter = shBuff;                      // blockDim.y*blockDim.x  T_reals  in shared memory
    int    *shColIter = (int*)&shValIter[blockDim.y*blockDim.x]; // blockDim.y*blockDim.x  ints     in shared memory
    int    *shNnzIter = &shColIter[blockDim.y*blockDim.x];  // blockDim.y  int      in shared memory
    
    if (threadIdx.x == 0) {
        shNnzIter[threadIdx.y] = 0;
    }
    
    if (row < nbPoints) {
        offset = GPU_csrRow[row];
    }
    
    __syncthreads();
    
    while (col < maxCol && row < nbPoints) {
        if (col < nbPoints) {
            #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[yofs + threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[yofs + threadIdx.x] = 1.0f;
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[yofs + threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[yofs + threadIdx.x] = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                T_real sim = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
                shColIter[yofs + threadIdx.x] = -1;
                if (sim > tholdSim && row != col) {
                    shValIter[yofs + threadIdx.x] = sim;
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
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
                shColIter[yofs + threadIdx.x] = -1;
                T_real sqSim = (dot*dot)/(sq1*sq2);
                if (sqSim > tholdSim*tholdSim && row != col) {
                    shValIter[yofs + threadIdx.x] = SQRT(sqSim);
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
            #endif
        }
        
        __syncthreads();
        
        if (shNnzIter[threadIdx.y] > 0 && threadIdx.x == 0) {  // The order of conditions makes a difference!
            for (int i = 0; i < blockDim.x && col + i < nbPoints; i++) {
                // if (shValIter[yofs + i] > 0.0f) { // Error risk due to possible underflow of shValIter[yofs + i]
                if (shColIter[yofs + i] != -1) {
                    GPU_csrVal[offset] = shValIter[yofs + i];
                    GPU_csrCol[offset] = shColIter[yofs + i];
                    offset++;
                }
            }
            shNnzIter[threadIdx.y] = 0;
        }
        
        __syncthreads();
        
        col += blockDim.x; 
    }   // end of while
}


__global__ void kernel_full_pass_CSR2_1D_grid_1D_blocks (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                         T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                         int hypoMaxNnzRow, int pad1, int pad2,
                                                         int *GPU_idxNzRowRestart, int *GPU_colRestart,
                                                         int *GPU_nnzRow, T_real *GPU_csrValMaxS, int *GPU_csrColMaxS)
{
    // 1D block in x-axis, 1D grid in x-axis but regarded as in y-axis
    int row = blockIdx.x;
    int col = threadIdx.x;
    int maxCol = ((nbPoints - 1)/blockDim.x + 1) * blockDim.x;
    int flagReachHypo = 0;
    int idxNzRowRestart = 0;
    int colRestart = nbPoints;
    
    // Declare shared memory arrays allocated dynamically
    T_real *shValIter  = shBuff;                                  // blockDim.x            T_reals  in shared memory
    T_real *shNzValMax = &shValIter[blockDim.x + pad1];           // hypoMaxNnzRow + pad2  T_reals  in shared memory
    int    *shColIter  = (int*)&shNzValMax[hypoMaxNnzRow + pad2]; // blockDim.x            ints     in shared memory
    int    *shNzColMax = &shColIter[blockDim.x];                  // hypoMaxNnzRow + pad3  ints     in shared memory
    int    *shNnzIter  = &shNzColMax[hypoMaxNnzRow];              // 1                     ints     in shared memory
    int    *shNnzRow   = &shNnzIter[1];                           // 1                     ints     in shared memory
    int    *shIdxNzRowRestart = &shNnzRow[1];                     // 1                     ints     in shared memory
    
    if (threadIdx.x == 0) {
        *shNnzIter = 0;
        *shNnzRow = 0;
    }
    
    __syncthreads();
    
    while (col < maxCol) {
        if (col < nbPoints) {
            #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[threadIdx.x] = 1.0f;
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[threadIdx.x] = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma)); // possible underflow of shValIter
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                T_real sim = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
                shColIter[threadIdx.x] = -1;
                if (sim > tholdSim && row != col) {
                    shValIter[threadIdx.x] = sim;
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
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
                shColIter[threadIdx.x] = -1;
                if (sqSim > tholdSim*tholdSim && row != col) {
                    shValIter[threadIdx.x] = SQRT(sqSim);
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
            #endif
        } // End of if (col < nbPoints)
        
        __syncthreads();
            
        // Only thread 0 works
        if (*shNnzIter > 0 && threadIdx.x == 0) {  // The order of conditions makes a difference!
        
            int i = 0;
            int idxNzRow = *shNnzRow;
            
            if (flagReachHypo == 0) {
                for (; (i < blockDim.x) && (idxNzRow < hypoMaxNnzRow) && (col + i < nbPoints); i++) {
                    if (i%32 == 0) {
                        idxNzRowRestart = idxNzRow;
                        colRestart      = col + i;
                    }
                    // if (shValIter[i] > 0.0f) { // Error risk due to possible underflow of shValIter[i]
                    if (shColIter[i] != -1) {
                        shNzValMax[idxNzRow] = shValIter[i];
                        shNzColMax[idxNzRow] = col + i;  // shColIter[i]
                        idxNzRow++;
                    }
                } // End of for loop
                if (idxNzRow == hypoMaxNnzRow) {flagReachHypo = 1;}
            } // End of if (flagReachHypo == 0)
            
            for (; (idxNzRow == hypoMaxNnzRow) && (i < blockDim.x) && (col + i < nbPoints); i++) {
                if (i%32 == 0) {
                    idxNzRowRestart = hypoMaxNnzRow;
                    colRestart = col + i;
                }
                // if (shValIter[i] > 0.0f) { // Error risk due to possible underflow of shValIter[i]
                if (shColIter[i] != -1) {
                    idxNzRow++;
                }
            }
            
            *shNnzRow  += (*shNnzIter);
            *shNnzIter = 0;
            
        } // End of if (*shNnzIter > 0 && threadIdx.x == 0)
        
        __syncthreads();
        col += blockDim.x;
        
    } // End of while loop
    
    // Only thread 0 works
    if (threadIdx.x == 0) {
        if (*shNnzRow <= hypoMaxNnzRow) {
            idxNzRowRestart = *shNnzRow;
            colRestart = nbPoints;
        }
        *shIdxNzRowRestart = idxNzRowRestart;
    }
    
    __syncthreads();
    
    col = threadIdx.x;
    index_t idxOffset = ((index_t)row)*((index_t)hypoMaxNnzRow);
    while (col < *shIdxNzRowRestart /*&& col < hypoMaxNnzRow */) {
        GPU_csrValMaxS[idxOffset + (index_t)col] = shNzValMax[col];    // Possible overflow of int-type index when row*hypoMaxNnzRow is very large
        GPU_csrColMaxS[idxOffset + (index_t)col] = shNzColMax[col];    // Possible overflow of int-type index when row*hypoMaxNnzRow is very large
        col += blockDim.x;
    }
    
    // Only thread 0 works
    if (threadIdx.x == 0) {
        GPU_nnzRow[row] = *shNnzRow;
        GPU_idxNzRowRestart[row] = idxNzRowRestart;
        GPU_colRestart[row]      = colRestart;
    }
}


__global__ void kernel_full_pass_CSR2_1D_grid_2D_blocks (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                         T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                         int hypoMaxNnzRow,
                                                         int *GPU_idxNzRowRestart, int *GPU_colRestart,
                                                         int *GPU_nnzRow, T_real *GPU_csrValMaxS, int *GPU_csrColMaxS)
{
    // 2D block, 1D grid in x-axis but regarded as in y-axis
    int row = blockDim.y * blockIdx.x + threadIdx.y;
    int col = threadIdx.x;
    int maxCol = ((nbPoints - 1)/blockDim.x + 1) * blockDim.x;
    int flagReachHypo = 0;
    int idxNzRowRestart = 0;
    int colRestart = nbPoints;
    int yofs = threadIdx.y*blockDim.x;
    int ymofs = threadIdx.y*hypoMaxNnzRow;
    
    // Declare shared memory arrays allocated dynamically
    T_real *shValIter  = shBuff;                                      // blockDim.y*blockDim.x            T_reals  in shared memory
    T_real *shNzValMax = &shValIter[blockDim.y*blockDim.x];           // blockDim.y*hypoMaxNnzRow + pad2  T_reals  in shared memory
    int    *shColIter  = (int*)&shNzValMax[blockDim.y*hypoMaxNnzRow]; // blockDim.y*blockDim.x            ints     in shared memory
    int    *shNzColMax = &shColIter[blockDim.y*blockDim.x];           // blockDim.y*hypoMaxNnzRow + pad3  ints     in shared memory
    int    *shNnzIter  = &shNzColMax[blockDim.y*hypoMaxNnzRow];       // blockDim.y                       ints     in shared memory
    int    *shNnzRow   = &shNnzIter[blockDim.y];                      // blockDim.y                       ints     in shared memory
    int    *shIdxNzRowRestart = &shNnzRow[blockDim.y];                // blockDim.y                       ints     in shared memory
    
    if (threadIdx.x == 0) {
        shNnzIter[threadIdx.y] = 0;
        shNnzRow[threadIdx.y] = 0;
    }
    
    __syncthreads();
    
    while (col < maxCol && row < nbPoints) {
        if (col < nbPoints) {
            #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[yofs + threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[yofs + threadIdx.x] = 1.0f;
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[yofs + threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[yofs + threadIdx.x] = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma)); // possible underflow of shValIter
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                T_real sim = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
                shColIter[yofs + threadIdx.x] = -1;
                if (sim > tholdSim && row != col) {
                    shValIter[yofs + threadIdx.x] = sim;
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
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
                shColIter[yofs + threadIdx.x] = -1;
                if (sqSim > tholdSim*tholdSim && row != col) {
                    shValIter[yofs + threadIdx.x] = SQRT(sqSim);
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
            #endif
        } // End of if (col < nbPoints)
        
        __syncthreads();
            
        
        if (shNnzIter[threadIdx.y] > 0 && threadIdx.x == 0) {  // The order of conditions makes a difference!
        
            int idxNzRow = shNnzRow[threadIdx.y];
            int i = 0;
            
            if (flagReachHypo == 0) {
                for (; (i < blockDim.x) && (idxNzRow < hypoMaxNnzRow) && (col + i < nbPoints); i++) {
                    if (i%32 == 0) {
                        idxNzRowRestart = idxNzRow;
                        colRestart      = col + i;
                    }
                    // if (shValIter[i] > 0.0f) { // Error risk due to possible underflow of shValIter[i]
                    if (shColIter[yofs + i] != -1) {
                        shNzValMax[ymofs + idxNzRow] = shValIter[yofs + i];
                        shNzColMax[ymofs + idxNzRow] = col + i;  // shColIter[yofs + i]
                        idxNzRow++;
                    }
                } // End of for loop
                if (idxNzRow == hypoMaxNnzRow) {flagReachHypo = 1;}
            } // End of if (flagReachHypo == 0)
            
            for (; (idxNzRow == hypoMaxNnzRow) && (i < blockDim.x) && (col + i < nbPoints); i++) {
                if (i%32 == 0) {
                    idxNzRowRestart = hypoMaxNnzRow;
                    colRestart = col + i;
                }
                // if (shValIter[i] > 0.0f) { // Error risk due to possible underflow of shValIter[i]
                if (shColIter[yofs + i] != -1) {
                    idxNzRow++;
                }
            }
            
            shNnzRow[threadIdx.y]  += shNnzIter[threadIdx.y];
            shNnzIter[threadIdx.y] = 0;
            
        } // End of if (shNnzIter[threadIdx.y] > 0 && threadIdx.x == 0)
        
        __syncthreads();
        col += blockDim.x;
        
    } // End of while loop
    
    
    if (threadIdx.x == 0 && row < nbPoints) {
        if (shNnzRow[threadIdx.y] <= hypoMaxNnzRow) {
            idxNzRowRestart = shNnzRow[threadIdx.y];
            colRestart = nbPoints;
        }
        shIdxNzRowRestart[threadIdx.y] = idxNzRowRestart;
    }
    
    __syncthreads();
    
    col = threadIdx.x;
    index_t idxOffset = ((index_t)row)*((index_t)hypoMaxNnzRow);
    while (col < shIdxNzRowRestart[threadIdx.y] && row < nbPoints /*&& col < hypoMaxNnzRow */) {
        GPU_csrValMaxS[idxOffset + (index_t)col] = shNzValMax[ymofs + col];    // Possible overflow of int-type index when row*hypoMaxNnzRow is very large
        GPU_csrColMaxS[idxOffset + (index_t)col] = shNzColMax[ymofs + col];    // Possible overflow of int-type index when row*hypoMaxNnzRow is very large
        col += blockDim.x;
    }
    
    if (threadIdx.x == 0 && row < nbPoints) {
        GPU_nnzRow[row] = shNnzRow[threadIdx.y];
        GPU_idxNzRowRestart[row] = idxNzRowRestart;
        GPU_colRestart[row]      = colRestart;
    }
}


__global__ void kernel_ellpack_to_csr_CSR2_1D_grid_1D_blocks (int *GPU_csrRow, int *GPU_idxNzRowRestart,
                                                              int hypoMaxNnzRow,
                                                              T_real *GPU_csrValMaxS, int *GPU_csrColMaxS,
                                                              T_real *GPU_csrVal, int *GPU_csrCol,
                                                              int *GPU_idxNzTotalRestart)
{
    // 1D block in x-axis, 1D grid in x-axis but regarded as in y-axis
    int row = blockIdx.x;
    int col = threadIdx.x;
    int nnzOffset = GPU_csrRow[row];
    int idxNzRowRestart = GPU_idxNzRowRestart[row];
    index_t idxOffset = ((index_t)row)*((index_t)hypoMaxNnzRow);
    
    while (col < idxNzRowRestart /*&& col < hypoMaxNnzRow*/) {
        GPU_csrVal[nnzOffset + col] = GPU_csrValMaxS[idxOffset + (index_t)col];    // Possible overflow of int-type index when row*hypoMaxNnzRow is very large
        GPU_csrCol[nnzOffset + col] = GPU_csrColMaxS[idxOffset + (index_t)col];    // Possible overflow of int-type index when row*hypoMaxNnzRow is very large
        col += blockDim.x;
    }
    
    if (threadIdx.x == 0) {
        GPU_idxNzTotalRestart[row] = nnzOffset + idxNzRowRestart;
    }
}


__global__ void kernel_ellpack_to_csr_CSR2_1D_grid_2D_blocks (int *GPU_csrRow, int *GPU_idxNzRowRestart,
                                                              int hypoMaxNnzRow, int nbPoints,
                                                              T_real *GPU_csrValMaxS, int *GPU_csrColMaxS,
                                                              T_real *GPU_csrVal, int *GPU_csrCol,
                                                              int *GPU_idxNzTotalRestart)
{
    // 2D block, 1D grid in x-axis but regarded as in y-axis
    int row = blockDim.y * blockIdx.x + threadIdx.y;
    int col = threadIdx.x;
    int nnzOffset, idxNzRowRestart;
    if (row < nbPoints) {
        nnzOffset = GPU_csrRow[row];
        idxNzRowRestart = GPU_idxNzRowRestart[row];
    }
    index_t idxOffset = ((index_t)row)*((index_t)hypoMaxNnzRow);
    
    while (row < nbPoints && col < idxNzRowRestart /*&& col < hypoMaxNnzRow*/) {
        GPU_csrVal[nnzOffset + col] = GPU_csrValMaxS[idxOffset + (index_t)col];    // Possible overflow of int-type index when row*hypoMaxNnzRow is very large
        GPU_csrCol[nnzOffset + col] = GPU_csrColMaxS[idxOffset + (index_t)col];    // Possible overflow of int-type index when row*hypoMaxNnzRow is very large
        col += blockDim.x;
    }
    
    if (threadIdx.x == 0 && row < nbPoints) {
        GPU_idxNzTotalRestart[row] = nnzOffset + idxNzRowRestart;
    }
}



__global__ void kernel_supplementary_pass_CSR2_1D_grid_1D_blocks (int nbPoints, int nbDims, 
                                                                  T_real *GPU_dataT, int *GPU_csrRow,
                                                                  int *GPU_idxNzTotalRestart, int *GPU_colRestart,
                                                                  T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                                  T_real *GPU_csrVal, int *GPU_csrCol)
{
    // 1D block in x-axis, 1D grid in x-axis but regarded as in y-axis
    int row        = blockIdx.x;
    int colRestart = GPU_colRestart[row];
    int col        = colRestart + threadIdx.x;
    int maxCol     = colRestart + ((nbPoints - colRestart - 1)/blockDim.x + 1) * blockDim.x;
    int idxNzTotal = GPU_idxNzTotalRestart[row];
    
    // To align the access of shared memory, the ordering of different types of variable should be 
    // 1) double (multiple of 8 bytes) 2) float / int (multiple of 4 bytes) 3) char (multiple of 1 bytes)
    T_real *shValIter    = shBuff;                         // blockDim.x  T_reals  in shared memory
    int    *shColIter    = (int*)&shValIter[blockDim.x];   // blockDim.x  ints     in shared memory
    int    *shNnzIter    = &shColIter[blockDim.x];         // 1           int      in shared memory
    
    if (threadIdx.x == 0) {
        *shNnzIter    = 0;
    }
    
    __syncthreads();
    
    while (col < maxCol) {
        if (col < nbPoints) {
            #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[threadIdx.x] = 1.0f;
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[threadIdx.x] = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma)); // possible underflow of shValIter
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                T_real sim = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
                shColIter[threadIdx.x] = -1;
                if (sim > tholdSim && row != col) {
                    shValIter[threadIdx.x] = sim;
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
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
                shColIter[threadIdx.x] = -1;
                if (sqSim > tholdSim*tholdSim && row != col) {
                    shValIter[threadIdx.x] = SQRT(sqSim);
                    shColIter[threadIdx.x] = col;
                    atomicAdd_block(shNnzIter, 1);
                }
            #endif

        } // End of while
        
        __syncthreads();
        
        if (*shNnzIter > 0 && threadIdx.x == 0) {
            for (int i = 0; i < blockDim.x && col + i < nbPoints; i++) {
                // if (shValIter[i] > 0.0f) { // Error risk due to possible underflow of shValIter[i]
                if (shColIter[i] != -1) {
                    GPU_csrVal[idxNzTotal] = shValIter[i];
                    GPU_csrCol[idxNzTotal] = col + i;  // shColIter[i]
                    idxNzTotal++;
                }
            }
            *shNnzIter = 0;
        }
        
        __syncthreads();
        
        col += blockDim.x;
    }   // end of while
}


__global__ void kernel_supplementary_pass_CSR2_1D_grid_2D_blocks (int nbPoints, int nbDims, 
                                                                  T_real *GPU_dataT, int *GPU_csrRow,
                                                                  int *GPU_idxNzTotalRestart, int *GPU_colRestart,
                                                                  T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                                  T_real *GPU_csrVal, int *GPU_csrCol)
{
    // 2D block, 1D grid in x-axis but regarded as in y-axis
    int row = blockDim.y * blockIdx.x + threadIdx.y;
    int yofs = threadIdx.y*blockDim.x;
    int col, colRestart, maxCol, idxNzTotal;
    if (row < nbPoints) {
        colRestart = GPU_colRestart[row];
        col        = colRestart + threadIdx.x;
        maxCol     = colRestart + ((nbPoints - colRestart - 1)/blockDim.x + 1) * blockDim.x;
        idxNzTotal = GPU_idxNzTotalRestart[row];
    }
    
    // To align the access of shared memory, the ordering of different types of variable should be 
    // 1) double (multiple of 8 bytes) 2) float / int (multiple of 4 bytes) 3) char (multiple of 1 bytes)
    T_real *shValIter    = shBuff;                                  // blockDim.y*blockDim.x  T_reals  in shared memory
    int    *shColIter    = (int*)&shValIter[blockDim.y*blockDim.x]; // blockDim.y*blockDim.x  ints     in shared memory
    int    *shNnzIter    = &shColIter[blockDim.y*blockDim.x];       // blockDim.y             int      in shared memory
    
    if (threadIdx.x == 0) {
        shNnzIter[threadIdx.y] = 0;
    }
    
    __syncthreads();
    
    while (col < maxCol && row < nbPoints) {
        if (col < nbPoints) {
            #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[yofs + threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[yofs + threadIdx.x] = 1.0f;
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                shColIter[yofs + threadIdx.x] = -1;
                if (sqDist < tholdSqDist && row != col) {
                    shValIter[yofs + threadIdx.x] = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma)); // possible underflow of shValIter
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
            #endif
            
            #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
                T_real diff, sqDist = 0.0f;
                for (int j = 0; j < nbDims; j++) {
                    index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                    diff = GPU_dataT[idxOffset + (index_t)row] - GPU_dataT[idxOffset + (index_t)col];
                    sqDist += diff*diff;
                }
                T_real sim = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
                shColIter[yofs + threadIdx.x] = -1;
                if (sim > tholdSim && row != col) {
                    shValIter[yofs + threadIdx.x] = sim;
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
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
                shColIter[yofs + threadIdx.x] = -1;
                if (sqSim > tholdSim*tholdSim && row != col) {
                    shValIter[yofs + threadIdx.x] = SQRT(sqSim);
                    shColIter[yofs + threadIdx.x] = col;
                    atomicAdd(&shNnzIter[threadIdx.y], 1);
                }
            #endif

        } // End of while
        
        __syncthreads();
        
        if (shNnzIter[threadIdx.y] > 0 && threadIdx.x == 0) {
            for (int i = 0; i < blockDim.x && col + i < nbPoints; i++) {
                // if (shValIter[yofs + i] > 0.0f) { // Error risk due to possible underflow of shValIter[yofs + i]
                if (shColIter[yofs + i] != -1) {
                    GPU_csrVal[idxNzTotal] = shValIter[yofs + i];
                    GPU_csrCol[idxNzTotal] = col + i;  // shColIter[yofs + i]
                    idxNzTotal++;
                }
            }
            shNnzIter[threadIdx.y] = 0;
        }
        
        __syncthreads();
        
        col += blockDim.x;
    }   // end of while
}



// Using 1D grids with 1D blocks
/*
void algo_CSR1_for_similarity_matrix_construction (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                   int *GPU_nnzPerRowS, int *minNnzRowS, int *maxNnzRowS, int *nnzS,
                                                   T_real **GPU_csrValS, int **GPU_csrColS, int *GPU_csrRowS)
{
    float elapsed;
    dim3 Dg, Db;
    size_t shMemSize;
    
    // Count nnz per row with 1D grid
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints;
    Dg.y = 1;
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    kernel_first_pass_CSR1_1D_grid_1D_blocks<<<Dg,Db>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                        sigma, tholdSim, tholdSqDist,        // input
                                                        GPU_nnzPerRowS);              // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
    CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
    printf("    kernel_first_pass_CSR1_1D_grid_1D_blocks: %f ms\n", elapsed);
    
    // Find minimal and maximal nnz in a row and their corresponding row numbers
    // "The thrust::minmax_element is potentially more efficient than separate calls to thrust::min_element and thrust::max_element."
    thrust::device_ptr<int> d_nnzPerRowS(GPU_nnzPerRowS);
    thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<int>> extrema = thrust::minmax_element(thrust::device, d_nnzPerRowS, d_nnzPerRowS + nbPoints);
    int idxMinNnzRowS = extrema.first - d_nnzPerRowS;
    int idxMaxNnzRowS = extrema.second - d_nnzPerRowS;
    *minNnzRowS = *extrema.first;
    *maxNnzRowS = *extrema.second;
    printf("    Min nnz in one row: %d (at row %d)\n", *minNnzRowS, idxMinNnzRowS);
    printf("    Max nnz in one row: %d (at row %d)\n", *maxNnzRowS, idxMaxNnzRowS);
    
    // Compute GPU_csrRowS by an exclusive scan on GPU_nnzPerRowS
    thrust::device_ptr<int> d_csrRowS(GPU_csrRowS);
    thrust::exclusive_scan(thrust::device, d_nnzPerRowS, d_nnzPerRowS + nbPoints + 1, d_csrRowS);
    
    // Get the total nnz of similarity matrix
    (*nnzS) = d_csrRowS[nbPoints];
    if ((*nnzS) > 0) {
        printf("    Average nnz per row: %d\n", (*nnzS)/nbPoints);
        printf("    Total nnz:           %d\n", (*nnzS));
        printf("    Sparsity:            %.3lf%%\n", 100 - ((((double)(*nnzS)/nbPoints)*100)/nbPoints));
    } else {
        printf("Total number of nonzeros/edges exceeds the limit of nvGRAPH (%d), leading to integer overflow !\n", INT_MAX);
        exit(EXIT_FAILURE);
    }
    
    // Memory allocation for csrVal & csrCol
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_csrColS, sizeof(int)*(*nnzS)));     // Note that "(void**) GPU_csrColS" instead of "(void**) &GPU_csrColS"
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_csrValS, sizeof(T_real)*(*nnzS)));  // Note that "(void**) GPU_csrValS" instead of "(void**) &GPU_csrValS"
    
    // Construct csrVal & csrCol by a complete second pass
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints;
    Dg.y = 1;
    shMemSize = sizeof(T_real)*Db.x +      // T_real shValIter[blockDim.x]
             sizeof(int)*Db.x +         // int    shColIter[blockDim.x]
             sizeof(int);               // int    shNnzIter
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_second_pass_CSR1_1D_grid_1D_blocks needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    kernel_second_pass_CSR1_1D_grid_1D_blocks<<<Dg, Db, shMemSize>>>(nbPoints, nbDims,              // input
                                                                     GPU_dataT, GPU_csrRowS,        // input
                                                                     sigma, tholdSim, tholdSqDist,  // input
                                                                     *GPU_csrValS, *GPU_csrColS);   // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
    CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
    printf("    kernel_second_pass_CSR1_1D_grid_1D_blocks: %f ms\n", elapsed);
        
        // Save CSR representation of similarity matrix
        // T_real *csrValS;
        // int *csrRowS;
        // int *csrColS;
        // csrValS = (T_real *) malloc(sizeof(T_real)*(*nnzS));
        // csrRowS = (int *) malloc(sizeof(int)*(nbPoints + 1));
        // csrColS = (int *) malloc(sizeof(int)*(*nnzS));
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrValS, *GPU_csrValS, sizeof(T_real)*(*nnzS), cudaMemcpyDeviceToHost)); 
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrRowS, GPU_csrRowS, sizeof(int)*(nbPoints + 1), cudaMemcpyDeviceToHost)); 
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrColS, *GPU_csrColS, sizeof(int)*(*nnzS), cudaMemcpyDeviceToHost)); 
        // save_file_real(csrValS, (*nnzS),      1, "output/csrValS.txt", "", 0);
        // save_file_int (csrRowS, nbPoints + 1, 1, "output/csrRowS.txt", "", 0);
        // save_file_int (csrColS, (*nnzS),      1, "output/csrColS.txt", "", 0);
        // free(csrValS);
        // free(csrRowS);
        // free(csrColS);
}
*/


// Using 1D/2D grids with 2D blocks

void algo_CSR1_for_similarity_matrix_construction (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                   int *GPU_nnzPerRowS, int *minNnzRowS, int *maxNnzRowS, int *nnzS,
                                                   T_real **GPU_csrValS, int **GPU_csrColS, int *GPU_csrRowS)
{
    float elapsed;
    dim3 Dg, Db;
    size_t shMemSize;
    
    // Count nnz per row
    float Tcde_kernel = 0.0f;
    Db.x = BsXK1;
    Db.y = BsYK1;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = nbPoints/Db.y + (nbPoints%Db.y > 0 ? 1 : 0);
    if (BsXK1*BsYK1 > 1024) {
        printf("<-bsxk1>*<-bsyk1> should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    shMemSize = sizeof(int)*Db.y;   // int shNnz[blockDim.y]
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("    The kernel_first_pass_CSR1_2D_grid_2D_blocks needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_nnzPerRowS, 0, sizeof(int)*(nbPoints + 1)));
    #define DG_Y_LIMIT  65535
    int nbRowsPerGrid = (Dg.y > DG_Y_LIMIT ? Db.y*DG_Y_LIMIT : nbPoints);
    int nbGrids = nbPoints/nbRowsPerGrid + (nbPoints%nbRowsPerGrid > 0 ? 1 : 0);
    cudaStream_t streams[2];
    for(int i = 0; i < 2; i++) {
        CHECK_CUDA_SUCCESS(cudaStreamCreate(&streams[i]));
    }
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    for (int i = 0; i < nbGrids; i++) {
        int offset = i * nbRowsPerGrid;
        int length = (i == nbGrids - 1 ? nbPoints - offset : nbRowsPerGrid);
        // printf("    seg_num = %d, offset = %d, length = %d\n", i, offset, length);
        Dg.y = length/Db.y + (length%Db.y > 0 ? 1 : 0);
        kernel_first_pass_CSR1_2D_grid_2D_blocks<<<Dg, Db, shMemSize, streams[i%2]>>>(nbPoints, nbDims, GPU_dataT,  // input
                                                                                      offset, length,               // input
                                                                                      sigma, tholdSim, tholdSqDist, // input
                                                                                      GPU_nnzPerRowS);              // output
    }
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
    CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
    Tcde_kernel += elapsed;
    for(int i = 0; i < 2; i++) {
        CHECK_CUDA_SUCCESS(cudaStreamDestroy(streams[i]));
    }
    printf("    kernel_first_pass_CSR1_2D_grid_2D_blocks: %.2f s\n", Tcde_kernel/1.0E3f);
    
    // Find minimal and maximal nnz in a row and their corresponding row numbers
    // "The thrust::minmax_element is potentially more efficient than separate calls to thrust::min_element and thrust::max_element."
    thrust::device_ptr<int> d_nnzPerRowS(GPU_nnzPerRowS);
    thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<int>> extrema = thrust::minmax_element(thrust::device, d_nnzPerRowS, d_nnzPerRowS + nbPoints);
    int idxMinNnzRowS = extrema.first - d_nnzPerRowS;
    int idxMaxNnzRowS = extrema.second - d_nnzPerRowS;
    *minNnzRowS = *extrema.first;
    *maxNnzRowS = *extrema.second;
    printf("    Min nnz in one row: %d (at row %d)\n", *minNnzRowS, idxMinNnzRowS);
    printf("    Max nnz in one row: %d (at row %d)\n", *maxNnzRowS, idxMaxNnzRowS);
    
    // Compute GPU_csrRowS by an exclusive scan on GPU_nnzPerRowS
    thrust::device_ptr<int> d_csrRowS(GPU_csrRowS);
    thrust::exclusive_scan(thrust::device, d_nnzPerRowS, d_nnzPerRowS + nbPoints + 1, d_csrRowS);
    
    // Get the total nnz of similarity matrix
    (*nnzS) = d_csrRowS[nbPoints];
    if ((*nnzS) > 0) {
        printf("    Average nnz per row: %d\n", (*nnzS)/nbPoints);
        printf("    Total nnz:           %d\n", (*nnzS));
        printf("    Sparsity:            %.3lf%%\n", 100 - ((((double)(*nnzS)/nbPoints)*100)/nbPoints));
    } else {
        printf("Total number of nonzeros/edges exceeds the limit of nvGRAPH (%d), leading to integer overflow !\n", INT_MAX);
        exit(EXIT_FAILURE);
    }
    
    // Memory allocation for csrVal & csrCol
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_csrColS, sizeof(int)*(*nnzS)));     // Note that "(void**) GPU_csrColS" instead of "(void**) &GPU_csrColS"
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_csrValS, sizeof(T_real)*(*nnzS)));  // Note that "(void**) GPU_csrValS" instead of "(void**) &GPU_csrValS"
    
    // Construct csrVal & csrCol by a complete second pass
    Db.x = BsXK2;
    Db.y = BsYK2;
    Dg.x = nbPoints/Db.y + (nbPoints%Db.y > 0 ? 1 : 0);
    Dg.y = 1;
    if (BsXK2*BsYK2 > 1024) {
        printf("<-bsxk2>*<-bsyk2> should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    shMemSize = sizeof(T_real)*Db.y*Db.x +   // T_real shValIter[blockDim.y*blockDim.x]
                sizeof(int)*Db.y*Db.x +      // int    shColIter[blockDim.y*blockDim.x]
                sizeof(int)*Db.y;            // int    shNnzIter[blockDim.y]
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_second_pass_CSR1_1D_grid_2D_blocks needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    kernel_second_pass_CSR1_1D_grid_2D_blocks<<<Dg, Db, shMemSize>>>(nbPoints, nbDims,              // input
                                                                     GPU_dataT, GPU_csrRowS,        // input
                                                                     sigma, tholdSim, tholdSqDist,  // input
                                                                     *GPU_csrValS, *GPU_csrColS);   // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
    CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
    printf("    kernel_second_pass_CSR1_1D_grid_2D_blocks: %.2f s\n", elapsed/1.0E3f);
    
        
        // Save CSR representation of similarity matrix
        // T_real *csrValS;
        // int *csrRowS;
        // int *csrColS;
        // csrValS = (T_real *) malloc(sizeof(T_real)*(*nnzS));
        // csrRowS = (int *) malloc(sizeof(int)*(nbPoints + 1));
        // csrColS = (int *) malloc(sizeof(int)*(*nnzS));
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrValS, *GPU_csrValS, sizeof(T_real)*(*nnzS), cudaMemcpyDeviceToHost)); 
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrRowS, GPU_csrRowS, sizeof(int)*(nbPoints + 1), cudaMemcpyDeviceToHost)); 
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrColS, *GPU_csrColS, sizeof(int)*(*nnzS), cudaMemcpyDeviceToHost)); 
        // save_file_real(csrValS, (*nnzS),      1, "output/csrValS.txt", "", 0);
        // save_file_int (csrRowS, nbPoints + 1, 1, "output/csrRowS.txt", "", 0);
        // save_file_int (csrColS, (*nnzS),      1, "output/csrColS.txt", "", 0);
        // free(csrValS);
        // free(csrRowS);
        // free(csrColS);
}



// Using 1D grids with 1D blocks
/*
void algo_CSR2_for_similarity_matrix_construction (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                   int hypoMaxNnzRow, int pad1, int pad2,
                                                   int *GPU_nnzPerRowS, int *minNnzRowS, int *maxNnzRowS, int *nnzS,
                                                   T_real **GPU_csrValS, int **GPU_csrColS, int *GPU_csrRowS)
{
    float elapsed;
    dim3 Dg, Db;
    size_t shMemSize;
    int    *GPU_idxNzRowRestart;  // Array for nnz offset per row (multiple of 32) of similarity matrix
    int    *GPU_colRestart;       // Array for column offset per row (multiple of 32) of similarity matrix
    T_real *GPU_csrValMaxS;       // Array for csrValMax of similarity matrix
    int    *GPU_csrColMaxS;       // Array for csrColMax of similarity matrix
    
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_idxNzRowRestart, sizeof(int)*nbPoints));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_colRestart, sizeof(int)*nbPoints));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_csrColMaxS, (sizeof(int)*hypoMaxNnzRow)*nbPoints));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_csrValMaxS, (sizeof(T_real)*hypoMaxNnzRow)*nbPoints));

    // Count nnz per row & construct csrValMax and csrColMax
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints;
    Dg.y = 1;
    shMemSize = sizeof(T_real)*(Db.x + pad1) +           // T_real shValIter[blockDim.x + pad1]
                sizeof(T_real)*(hypoMaxNnzRow + pad2) +  // T_real shNzValMax[hypoMaxNnzRow + pad2]
                sizeof(int)*Db.x +                       // int    shColIter[blockDim.x]
                sizeof(int)*hypoMaxNnzRow +              // int    shNzColMax[hypoMaxNnzRow]
                sizeof(int)*3;                           // int    shNnzIter, shNnzRow, shIdxNzRowRestart
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_full_pass_CSR2_1D_grid_1D_blocks needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    kernel_full_pass_CSR2_1D_grid_1D_blocks<<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,                     // input
                                                                   sigma, tholdSim, tholdSqDist,                    // input
                                                                   hypoMaxNnzRow, pad1, pad2,                       // input
                                                                   GPU_idxNzRowRestart, GPU_colRestart,             // output
                                                                   GPU_nnzPerRowS, GPU_csrValMaxS, GPU_csrColMaxS); // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
    CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
    printf("    kernel_full_pass_CSR2_1D_grid_1D_blocks: %f ms\n", elapsed);
    
    // Find minimal and maximal nnz in a row and their corresponding row numbers
    // "The thrust::minmax_element is potentially more efficient than separate calls to thrust::min_element and thrust::max_element."
    thrust::device_ptr<int> d_nnzPerRowS(GPU_nnzPerRowS);
    thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<int>> extrema = thrust::minmax_element(thrust::device, d_nnzPerRowS, d_nnzPerRowS + nbPoints);
    int idxMinNnzRowS = extrema.first - d_nnzPerRowS;
    int idxMaxNnzRowS = extrema.second - d_nnzPerRowS;
    *minNnzRowS = *extrema.first;
    *maxNnzRowS = *extrema.second;
    printf("    Min nnz in one row:  %d (at row %d) %s Hypothesis (%d)\n", *minNnzRowS, idxMinNnzRowS, (*minNnzRowS <= hypoMaxNnzRow)? "<=" : ">", hypoMaxNnzRow);
    printf("    Max nnz in one row:  %d (at row %d) %s Hypothesis (%d)\n", *maxNnzRowS, idxMaxNnzRowS, (*maxNnzRowS <= hypoMaxNnzRow)? "<=" : ">", hypoMaxNnzRow);
    
    // Compute GPU_csrRowS by an exclusive scan on GPU_nnzPerRowS
    thrust::device_ptr<int> d_csrRowS(GPU_csrRowS);
    thrust::exclusive_scan(thrust::device, d_nnzPerRowS, d_nnzPerRowS + nbPoints + 1, d_csrRowS);
    
    // Get the total nnz of similarity matrix
    (*nnzS) = d_csrRowS[nbPoints];
    if ((*nnzS) > 0) {
        printf("    Average nnz per row: %d\n", (*nnzS)/nbPoints);
        printf("    Total nnz:           %d\n", (*nnzS));
        printf("    Sparsity:            %.3lf%%\n", 100 - ((((double)(*nnzS)/nbPoints)*100)/nbPoints));
    } else {
        printf("Total number of nonzeros/edges exceeds the limit of nvGRAPH (%d), leading to integer overflow !\n", INT_MAX);
        exit(EXIT_FAILURE);
    }
    
    // Memory allocation for csrVal & csrCol
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_csrColS, sizeof(int)*(*nnzS)));     // Note that "(void**) GPU_csrColS" instead of "(void**) &GPU_csrColS"
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_csrValS, sizeof(T_real)*(*nnzS)));  // Note that "(void**) GPU_csrValS" instead of "(void**) &GPU_csrValS"
    
    // Fill csrVal & csrCol with obtained csrValMax & csrColMax
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints;
    Dg.y = 1;
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    kernel_ellpack_to_csr_CSR2_1D_grid_1D_blocks<<<Dg,Db>>>(GPU_csrRowS, GPU_idxNzRowRestart,  // input
                                                            hypoMaxNnzRow,                     // input
                                                            GPU_csrValMaxS, GPU_csrColMaxS,    // input
                                                            *GPU_csrValS, *GPU_csrColS,        // output
                                                            GPU_idxNzRowRestart);              // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
    CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
    printf("    kernel_ellpack_to_csr_CSR2_1D_grid_1D_blocks: %f ms\n", elapsed);

    if (*maxNnzRowS > hypoMaxNnzRow) {
        // Initialization
        Db.x = BsXN;
        Db.y = 1;
        Dg.x = nbPoints;
        Dg.y = 1;
        shMemSize = sizeof(T_real)*Db.x +       // T_real shValIter[blockDim.x]
                    // sizeof(T_real)*Db.x +    // T_real shNzVal[blockDim.x]
                    // sizeof(int)*Db.x +       // int    shNzCol[blockDim.x]
                    sizeof(int)*Db.x +          // int    shColIter[blockDim.x]
                    sizeof(int);                // int    shNnzIter
        if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
            printf("The kernel_supplementary_pass_CSR2_1D_grid_1D_blocks needs too much shared memory per block (%lu bytes)!\n", shMemSize);
            exit(EXIT_FAILURE);
        }
        
        // Complete csrVal & csrCol by an additional pass
        CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
        kernel_supplementary_pass_CSR2_1D_grid_1D_blocks<<<Dg, Db, shMemSize>>>(nbPoints, nbDims,                    // input
                                                                                GPU_dataT, GPU_csrRowS,              // input
                                                                                GPU_idxNzRowRestart, GPU_colRestart, // input
                                                                                sigma, tholdSim, tholdSqDist,        // input
                                                                                *GPU_csrValS, *GPU_csrColS);         // output
        CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
        CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
        CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
        CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
        printf("    kernel_supplementary_pass_CSR2_1D_grid_1D_blocks: %f ms\n", elapsed);
    }
    
    CHECK_CUDA_SUCCESS(cudaFree(GPU_idxNzRowRestart));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_colRestart));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_csrValMaxS));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_csrColMaxS));

        // Save CSR representation of similarity matrix
        // T_real *csrValS;
        // int *csrRowS;
        // int *csrColS;
        // csrValS = (T_real *) malloc(sizeof(T_real)*(*nnzS));
        // csrRowS = (int *) malloc(sizeof(int)*(nbPoints + 1));
        // csrColS = (int *) malloc(sizeof(int)*(*nnzS));
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrValS, *GPU_csrValS, sizeof(T_real)*(*nnzS), cudaMemcpyDeviceToHost)); 
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrRowS, GPU_csrRowS, sizeof(int)*(nbPoints + 1), cudaMemcpyDeviceToHost)); 
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrColS, *GPU_csrColS, sizeof(int)*(*nnzS), cudaMemcpyDeviceToHost)); 
        // save_file_real(csrValS, (*nnzS),      1, "output/csrValS.txt", "", 0);
        // save_file_int (csrRowS, nbPoints + 1, 1, "output/csrRowS.txt", "", 0);
        // save_file_int (csrColS, (*nnzS),      1, "output/csrColS.txt", "", 0);
        // free(csrValS);
        // free(csrRowS);
        // free(csrColS);
}
*/


// Using 1D grids with 2D blocks
void algo_CSR2_for_similarity_matrix_construction (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                   int hypoMaxNnzRow, int pad1, int pad2,
                                                   int *GPU_nnzPerRowS, int *minNnzRowS, int *maxNnzRowS, int *nnzS,
                                                   T_real **GPU_csrValS, int **GPU_csrColS, int *GPU_csrRowS)
{
    float elapsed;
    dim3 Dg, Db;
    size_t shMemSize;
    int    *GPU_idxNzRowRestart;  // Array for nnz offset per row (multiple of 32) of similarity matrix
    int    *GPU_colRestart;       // Array for column offset per row (multiple of 32) of similarity matrix
    T_real *GPU_csrValMaxS;       // Array for csrValMax of similarity matrix
    int    *GPU_csrColMaxS;       // Array for csrColMax of similarity matrix
    
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_idxNzRowRestart, sizeof(int)*nbPoints));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_colRestart, sizeof(int)*nbPoints));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_csrColMaxS, (sizeof(int)*hypoMaxNnzRow)*nbPoints));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_csrValMaxS, (sizeof(T_real)*hypoMaxNnzRow)*nbPoints));

    // Count nnz per row & construct csrValMax and csrColMax
    Db.x = BsXK3;
    Db.y = BsYK3;
    Dg.x = nbPoints/Db.y + (nbPoints%Db.y > 0 ? 1 : 0);
    Dg.y = 1;
    if (BsXK3*BsYK3 > 1024) {
        printf("<-bsxk3>*<-bsyk3> should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    shMemSize = sizeof(T_real)*Db.y*Db.x +           // T_real shValIter[blockDim.y*blockDim.x]
                sizeof(T_real)*Db.y*hypoMaxNnzRow +  // T_real shNzValMax[blockDim.y*hypoMaxNnzRow]
                sizeof(int)*Db.y*Db.x +              // int    shColIter[blockDim.y*blockDim.x]
                sizeof(int)*Db.y*hypoMaxNnzRow +     // int    shNzColMax[blockDim.y*hypoMaxNnzRow]
                sizeof(int)*Db.y*3;                  // int    shNnzIter[blockDim.y], shNnzRow[blockDim.y], shIdxNzRowRestart[blockDim.y]
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_full_pass_CSR2_1D_grid_1D_blocks needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    kernel_full_pass_CSR2_1D_grid_2D_blocks<<<Dg, Db, shMemSize>>>(nbPoints, nbDims, GPU_dataT,                     // input
                                                                   sigma, tholdSim, tholdSqDist,                    // input
                                                                   hypoMaxNnzRow,                                   // input
                                                                   GPU_idxNzRowRestart, GPU_colRestart,             // output
                                                                   GPU_nnzPerRowS, GPU_csrValMaxS, GPU_csrColMaxS); // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
    CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
    printf("    kernel_full_pass_CSR2_1D_grid_2D_blocks: %.2f s\n", elapsed/1.0E3f);
    
    // Find minimal and maximal nnz in a row and their corresponding row numbers
    // "The thrust::minmax_element is potentially more efficient than separate calls to thrust::min_element and thrust::max_element."
    thrust::device_ptr<int> d_nnzPerRowS(GPU_nnzPerRowS);
    thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<int>> extrema = thrust::minmax_element(thrust::device, d_nnzPerRowS, d_nnzPerRowS + nbPoints);
    int idxMinNnzRowS = extrema.first - d_nnzPerRowS;
    int idxMaxNnzRowS = extrema.second - d_nnzPerRowS;
    *minNnzRowS = *extrema.first;
    *maxNnzRowS = *extrema.second;
    printf("    Min nnz in one row:  %d (at row %d) %s Hypothesis (%d)\n", *minNnzRowS, idxMinNnzRowS, (*minNnzRowS <= hypoMaxNnzRow)? "<=" : ">", hypoMaxNnzRow);
    printf("    Max nnz in one row:  %d (at row %d) %s Hypothesis (%d)\n", *maxNnzRowS, idxMaxNnzRowS, (*maxNnzRowS <= hypoMaxNnzRow)? "<=" : ">", hypoMaxNnzRow);
    
    // Compute GPU_csrRowS by an exclusive scan on GPU_nnzPerRowS
    thrust::device_ptr<int> d_csrRowS(GPU_csrRowS);
    thrust::exclusive_scan(thrust::device, d_nnzPerRowS, d_nnzPerRowS + nbPoints + 1, d_csrRowS);
    
    // Get the total nnz of similarity matrix
    (*nnzS) = d_csrRowS[nbPoints];
    if ((*nnzS) > 0) {
        printf("    Average nnz per row: %d\n", (*nnzS)/nbPoints);
        printf("    Total nnz:           %d\n", (*nnzS));
        printf("    Sparsity:            %.3lf%%\n", 100 - ((((double)(*nnzS)/nbPoints)*100)/nbPoints));
    } else {
        printf("Total number of nonzeros/edges exceeds the limit of nvGRAPH (%d), leading to integer overflow !\n", INT_MAX);
        exit(EXIT_FAILURE);
    }
    
    // Memory allocation for csrVal & csrCol
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_csrColS, sizeof(int)*(*nnzS)));     // Note that "(void**) GPU_csrColS" instead of "(void**) &GPU_csrColS"
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_csrValS, sizeof(T_real)*(*nnzS)));  // Note that "(void**) GPU_csrValS" instead of "(void**) &GPU_csrValS"
    
    // Fill csrVal & csrCol with obtained csrValMax & csrColMax
    Db.x = BsXK4;
    Db.y = BsYK4;
    Dg.x = nbPoints/Db.y + (nbPoints%Db.y > 0 ? 1 : 0);
    Dg.y = 1;
    if (BsXK4*BsYK4 > 1024) {
        printf("<-bsxk4>*<-bsyk4> should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    kernel_ellpack_to_csr_CSR2_1D_grid_2D_blocks<<<Dg,Db>>>(GPU_csrRowS, GPU_idxNzRowRestart,  // input
                                                            hypoMaxNnzRow, nbPoints,           // input
                                                            GPU_csrValMaxS, GPU_csrColMaxS,    // input
                                                            *GPU_csrValS, *GPU_csrColS,        // output
                                                            GPU_idxNzRowRestart);              // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
    CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
    printf("    kernel_ellpack_to_csr_CSR2_1D_grid_2D_blocks: %.2f s\n", elapsed/1.0E3f);
    
    if (*maxNnzRowS > hypoMaxNnzRow) {
        // Complete csrVal & csrCol by a supplementary pass
        Db.x = BsXK5;
        Db.y = BsYK5;
        Dg.x = nbPoints/Db.y + (nbPoints%Db.y > 0 ? 1 : 0);
        Dg.y = 1;
        if (BsXK5*BsYK5 > 1024) {
            printf("<-bsxk5>*<-bsyk5> should not exceed 1024!\n");
            exit(EXIT_FAILURE);
        }
        shMemSize = sizeof(T_real)*Db.y*Db.x +   // T_real shValIter[blockDim.y*blockDim.x]
                    sizeof(int)*Db.y*Db.x +      // int    shColIter[blockDim.y*blockDim.x]
                    sizeof(int)*Db.y;            // int    shNnzIter[blockDim.y]
        if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
            printf("The kernel_supplementary_pass_CSR2_1D_grid_2D_blocks needs too much shared memory per block (%lu bytes)!\n", shMemSize);
            exit(EXIT_FAILURE);
        }
        CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
        kernel_supplementary_pass_CSR2_1D_grid_2D_blocks<<<Dg, Db, shMemSize>>>(nbPoints, nbDims,                     // input
                                                                                GPU_dataT, GPU_csrRowS,               // input
                                                                                GPU_idxNzRowRestart, GPU_colRestart,  // input
                                                                                sigma, tholdSim, tholdSqDist,         // input
                                                                                *GPU_csrValS, *GPU_csrColS);          // output
        CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
        CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
        CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
        CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
        printf("    kernel_supplementary_pass_CSR2_1D_grid_2D_blocks: %.2f s\n", elapsed/1.0E3f);
    }
    
    CHECK_CUDA_SUCCESS(cudaFree(GPU_idxNzRowRestart));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_colRestart));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_csrValMaxS));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_csrColMaxS));

        // Save CSR representation of similarity matrix
        // T_real *csrValS;
        // int *csrRowS;
        // int *csrColS;
        // csrValS = (T_real *) malloc(sizeof(T_real)*(*nnzS));
        // csrRowS = (int *) malloc(sizeof(int)*(nbPoints + 1));
        // csrColS = (int *) malloc(sizeof(int)*(*nnzS));
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrValS, *GPU_csrValS, sizeof(T_real)*(*nnzS), cudaMemcpyDeviceToHost)); 
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrRowS, GPU_csrRowS, sizeof(int)*(nbPoints + 1), cudaMemcpyDeviceToHost)); 
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrColS, *GPU_csrColS, sizeof(int)*(*nnzS), cudaMemcpyDeviceToHost)); 
        // save_file_real(csrValS, (*nnzS),      1, "output/csrValS.txt", "", 0);
        // save_file_int (csrRowS, nbPoints + 1, 1, "output/csrRowS.txt", "", 0);
        // save_file_int (csrColS, (*nnzS),      1, "output/csrColS.txt", "", 0);
        // free(csrValS);
        // free(csrRowS);
        // free(csrColS);
}

