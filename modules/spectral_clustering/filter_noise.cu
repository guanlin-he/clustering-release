#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>
#include <omp.h>     // omp_get_wtime

#include <thrust/execution_policy.h>  // thrust::device, thrust::host
#include <thrust/device_vector.h>     // thrust::device_ptr
#include <thrust/extrema.h>           // thrust::minmax_element
#include <thrust/scan.h>              // thrust::exclusive_scan
#include <thrust/copy.h>              // thrust::copy_if
#include <thrust/partition.h>         // thrust::stable_partition

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/spectral_clustering/filter_noise.h"


extern __shared__ unsigned long long int shBuffLong[];

template <int BLOCK_SIZE_X>
__global__ void kernel_find_noise_based_on_scaled_score (int nbPoints, T_real tholdNoise,
                                                         T_real *GPU_scaledScore,
                                                         unsigned long long int *GPU_nbNoise,
                                                         int *GPU_isNoise, int *GPU_idxNoise,
                                                         int *GPU_labels)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned long long int *shFlagNoise = shBuffLong;  // blockDim.x  unsigned long long int in shared memory
    
    shFlagNoise[threadIdx.x] = 0;
    
    if (tid < nbPoints) {
        T_real scaledScore = GPU_scaledScore[tid];
        GPU_isNoise[tid] = 0;
        GPU_idxNoise[tid] = -1;
        if (scaledScore <= tholdNoise) {
            shFlagNoise[threadIdx.x] = 1;
            GPU_isNoise[tid] = 1;
            GPU_idxNoise[tid] = tid;
            GPU_labels[tid] = -1;
        }
    }
    
    // Count the number of noise into GPU_nbNoise: two-part reduction
    // 1 - Parallel reduction of 1D block shared array shFlagNoise[*] into shFlagNoise[0],
    //     kill useless warps step by step, only the first warp survives at the end.
    if (BLOCK_SIZE_X > 512) {
        __syncthreads();
        if (threadIdx.x < 512) {
            shFlagNoise[threadIdx.x] += shFlagNoise[threadIdx.x + 512];
        } else {
            return;
        }
    }

    if (BLOCK_SIZE_X > 256) {
        __syncthreads();
        if (threadIdx.x < 256) {
            shFlagNoise[threadIdx.x] += shFlagNoise[threadIdx.x + 256];
        } else {
            return;
        }
    }

    if (BLOCK_SIZE_X > 128) {
        __syncthreads();
        if (threadIdx.x < 128) {
            shFlagNoise[threadIdx.x] += shFlagNoise[threadIdx.x + 128];
        } else {
            return;
        }
    }

    if (BLOCK_SIZE_X > 64) {
        __syncthreads();
        if (threadIdx.x < 64) {
            shFlagNoise[threadIdx.x] += shFlagNoise[threadIdx.x + 64];
        } else {
            return;
        }
    }

    if (BLOCK_SIZE_X > 32) {
        __syncthreads();
        if (threadIdx.x < 32) {
            shFlagNoise[threadIdx.x] += shFlagNoise[threadIdx.x + 32];
        } else {
            return;
        }
    }

    if (BLOCK_SIZE_X > 16) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 16) {
            shFlagNoise[threadIdx.x] += shFlagNoise[threadIdx.x + 16];
        }
    }

    if (BLOCK_SIZE_X > 8) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 8) {
            shFlagNoise[threadIdx.x] += shFlagNoise[threadIdx.x + 8];
        }
    }

    if (BLOCK_SIZE_X > 4) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 4) {
            shFlagNoise[threadIdx.x] += shFlagNoise[threadIdx.x + 4];
        }
    }

    if (BLOCK_SIZE_X > 2) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 2) {
            shFlagNoise[threadIdx.x] += shFlagNoise[threadIdx.x + 2];
        }
    }

    if (BLOCK_SIZE_X > 1) {
        __syncwarp();            // avoid races between threads within the same warp
        if (threadIdx.x < 1) {
            shFlagNoise[threadIdx.x] += shFlagNoise[threadIdx.x + 1];
        }
    }
    
    // 2 - Final reduction into a global array
    if (threadIdx.x == 0) {
        if (shFlagNoise[0] > 0) {
            atomicAdd(GPU_nbNoise, shFlagNoise[0]);
        }
    }
}


inline void template_kernel_find_noise_based_on_scaled_score (dim3 Dg, dim3 Db, size_t shMemSize,
                                                              int nbPoints, T_real tholdNoise,
                                                              T_real *GPU_scaledScore,
                                                              unsigned long long int *GPU_nbNoise,
                                                              int *GPU_isNoise, int *GPU_idxNoise,
                                                              int *GPU_labels)
{
    switch (Db.x) {
        case 1024: kernel_find_noise_based_on_scaled_score<1024><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        case  512: kernel_find_noise_based_on_scaled_score< 512><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        case  256: kernel_find_noise_based_on_scaled_score< 256><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        case  128: kernel_find_noise_based_on_scaled_score< 128><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        case   64: kernel_find_noise_based_on_scaled_score<  64><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        case   32: kernel_find_noise_based_on_scaled_score<  32><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        case   16: kernel_find_noise_based_on_scaled_score<  16><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        case    8: kernel_find_noise_based_on_scaled_score<   8><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        case    4: kernel_find_noise_based_on_scaled_score<   4><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        case    2: kernel_find_noise_based_on_scaled_score<   2><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        case    1: kernel_find_noise_based_on_scaled_score<   1><<<Dg, Db, shMemSize>>>(nbPoints, tholdNoise,
                                                                                        GPU_scaledScore,
                                                                                        GPU_nbNoise,
                                                                                        GPU_isNoise, GPU_idxNoise,
                                                                                        GPU_labels);
                   break;
        default:   fprintf(stderr, "Unsupported value for Db.x of kernel_find_noise_based_on_scaled_score kernel!\n"); 
                   exit(EXIT_FAILURE);
    }
}


struct is_nonzero
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x != 0);
  }
};

struct is_zero
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x == 0);
  }
};

struct is_not_minus_one
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x != -1);
  }
};


__global__ void kernel_mark_noise_in_csrcol (int *GPU_csrRowS, int *GPU_labels,
                                             int nbNoise, int *GPU_nbNoiseFront, int *GPU_idxNoiseReduced,
                                             int *GPU_nnzNoise, int *GPU_csrColS, int *GPU_nnzPerRowS)
{
    // 1D block in x-axis, 1D grid in x-axis but regarded as in y-axis
    int row = blockIdx.x;
    int col = threadIdx.x;
    int nnzOffset = GPU_csrRowS[row];
    int nnzRow = GPU_csrRowS[row + 1] - nnzOffset;
    int label = GPU_labels[row];
    
    int *shNnzNoise = (int*)shBuff;
    
    if (label == -1) {
        while (col < nnzRow) {
            GPU_csrColS[nnzOffset + col] = -1;
            col += blockDim.x;
        }
        if (threadIdx.x == 0) {
            atomicAdd(GPU_nnzNoise, nnzRow);
            GPU_nnzPerRowS[row] = 0;
        }
    
    } else {
        if (threadIdx.x == 0) {
            *shNnzNoise = 0;
        }
        __syncthreads();
        while (col < nnzRow) {
            int oldColIdx = GPU_csrColS[nnzOffset + col];
            int nbNoiseFront = GPU_nbNoiseFront[oldColIdx];
            int newColIdx = oldColIdx - nbNoiseFront;
            for (int i = 0; i < nbNoise; i++) {
                if (oldColIdx == GPU_idxNoiseReduced[i]) {
                    newColIdx = -1;
                    atomicAdd_block(shNnzNoise, 1);
                }
            }
            GPU_csrColS[nnzOffset + col] = newColIdx;
            col += blockDim.x;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(GPU_nnzNoise, *shNnzNoise);
            GPU_nnzPerRowS[row] -= *shNnzNoise;
        }
    }
}


void print_histogram_and_get_threshold (int flagInteractive,
                                        int nbBinsHist, T_real widthBin,
                                        int *GPU_histogram,
                                        T_real *tholdNoise, double *timeUserResponse)
{
    // Estimate the optimal threshold for filtering noise based on the histogram
    int *histogram;
    histogram = (int *) malloc(sizeof(int)*nbBinsHist);
    CHECK_CUDA_SUCCESS(cudaMemcpy(histogram, GPU_histogram, sizeof(int)*nbBinsHist, cudaMemcpyDeviceToHost));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_histogram));
    int frequency;
    int maxFrequency = 0;
    int idxBinMaxFreq = 0;
    for (int i = 0; i < nbBinsHist; i++) {
        frequency = histogram[i];
        if (frequency > maxFrequency) {
            maxFrequency = frequency;
            idxBinMaxFreq = i;
        }
    }
    T_real ratio;
    T_real times;
    T_real maxTimes = 1.0f;
    int idxHistBestThold = 0;
    int flagExistNoise = 0;
    for (int i = idxBinMaxFreq; i > 0; i--) {
        ratio = (T_real)histogram[i] / (T_real)maxFrequency;
        times = (T_real)histogram[i] / (T_real)histogram[i - 1];
        if (times > 1.0f || ratio > 0.3f) {
            if (times > maxTimes) {
                maxTimes = times;
                idxHistBestThold = i;
            }
        } else {
            flagExistNoise = 1;
            break;
        }
    }
    T_real optTholdNoise = widthBin * idxHistBestThold;

    // Print the histogram and recommended threshold to help user determine the threshold for filtering noise
    for (int i = 0; i < nbBinsHist; i++) {
        int count = ((T_real)histogram[i] / (T_real)maxFrequency) * 20.0f;
        printf("    %.2f ~ %.2f |", widthBin*(T_real)i, widthBin*(T_real)(i+1));
        for (int j = 0; j < count; j++) {
            printf("*");
        }
        printf(" %d\n", histogram[i]);
    }
    free(histogram);
    if (flagExistNoise == 0) {
        printf("    The program did not detect any noise.\n");
        // exit(EXIT_FAILURE);
    }
    if (*tholdNoise == 0.0f) {
        if (flagInteractive) {
            printf("    Please determine the noise threshold (recommend %.2f): ", optTholdNoise);
            double begin, finish;
            begin = omp_get_wtime();
            scanf("%f", tholdNoise);
            while (*tholdNoise <= 0.0f || *tholdNoise >= 1.0f) {
                fprintf(stderr,"    Error: threshold has to be in (0.0, 1.0)!\n");
                printf("    Please enter a valid threshold: ");
                scanf("%f", tholdNoise);
            }
            finish = omp_get_wtime();
            *timeUserResponse += (finish - begin);
        } else {
            *tholdNoise = optTholdNoise;
            printf("    Auto-determined noise threshold: %f\n", *tholdNoise);
        }
    }
}


void identify_noise_and_get_noise_free_similarity_matrix_in_csr (int nbPoints, int *GPU_nnzPerRowS, int nnzS,
                                                                 int *GPU_csrRowS, int *GPU_csrColS, T_real *GPU_csrValS,
                                                                 T_real *GPU_scaledScore, T_real tholdNoise,
                                                                 int *GPU_isNoise, int *GPU_nbNoiseFront,
                                                                 int *nbPointsNF, 
                                                                 int **GPU_nnzPerRowSNF, int *nnzSNF,
                                                                 int **GPU_csrRowSNF, int **GPU_csrColSNF, T_real **GPU_csrValSNF,
                                                                 int *GPU_labels)
{
    dim3 Dg, Db;
    size_t shMemSize;
    
    // Identify noise and outliers based on tholdNoise
    int *GPU_idxNoise;
    unsigned long long int *GPU_nbNoise;
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_nbNoise, sizeof(unsigned long long int)));    
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_idxNoise, sizeof(int)*nbPoints));    
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_nbNoise, 0, sizeof(unsigned long long int)));
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    shMemSize = sizeof(unsigned long long int)*Db.x;    // unsigned long long int shFlagNoise[blockDim.x]
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_find_noise_based_on_scaled_score kernel needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    template_kernel_find_noise_based_on_scaled_score(Dg, Db, shMemSize,
                                                     nbPoints, tholdNoise,
                                                     GPU_scaledScore,
                                                     GPU_nbNoise,
                                                     GPU_isNoise, GPU_idxNoise,
                                                     GPU_labels);
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaFree(GPU_scaledScore));

    // Print nb of noise and non-noise points
    unsigned long long int nbNoise;
    CHECK_CUDA_SUCCESS(cudaMemcpy(&nbNoise, GPU_nbNoise, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_nbNoise));
    *nbPointsNF = nbPoints - (int)nbNoise;
    printf("    Nb of noise:  %llu\n", nbNoise);
    printf("    Nb of non-noise points:   %d\n", *nbPointsNF);
    
    // Mark noise as -1 in the CSR format of similarity matrix
    int *GPU_nnzNoise;
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_nnzNoise, sizeof(int)));
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_nnzNoise, 0, sizeof(int)));
    thrust::device_ptr<int> d_isNoise(GPU_isNoise);
    thrust::device_ptr<int> d_nbNoiseFront(GPU_nbNoiseFront);
    thrust::device_ptr<int> d_idxNoise(GPU_idxNoise);
    thrust::exclusive_scan(thrust::device, d_isNoise, d_isNoise + nbPoints, d_nbNoiseFront);
    thrust::stable_partition(thrust::device, d_idxNoise, d_idxNoise + nbPoints, is_not_minus_one());
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints;
    Dg.y = 1;
    kernel_mark_noise_in_csrcol<<<Dg, Db>>>(GPU_csrRowS, GPU_labels, 
                                            (int)nbNoise, GPU_nbNoiseFront, GPU_idxNoise,
                                            GPU_nnzNoise, GPU_csrColS, GPU_nnzPerRowS);
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaFree(GPU_idxNoise));


    // Print nb of noise nonzeros and non-noise nonzeros
    int nnzNoise;
    CHECK_CUDA_SUCCESS(cudaMemcpy(&nnzNoise, GPU_nnzNoise, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_nnzNoise));
    *nnzSNF = nnzS - nnzNoise;
    printf("    Nb of noise nonzeros:     %d\n", nnzNoise);
    printf("    Nb of non-noise nonzeros: %d\n", *nnzSNF);
    
    
    // Get noise-free CSR-format similarity matrix in place, i.e. reuse the memory of GPU_csrRowS, GPU_csrColS and GPU_csrValS
    thrust::device_ptr<int> d_nnzPerRowS(GPU_nnzPerRowS);
    thrust::stable_partition(thrust::device, d_nnzPerRowS, d_nnzPerRowS + nbPoints + 1, is_nonzero());
    *GPU_nnzPerRowSNF = GPU_nnzPerRowS;
    *GPU_csrRowSNF = GPU_csrRowS;
    thrust::device_ptr<int> d_csrRowSNF(*GPU_csrRowSNF);
    thrust::exclusive_scan(thrust::device, d_nnzPerRowS, d_nnzPerRowS + (*nbPointsNF) + 1, d_csrRowSNF);
    
    thrust::device_ptr<T_real> d_csrValS(GPU_csrValS);
    thrust::device_ptr<int> d_csrColS(GPU_csrColS);
    thrust::stable_partition(thrust::device, d_csrValS, d_csrValS + nnzS, d_csrColS, is_not_minus_one());
    thrust::stable_partition(thrust::device, d_csrColS, d_csrColS + nnzS, is_not_minus_one());
    *GPU_csrValSNF = GPU_csrValS;
    *GPU_csrColSNF = GPU_csrColS;
    
        // int *csrRowSNF;
        // csrRowSNF = (int *) malloc(sizeof(int)*(*nbPointsNF + 1));
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrRowSNF, *GPU_csrRowSNF, sizeof(int)*(*nbPointsNF + 1), cudaMemcpyDeviceToHost));
        // save_file_int(csrRowSNF, *nbPointsNF + 1, 1, "output/csrRowSNF.txt", "");
        // free(csrRowSNF);
    
        // int *csrColSNF;
        // csrColSNF = (int *) malloc(sizeof(int)*(*nnzSNF));
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrColSNF, *GPU_csrColSNF, sizeof(int)*(*nnzSNF), cudaMemcpyDeviceToHost));
        // save_file_int(csrColSNF, *nnzSNF, 1, "output/csrColSNF.txt", "");
        // free(csrColSNF);
        
        // T_real *csrValSNF;
        // csrValSNF = (T_real *) malloc(sizeof(T_real)*(*nnzSNF));
        // CHECK_CUDA_SUCCESS(cudaMemcpy(csrValSNF, *GPU_csrValSNF, sizeof(T_real)*(*nnzSNF), cudaMemcpyDeviceToHost));
        // save_file_real(csrValSNF, *nnzSNF, 1, "output/csrValSNF.txt", "");
        // free(csrValSNF);
}



__global__ void kernel_compute_scaled_nnz_per_row_and_histogram (int nbPoints,
                                                                 int *GPU_nnzPerRowS, 
                                                                 int minNnzRowS, int maxNnzRowS,
                                                                 int nbBinsHist, T_real widthBin,
                                                                 T_real *GPU_scaledNnzPerRowS, 
                                                                 int *GPU_histogram)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = threadIdx.x;
    
    int *shHist = (int*)shBuff;
    
    // Initialization
    while (bid < nbBinsHist) {
        shHist[bid] = 0;
        bid += blockDim.x;
    }
    
    __syncthreads();
    
    // Accumulation into local histogram in shared memory
    if (tid < nbPoints) {
        int width = maxNnzRowS - minNnzRowS;  // Nonzero checking on width has been done elsewhere beforehand.
        T_real scaledNnzPerRow = ((T_real)(GPU_nnzPerRowS[tid] - minNnzRowS)) / (T_real)width;
        GPU_scaledNnzPerRowS[tid] = scaledNnzPerRow;
        for (int i = 0; i < nbBinsHist; i++) {
            if (scaledNnzPerRow >= widthBin*i && scaledNnzPerRow < widthBin*(i + 1)) {
                atomicAdd_block(&shHist[i], 1);
            }
        }
    }
    
    __syncthreads();
    
    // Accumulation from shared memory into global memory
    bid = threadIdx.x;
    while (bid < nbBinsHist) {
        atomicAdd(&GPU_histogram[bid], shHist[bid]);
        bid += blockDim.x;
    }
}


void filter_noise_based_on_nnz_per_row (int nbPoints,
                                        int *GPU_nnzPerRowS, int minNnzRowS, int maxNnzRowS, int nnzS,
                                        int *GPU_csrRowS, int *GPU_csrColS, T_real *GPU_csrValS,
                                        int nbBinsHist, int flagInteractive, T_real tholdNoise,
                                        int *nbPointsNF, 
                                        int **GPU_nnzPerRowSNF, int *nnzSNF,
                                        int **GPU_csrRowSNF, int **GPU_csrColSNF, T_real **GPU_csrValSNF,
                                        int *GPU_isNoise, int *GPU_nbNoiseFront,
                                        int *GPU_labels, double *timeUserResponse)
{
    // Check conditions
    if (maxNnzRowS == minNnzRowS) {
        printf("    The nnz per row is equal to each other !\n");
        exit(EXIT_FAILURE);
    }
    #define MIN_NNZ_NOISE_ROW  10
    if (minNnzRowS > MIN_NNZ_NOISE_ROW) {
        printf("    Minimal nnz in one row is %d, too large for noise identification !\n", minNnzRowS);
        printf("    Need to be decreased under %d !\n", MIN_NNZ_NOISE_ROW);
        exit(EXIT_FAILURE);
    }
    
    // General variables
    dim3 Dg, Db;
    size_t shMemSize;
    
    printf("    Min nnz in a row:  %d\n", minNnzRowS);
    printf("    Max nnz in a row:  %d\n", maxNnzRowS);
    
    // Min-max scaling to obtain the scaled nnz per row in the range of [0.0, 1.0]
    // Compute the histogram on the scaled nnz per row
    T_real *GPU_scaledNnzPerRowS; // range [0.0, 1.0]
    int *GPU_histogram;
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_scaledNnzPerRowS, sizeof(T_real)*nbPoints));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_histogram, sizeof(int)*nbBinsHist));
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_histogram, 0, sizeof(int)*nbBinsHist));
    T_real widthBin = 1.0f / (T_real)nbBinsHist;
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    shMemSize = sizeof(int)*nbBinsHist;    // int shHist[nbBinsHist]
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_compute_scaled_nnz_per_row_and_histogram kernel needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    kernel_compute_scaled_nnz_per_row_and_histogram<<<Dg, Db, shMemSize>>>(nbPoints,
                                                                           GPU_nnzPerRowS,
                                                                           minNnzRowS, maxNnzRowS,
                                                                           nbBinsHist, widthBin,
                                                                           GPU_scaledNnzPerRowS,
                                                                           GPU_histogram);
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    
    
    // Estimate the optimal threshold for filtering noise based on the histogram of the scaled nnz per row
    // Print the histogram and recommended threshold to help user determine the threshold for filtering noise
    printf("    Histogram of the scaled nnz per row:\n");
    print_histogram_and_get_threshold(flagInteractive,
                                      nbBinsHist, widthBin,
                                      GPU_histogram,
                                      &tholdNoise, timeUserResponse);

    
    // Identify noise and outliers based on tholdNoise
    // Mark noise as -1 in the CSR format of similarity matrix
    // Get noise-free CSR-format similarity matrix in place, i.e. reuse the memory of GPU_csrRowS, GPU_csrColS and GPU_csrValS
    identify_noise_and_get_noise_free_similarity_matrix_in_csr(nbPoints, GPU_nnzPerRowS, nnzS,               // input
                                                               GPU_csrRowS, GPU_csrColS, GPU_csrValS,        // input
                                                               GPU_scaledNnzPerRowS, tholdNoise,             // input
                                                               GPU_isNoise, GPU_nbNoiseFront,                // output
                                                               nbPointsNF,                                   // output
                                                               GPU_nnzPerRowSNF, nnzSNF,                     // output
                                                               GPU_csrRowSNF, GPU_csrColSNF, GPU_csrValSNF,  // output
                                                               GPU_labels);                                  // output
}






template <int BLOCK_SIZE_X>
__global__ void kernel_compute_vertex_degree(int *GPU_csrRowS,
                                           T_real *GPU_csrValS,
                                           T_real *GPU_deg)
{
    // 1D block in x-axis, 1D grid in x-axis but regarded as in y-axis
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    T_real *shVal = shBuff;
    
    shVal[threadIdx.x] = 0.0f;
    
    int idxNzRowStart = GPU_csrRowS[row];
    int idxNzRowEnd   = GPU_csrRowS[row + 1];
    int nnzRow = idxNzRowEnd - idxNzRowStart;
    
    while (col < nnzRow) {
        shVal[threadIdx.x] += GPU_csrValS[idxNzRowStart + col];
        col += blockDim.x;
    }
    
    if (nnzRow > 0) {
        // Two-part reduction
        // 1 - Parallel reduction of 1D block shared array shFlagNoise[*] into shFlagNoise[0],
        //     kill useless warps step by step, only the first warp survives at the end.
        if (BLOCK_SIZE_X > 512) {
            __syncthreads();
            if (threadIdx.x < 512) {
                shVal[threadIdx.x] += shVal[threadIdx.x + 512];
            } else {
                return;
            }
        }

        if (BLOCK_SIZE_X > 256) {
            __syncthreads();
            if (threadIdx.x < 256) {
                shVal[threadIdx.x] += shVal[threadIdx.x + 256];
            } else {
                return;
            }
        }

        if (BLOCK_SIZE_X > 128) {
            __syncthreads();
            if (threadIdx.x < 128) {
                shVal[threadIdx.x] += shVal[threadIdx.x + 128];
            } else {
                return;
            }
        }

        if (BLOCK_SIZE_X > 64) {
            __syncthreads();
            if (threadIdx.x < 64) {
                shVal[threadIdx.x] += shVal[threadIdx.x + 64];
            } else {
                return;
            }
        }

        if (BLOCK_SIZE_X > 32) {
            __syncthreads();
            if (threadIdx.x < 32) {
                shVal[threadIdx.x] += shVal[threadIdx.x + 32];
            } else {
                return;
            }
        }

        if (BLOCK_SIZE_X > 16) {
            __syncwarp();            // avoid races between threads within the same warp
            if (threadIdx.x < 16) {
                shVal[threadIdx.x] += shVal[threadIdx.x + 16];
            }
        }

        if (BLOCK_SIZE_X > 8) {
            __syncwarp();            // avoid races between threads within the same warp
            if (threadIdx.x < 8) {
                shVal[threadIdx.x] += shVal[threadIdx.x + 8];
            }
        }

        if (BLOCK_SIZE_X > 4) {
            __syncwarp();            // avoid races between threads within the same warp
            if (threadIdx.x < 4) {
                shVal[threadIdx.x] += shVal[threadIdx.x + 4];
            }
        }

        if (BLOCK_SIZE_X > 2) {
            __syncwarp();            // avoid races between threads within the same warp
            if (threadIdx.x < 2) {
                shVal[threadIdx.x] += shVal[threadIdx.x + 2];
            }
        }

        if (BLOCK_SIZE_X > 1) {
            __syncwarp();            // avoid races between threads within the same warp
            if (threadIdx.x < 1) {
                shVal[threadIdx.x] += shVal[threadIdx.x + 1];
            }
        }
        
        // 2 - Final reduction into a global array
        if (threadIdx.x == 0) {
            GPU_deg[row] = shVal[0];
        }
    }
}
   

inline void template_kernel_compute_vertex_degree (dim3 Dg, dim3 Db, size_t shMemSize,
                                                 int *GPU_csrRowS, T_real *GPU_csrValS,
                                                 T_real *GPU_deg)
{
    switch (Db.x) {
        case 1024: kernel_compute_vertex_degree<1024><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        case  512: kernel_compute_vertex_degree< 512><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        case  256: kernel_compute_vertex_degree< 256><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        case  128: kernel_compute_vertex_degree< 128><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        case   64: kernel_compute_vertex_degree<  64><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        case   32: kernel_compute_vertex_degree<  32><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        case   16: kernel_compute_vertex_degree<  16><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        case    8: kernel_compute_vertex_degree<   8><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        case    4: kernel_compute_vertex_degree<   4><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        case    2: kernel_compute_vertex_degree<   2><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        case    1: kernel_compute_vertex_degree<   1><<<Dg, Db, shMemSize>>>(GPU_csrRowS,  // input
                                                                             GPU_csrValS,  // input
                                                                             GPU_deg);     // output
                   break;
        default:   fprintf(stderr, "Unsupported value for Db.x of kernel_compute_vertex_degree kernel!\n"); 
                   exit(EXIT_FAILURE);
    }
}


__global__ void kernel_compute_scaled_degree_and_histogram (int nbPoints,
                                                            T_real *GPU_deg, 
                                                            T_real minDeg, T_real maxDeg,
                                                            int nbBinsHist, T_real widthBin,
                                                            T_real *GPU_scaledDeg, 
                                                            int *GPU_histogram)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = threadIdx.x;
    
    int *shHist = (int*)shBuff;
    
    // Initialization
    while (bid < nbBinsHist) {
        shHist[bid] = 0;
        bid += blockDim.x;
    }
    
    __syncthreads();
    
    // Accumulation into local histogram in shared memory
    if (tid < nbPoints) {
        T_real width = maxDeg - minDeg;  // Nonzero checking on width has been done elsewhere beforehand.
        T_real scaledDeg = ((T_real)(GPU_deg[tid] - minDeg)) / (T_real)width;
        GPU_scaledDeg[tid] = scaledDeg;
        for (int i = 0; i < nbBinsHist; i++) {
            if (scaledDeg >= widthBin*i && scaledDeg < widthBin*(i + 1)) {
                atomicAdd_block(&shHist[i], 1);
            }
        }
    }
    
    __syncthreads();
    
    // Accumulation from shared memory into global memory
    bid = threadIdx.x;
    while (bid < nbBinsHist) {
        atomicAdd(&GPU_histogram[bid], shHist[bid]);
        bid += blockDim.x;
    }
}



void filter_noise_based_on_vertex_degree (int nbPoints,
                                          int *GPU_nnzPerRowS, int nnzS,
                                          int *GPU_csrRowS, int *GPU_csrColS, T_real *GPU_csrValS,
                                          int nbBinsHist, int flagInteractive, T_real tholdNoise,
                                          int *nbPointsNF, 
                                          int **GPU_nnzPerRowSNF, int *nnzSNF,
                                          int **GPU_csrRowSNF, int **GPU_csrColSNF, T_real **GPU_csrValSNF,
                                          int *GPU_isNoise, int *GPU_nbNoiseFront,
                                          int *GPU_labels, double *timeUserResponse)
{
    // General variables
    dim3 Dg, Db;
    size_t shMemSize;
    
    // Compute vertex degree
    T_real *GPU_deg;
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_deg, sizeof(T_real)*nbPoints));
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints;
    Dg.y = 1;
    shMemSize = sizeof(T_real)*Db.x;    // int shVal[blockDim.x]
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_compute_vertex_degree kernel needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    template_kernel_compute_vertex_degree(Dg, Db, shMemSize,
                                          GPU_csrRowS, GPU_csrValS,  // input
                                          GPU_deg);                  // output
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());


    // Find minimal and maximal vertex degree
    // "The thrust::minmax_element is potentially more efficient than separate calls to thrust::min_element and thrust::max_element."
    thrust::device_ptr<T_real> d_deg(GPU_deg);
    thrust::pair<thrust::device_ptr<T_real>, thrust::device_ptr<T_real>> extrema = thrust::minmax_element(thrust::device, d_deg, d_deg + nbPoints);
    T_real minDeg = *extrema.first;
    T_real maxDeg = *extrema.second;
    printf("    Min degree:  %f\n", minDeg);
    printf("    Max degree:  %f\n", maxDeg);
    
    
    // Min-max scaling to obtain the scaled vertex degree in the range of [0.0, 1.0]
    // Compute the histogram on the scaled vertex degree
    T_real *GPU_scaledDeg; // range [0.0, 1.0]
    int *GPU_histogram;
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_scaledDeg, sizeof(T_real)*nbPoints));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_histogram, sizeof(int)*nbBinsHist));
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_histogram, 0, sizeof(int)*nbBinsHist));
    T_real widthBin = 1.0f / (T_real)nbBinsHist;
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    shMemSize = sizeof(int)*nbBinsHist;    // int shHist[nbBinsHist]
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_compute_scaled_degree_and_histogram kernel needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    kernel_compute_scaled_degree_and_histogram<<<Dg, Db, shMemSize>>>(nbPoints,
                                                                      GPU_deg,
                                                                      minDeg, maxDeg,
                                                                      nbBinsHist, widthBin,
                                                                      GPU_scaledDeg,
                                                                      GPU_histogram);
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaFree(GPU_deg));
    
    
    // Estimate the optimal threshold for filtering noise based on the histogram of the scaled vertex degree
    // Print the histogram and recommended threshold to help user determine the threshold for filtering noise
    printf("    Histogram of the scaled vertex degree:\n");
    print_histogram_and_get_threshold(flagInteractive, 
                                      nbBinsHist, widthBin,
                                      GPU_histogram,
                                      &tholdNoise, timeUserResponse);

    
    // Identify noise and outliers based on tholdNoise
    // Mark noise as -1 in the CSR format of similarity matrix
    // Get noise-free CSR-format similarity matrix in place, i.e. reuse the memory of GPU_csrRowS, GPU_csrColS and GPU_csrValS
    identify_noise_and_get_noise_free_similarity_matrix_in_csr(nbPoints, GPU_nnzPerRowS, nnzS,               // input
                                                               GPU_csrRowS, GPU_csrColS, GPU_csrValS,        // input
                                                               GPU_scaledDeg, tholdNoise,                    // input
                                                               GPU_isNoise, GPU_nbNoiseFront,                // output
                                                               nbPointsNF, 
                                                               GPU_nnzPerRowSNF, nnzSNF,                    // output
                                                               GPU_csrRowSNF, GPU_csrColSNF, GPU_csrValSNF,  // output
                                                               GPU_labels);                                  // output
}


__global__ void kernel_get_noise_free_transposed_data_matrix (int nbDims, int nbPoints, T_real *GPU_dataT,
                                                              int *GPU_isNoise, int *GPU_nbNoiseFront,
                                                              int nbPointsNF,
                                                              T_real *GPU_dataTNF)
{
    // 2D block, 2D grid
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < nbPoints && row < nbDims) {
        int nbNoiseFront = GPU_nbNoiseFront[col];
        if (GPU_isNoise[col] == 0) {
            GPU_dataTNF[row*nbPointsNF + col - nbNoiseFront] = GPU_dataT[row*nbPoints + col];
        }
    }
}


__global__ void kernel_get_noise_free_transposed_data_matrix_using_shmem (int nbDims, int nbPoints, T_real *GPU_dataT,
                                                                          int *GPU_isNoise, int *GPU_nbNoiseFront,
                                                                          int nbPointsNF,
                                                                          T_real *GPU_dataTNF)
{
    // 2D block, 2D grid
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tidPerBlock = threadIdx.y*blockDim.x + threadIdx.x;
    T_real *shDataT = shBuff;
    T_real *shDataTNF = &shDataT[blockDim.y * blockDim.x];
    int *shNbNoiseFront = (int *)&shDataTNF[blockDim.y * blockDim.x];
    int *shNbNoisePerBlock = &shNbNoiseFront[blockDim.x];
    
    
    if (col < nbPoints && row < nbDims) {
        shDataT[tidPerBlock] = GPU_dataT[row*nbPoints + col];
        if (threadIdx.y == 0) {
            shNbNoiseFront[threadIdx.x] = GPU_nbNoiseFront[col];
        }
        if (tidPerBlock == 0) { // using "tidPerBlock == blockDim.y * blockDim.x - 1" is wrong since a thread may not satisfy "col < nbPoints && row < nbDims"
            int lastXIdxPerBlock = blockIdx.x * blockDim.x + blockDim.x - 1;
            int lastValidXIdxPerBlock = (lastXIdxPerBlock < nbPoints ? lastXIdxPerBlock : nbPoints - 1);
            *shNbNoisePerBlock = GPU_nbNoiseFront[lastValidXIdxPerBlock] - shNbNoiseFront[0];
        }
    }
    
    __syncthreads();
    
    
    if (col < nbPoints && row < nbDims) {
        int nbNoiseFrontPerBlock = shNbNoiseFront[threadIdx.x] - shNbNoiseFront[0];
        if (GPU_isNoise[col] == 0) {
            shDataTNF[tidPerBlock - nbNoiseFrontPerBlock] = shDataT[tidPerBlock];
        }
    }
    
    __syncthreads();
    
    
    if (col < nbPoints && row < nbDims) {
        int baseCol = blockIdx.x * blockDim.x;
        int validBlockLength= (baseCol + blockDim.x < nbPoints ? blockDim.x : nbPoints - baseCol);
        if (threadIdx.x < validBlockLength - *shNbNoisePerBlock) {
            GPU_dataTNF[row*nbPointsNF + col - shNbNoiseFront[0]] = shDataTNF[tidPerBlock];
        }
    }
}


void get_noise_free_transposed_data_matrix (int nbDims, int nbPoints, T_real *GPU_dataT,
                                            int *GPU_isNoise, int *GPU_nbNoiseFront,
                                            int nbPointsNF,
                                            T_real *GPU_dataTNF)
{
    dim3 Dg, Db;
    size_t shMemSize;
    
    Db.x = BsXN;
    Db.y = BSYD;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = nbDims/Db.y + (nbDims%Db.y > 0 ? 1 : 0);
    if (BsXN*BSYD > 1024) {
        printf("<-bsxn>*BSYD should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    shMemSize = sizeof(T_real)*Db.y*Db.x*2 +  // T_real shDataT[blockDim.y * blockDim.x] and shDataTNF[blockDim.y * blockDim.x]
                sizeof(int)*Db.x +            // int shNbNoiseFront[blockDim.x]
                1;                            // int shNbNoisePerBlock
    if (shMemSize > (TOTAL_SHMEM_BLOCK - SAFETY_THOLD)) {
        printf("The kernel_get_noise_free_transposed_data_matrix_using_shmem kernel needs too much shared memory per block (%lu bytes)!\n", shMemSize);
        exit(EXIT_FAILURE);
    }
    // kernel_get_noise_free_transposed_data_matrix<<<Dg, Db>>>(nbDims, nbPoints, GPU_dataT, 
                                                             // GPU_isNoise, GPU_nbNoiseFront,
                                                             // nbPointsNF,
                                                             // GPU_dataTNF);
    kernel_get_noise_free_transposed_data_matrix_using_shmem<<<Dg, Db, shMemSize>>>(nbDims, nbPoints, GPU_dataT, 
                                                                                    GPU_isNoise, GPU_nbNoiseFront,
                                                                                    nbPointsNF,
                                                                                    GPU_dataTNF);
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
}



__global__ void kernel_merge_labels (int nbPointsNF,
                                     int *GPU_labelsNF, int *GPU_nbNoiseFrontNF,
                                     int *GPU_labels)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nbPointsNF) {
        int nbNoiseFrontNF = GPU_nbNoiseFrontNF[tid];
        GPU_labels[tid + nbNoiseFrontNF] = GPU_labelsNF[tid];
    }
}


void merge_labels (int nbPoints, int nbPointsNF, 
                   int *GPU_isNoise, int *GPU_nbNoiseFront,
                   int *GPU_labelsNF,
                   int *GPU_labels)
{
    // Compute the number of noise in front of every noise-free point
    int *GPU_nbNoiseFrontNF;
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_nbNoiseFrontNF, sizeof(int)*nbPointsNF));
    thrust::device_ptr<int> d_isNoise(GPU_isNoise);
    thrust::device_ptr<int> d_nbNoiseFront(GPU_nbNoiseFront);
    thrust::device_ptr<int> d_nbNoiseFrontNF(GPU_nbNoiseFrontNF);
    thrust::copy_if(thrust::device, d_nbNoiseFront, d_nbNoiseFront + nbPoints, d_isNoise, d_nbNoiseFrontNF, is_zero());
    
    // Merge labels
    dim3 Dg, Db;
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPointsNF/Db.x + (nbPointsNF%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    kernel_merge_labels<<<Dg, Db>>>(nbPointsNF,
                                    GPU_labelsNF, GPU_nbNoiseFrontNF,
                                    GPU_labels);
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    CHECK_CUDA_SUCCESS(cudaFree(GPU_nbNoiseFrontNF)); 
}

