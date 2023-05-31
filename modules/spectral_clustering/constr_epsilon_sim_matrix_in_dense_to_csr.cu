#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <float.h>   // Library Macros (e.g. FLT_MAX, FLT_MIN)
#include <omp.h>     // omp_get_wtime
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/iterator/constant_iterator.h>   // thrust::constant_iterator
#include <thrust/device_vector.h>       // thrust::device_ptr
#include <thrust/execution_policy.h>    // thrust::host, thrust::device, thrust::cuda::par.on(stream)
#include <thrust/extrema.h>             // thrust::max_element, thrust::min_element, thrust::minmax_element
#include <thrust/scan.h>                // thrust::exclusive_scan
#include <thrust/transform.h>           // thrust::transform
#include <thrust/sequence.h>            // thrust::sequence
#include <thrust/functional.h>          // thrust::plus, thrust::greater, thrust::greater_equal
#include <iostream>
#include <iterator>

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/spectral_clustering/constr_epsilon_sim_matrix_in_dense_to_csr.h"


__global__ void kernel_construct_similarity_chunk (int chunkOffset, int chunkSize,
                                                   int nbPoints, int nbDims, T_real *GPU_dataT,
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                   T_real *GPU_sim)
{
    // 2D block, 2D grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < chunkSize && col < nbPoints) {
        #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
            T_real diff, sqDist = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                diff = GPU_dataT[idxOffset + (index_t)(row + chunkOffset)] - GPU_dataT[idxOffset + (index_t)col];
                sqDist += diff*diff;
            }
            T_real sim = 0.0f;
            if (sqDist < tholdSqDist && (row + chunkOffset != col)) {
                sim = 1.0f;
            }
            size_t idx = ((size_t)row)*((size_t)nbPoints) + ((size_t)col);      // Avoid integer overflow
            GPU_sim[idx] = sim;
        #endif
        
        #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
            T_real diff, sqDist = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                diff = GPU_dataT[idxOffset + (index_t)(row + chunkOffset)] - GPU_dataT[idxOffset + (index_t)col];
                sqDist += diff*diff;
            }
            T_real sim = 0.0f;
            if ((sqDist < tholdSqDist) && (row + chunkOffset != col)) {
                sim = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
            }
            size_t idx = ((size_t)row)*((size_t)nbPoints) + ((size_t)col);      // Avoid integer overflow
            GPU_sim[idx] = sim;
        #endif
        
        #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
            T_real diff, sqDist = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                diff = GPU_dataT[idxOffset + (index_t)(row + chunkOffset)] - GPU_dataT[idxOffset + (index_t)col];
                sqDist += diff*diff;
            }
            T_real sim = EXP((-1.0f)*sqDist/(2.0f*sigma*sigma));
            if (sim <= tholdSim || (row + chunkOffset == col)) {
                sim = 0.0f;
            }
            size_t idx = ((size_t)row)*((size_t)nbPoints) + ((size_t)col);      // Avoid integer overflow
            GPU_sim[idx] = sim;
        #endif
        
        #ifdef COS_SIM_WITH_THOLD // Cosine similarity with threshold
            T_real elm1, elm2, dot = 0.0f, sq1 = 0.0f, sq2 = 0.0f;
            for (int j = 0; j < nbDims; j++) {
                index_t idxOffset = ((index_t)j)*((index_t)nbPoints);
                elm1 = GPU_dataT[idxOffset + (index_t)(row + chunkOffset)];
                elm2 = GPU_dataT[idxOffset + (index_t)col];
                dot += elm1*elm2;
                sq1 += elm1*elm1;
                sq2 += elm2*elm2;
            }
            T_real sqSim = (dot*dot)/(sq1*sq2);
            T_real sim = 0.0f;
            if (sqSim > tholdSim*tholdSim && (row + chunkOffset != col)) {
                sim = SQRT(sqSim);
            }
            size_t idx = ((size_t)row)*((size_t)nbPoints) + ((size_t)col);      // Avoid integer overflow
            GPU_sim[idx] = sim;
        #endif
    }
}



void algo_CSR3_for_similarity_matrix_construction (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                   T_real memUsePercent, T_real maxNzPercent,
                                                   int *nnzS, T_real **GPU_csrValS, int **GPU_csrColS, int *GPU_csrRowS)
{
    // Declaration
    float elapsed;
    double begin, finish, Tomp_Init, Tomp_gpu_Dense2CSR, Tomp_gpu_CSRSort;
    float Tcde_gpu_Kernel;
    dim3 Dg, Db;
    size_t freeGPUMem, totalGPUMem, useFreeGPUMem;
    double freeGPUMemMB, freeGPUMemGB, useFreeGPUMemGB;
    size_t maxNbElms, maxNbRows;
    int nbChunks;
    int chunkOffset, chunkSize;
    int quotient, remainder;
    T_real *GPU_simChunk;
    int64_t *nnzChunk;
    T_real *GPU_csrValS_unsorted;
    size_t maxNnz;
    int nnzOffset;
    cusparseHandle_t handleCUSPARSE;

    // Initialization
    begin = omp_get_wtime();
    Tomp_Init = 0.0;
    Tcde_gpu_Kernel = 0.0f;
    Tomp_gpu_Dense2CSR = 0.0;
    Tomp_gpu_CSRSort = 0.0;
    maxNnz = (size_t)((double)nbPoints * (double)maxNzPercent / (double)100 * (double)nbPoints);
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_csrColS, sizeof(int)*maxNnz));  // Note that "(void**) GPU_csrColS" instead of "(void**) &GPU_csrColS"
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_csrValS_unsorted, sizeof(T_real)*maxNnz));
    CHECK_CUDA_SUCCESS(cudaMemGetInfo(&freeGPUMem, &totalGPUMem));   // Gets free and total device memory in bytes
    freeGPUMemMB = (double)freeGPUMem/(1024.0*1024.0);
    freeGPUMemGB = freeGPUMemMB/1024.0;
    printf("    Free GPU memory:     %.2lf GB\n", freeGPUMemGB);
    useFreeGPUMem = (size_t)((double)freeGPUMem * (double)memUsePercent / (double)100);
    useFreeGPUMemGB = (double)useFreeGPUMem/(1024.0*1024.0*1024.0);
    printf("    Mem for sim. mat.:   %.2lf GB (%.2f%% of free GPU RAM)\n", useFreeGPUMemGB, memUsePercent);
    maxNbElms = useFreeGPUMem / sizeof(T_real);
    maxNbRows = maxNbElms / ((size_t)nbPoints);
    if (maxNbRows > ((size_t)nbPoints)) {  // Auto-tuning of nbChunks
        nbChunks = 1;
    } else {
        nbChunks = nbPoints/((int)maxNbRows) + (nbPoints%((int)maxNbRows) > 0 ? 1 : 0);
    }
    quotient  = nbPoints / nbChunks;
    remainder = nbPoints % nbChunks;
    printf("    Nb of chunks:        %d\n", nbChunks);
    printf("    Average chunk size:  %d\n", quotient);
    Db.x = BsXK6;
    Db.y = BsYK6;
    if (BsXK6*BsYK6 > 1024) {
        printf("<-bsxk6>*<-bsyk6> should not exceed 1024!\n");
        exit(EXIT_FAILURE);
    }
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    thrust::device_ptr<int> d_csrRowS(GPU_csrRowS);
    CHECK_CUSPARSE_SUCCESS(cusparseCreate(&handleCUSPARSE));
    (*nnzS) = 0;
    nnzOffset = 0;
    nnzChunk = (int64_t *)malloc(sizeof(int64_t)*nbChunks);
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_simChunk, (sizeof(T_real)*(quotient + 1))*nbPoints));
    finish = omp_get_wtime();
    Tomp_Init += (finish - begin);
    
    // Chunk-wise similarity matrix construction on the GPU
    for (int c = 0; c < nbChunks; c++) {
        // Calculate chunkOffset, chunkSize, Dg.y
        chunkOffset = (c < remainder ? ((quotient + 1) * c) : (quotient * c + remainder));
        chunkSize = (c < remainder ? (quotient + 1) : quotient);
        Dg.y = chunkSize/Db.y + (chunkSize%Db.y > 0 ? 1 : 0);
        
        // Compute the similarity matrix in chunk-wise fashion
        // CHECK_CUDA_SUCCESS(cudaMemset(GPU_simChunk, 0, (sizeof(T_real)*(quotient + 1))*nbPoints));
        CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
        kernel_construct_similarity_chunk<<<Dg, Db>>>(chunkOffset, chunkSize,       // input
                                                      nbPoints, nbDims, GPU_dataT,  // input
                                                      sigma, tholdSim, tholdSqDist, // input
                                                      GPU_simChunk);                // output
        CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
        CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
        CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
        CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, StartEvent, StopEvent));
        Tcde_gpu_Kernel += elapsed;

        // Transform the chunk of similarity matrix from dense format into CSR format
        // See official samples at https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/dense2sparse_csr/
        begin = omp_get_wtime();
        
        // - Initialize the dense matrix descriptor
        cusparseDnMatDescr_t  dnMatDescrSimChunk;
        CHECK_CUSPARSE_SUCCESS(cusparseCreateDnMat(&dnMatDescrSimChunk,  // Dense matrix descriptor
                                                   chunkSize,            // Number of rows of the dense matrix
                                                   nbPoints,             // Number of columns of the dense matrix
                                                   nbPoints,             // Leading dimension of the dense matrix
                                                   GPU_simChunk,         // Values of the dense matrix
                                                   T_REAL,               // Enumerator specifying the datatype of values
                                                   CUSPARSE_ORDER_ROW)); // Enumerator specifying the memory layout of the dense matrix
        
        // - Initialize the sparse matrix descriptor in the CSR format
        cusparseSpMatDescr_t  spMatDescrSimChunk;
        CHECK_CUSPARSE_SUCCESS(cusparseCreateCsr(&spMatDescrSimChunk,       // Sparse matrix descriptor
                                                 chunkSize,                 // Number of rows of the sparse matrix
                                                 nbPoints,                  // Number of columns of the sparse matrix
                                                 0,                         // Number of non-zero entries of the sparse matrix
                                                 GPU_csrRowS + chunkOffset, // Row offsets of the sparse matrix
                                                 NULL,                      // Column indices of the sparse matrix
                                                 NULL,                      // Values of the sparse martix
                                                 CUSPARSE_INDEX_32I,        // data type of csrRowOffsets
                                                 CUSPARSE_INDEX_32I,        // data type of csrColInd
                                                 CUSPARSE_INDEX_BASE_ZERO,  // Base index of csrRowOffsets and csrColInd
                                                 T_REAL));                  // Datatype of csrValues
        
        // - Return the size of the workspace needed by cusparseDenseToSparse_analysis()
        size_t bufferSize = 0;
        void*  dBuffer    = NULL;
        CHECK_CUSPARSE_SUCCESS(cusparseDenseToSparse_bufferSize(handleCUSPARSE,                      // Handle to the cuSPARSE library context
                                                                dnMatDescrSimChunk,                  // Dense matrix
                                                                spMatDescrSimChunk,                  // Sparse matrix
                                                                CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,  // Algorithm for the computation
                                                                &bufferSize));                       // Number of bytes of workspace needed by cusparseDenseToSparse_analysis()
        CHECK_CUDA_SUCCESS(cudaMalloc(&dBuffer, bufferSize));
        
        // - Update the number of non-zero elements in the sparse matrix descriptor
        CHECK_CUSPARSE_SUCCESS(cusparseDenseToSparse_analysis(handleCUSPARSE,                      // Handle to the cuSPARSE library context
                                                              dnMatDescrSimChunk,                  // Dense matrix
                                                              spMatDescrSimChunk,                  // Sparse matrix
                                                              CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,  // Algorithm for the computation
                                                              dBuffer));                           // Pointer to workspace buffer
        
        // - Get the number of nonzero elements
        int64_t num_rows_tmp, num_cols_tmp;
        CHECK_CUSPARSE_SUCCESS(cusparseSpMatGetSize(spMatDescrSimChunk,  // Sparse matrix descriptor
                                                    &num_rows_tmp,       // Number of rows of the sparse matrix
                                                    &num_cols_tmp,       // Number of columns of the sparse matrix
                                                    &nnzChunk[c]));      // Number of non-zero entries of the sparse matrix
        
        // Update nnzOffset and (*nnzS)
        if (c > 0) {
            nnzOffset += ((int)nnzChunk[c - 1]);
        }
        (*nnzS) = nnzOffset + ((int)nnzChunk[c]);
        
        // Check whether the memory allocation for GPU_csrColS and GPU_csrValS is sufficient
        if ((size_t)(*nnzS) > maxNnz) {
            printf("    The default memory allocation for nonzeros is insufficient !\n");
            printf("    Need to increase -max-nz-percent (default: " T_REAL_PRINT ") !\n", DEFAULT_MAX_NZ_PERCENT);
            exit(EXIT_FAILURE);
        }
        
        // - Reset offsets, column indices, and values pointers
        CHECK_CUSPARSE_SUCCESS(cusparseCsrSetPointers(spMatDescrSimChunk,                 // Sparse matrix descriptor
                                                      GPU_csrRowS + chunkOffset,          // Row offsets of the sparse martix
                                                      *GPU_csrColS + nnzOffset,           // Column indices of the sparse martix
                                                      GPU_csrValS_unsorted + nnzOffset)); // Values of the sparse martix
        
        // - Execute dense-to-sparse conversion
        // - Attention !!! The results of cusparseDenseToSparse are not sorted !!! More accurately, the index arrays are sorted by row indices BUT are NOT sorted by column indices within the same row.
        // - See https://github.com/NVIDIA/CUDALibrarySamples/issues/21
        CHECK_CUSPARSE_SUCCESS(cusparseDenseToSparse_convert(handleCUSPARSE,                      // Handle to the cuSPARSE library context
                                                             dnMatDescrSimChunk,                  // Dense matrix
                                                             spMatDescrSimChunk,                  // Sparse matrix
                                                             CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,  // Algorithm for the computation
                                                             dBuffer));                           // Pointer to workspace buffer
        
        // Add nnzOffset to all values from "d_csrRowS + chunkOffset" to "d_csrRowS + chunkOffset + chunkSize"
        // See samples at https://github.com/NVIDIA/thrust/blob/master/examples/constant_iterator.cu
        // See doc at https://thrust.github.io/doc/group__transformations_gacca2dd17ae9de2f7bcbb8da6d6f6fce4.html#gacca2dd17ae9de2f7bcbb8da6d6f6fce4
        if (c < nbChunks - 1) {
            thrust::transform(thrust::device,                              // The execution policy to use for parallelization
                              d_csrRowS + chunkOffset,                     // The beginning of the first input sequence 
                              d_csrRowS + chunkOffset + chunkSize,         // The end of the first input sequence
                              thrust::constant_iterator<int>(nnzOffset),   // The beginning of the second input sequence
                              d_csrRowS + chunkOffset,                     // The beginning of the output sequence
                              thrust::plus<int>());                        // The tranformation operation
        } else {
            thrust::transform(thrust::device,
                              d_csrRowS + chunkOffset, d_csrRowS + chunkOffset + chunkSize + 1,
                              thrust::constant_iterator<int>(nnzOffset),
                              d_csrRowS + chunkOffset,
                              thrust::plus<int>());
        }
        
        // Destroy matrix descriptors
        CHECK_CUSPARSE_SUCCESS(cusparseDestroyDnMat(dnMatDescrSimChunk));
        CHECK_CUSPARSE_SUCCESS(cusparseDestroySpMat(spMatDescrSimChunk));
        
        // Memory deallocation
        CHECK_CUDA_SUCCESS(cudaFree(dBuffer));
        
        finish = omp_get_wtime();
        Tomp_gpu_Dense2CSR += (finish - begin);
    }
    
    // Memory deallocation
    CHECK_CUDA_SUCCESS(cudaFree(GPU_simChunk));
    free(nnzChunk);
    
    // Check nnz
    if ((*nnzS) > 0) {
        printf("    Average nnz per row: %d\n", (*nnzS)/nbPoints);
        printf("    Total nnz:           %d\n", (*nnzS));
        printf("    Sparsity:            %.3lf%%\n", 100 - ((((double)(*nnzS)/nbPoints)*100)/nbPoints));
    } else {
        printf("Total number of nonzeros/edges exceeds the limit of nvGRAPH (%d), leading to integer overflow !\n", INT_MAX);
        exit(EXIT_FAILURE);
    }
    
    // Sort CSR format (The stable sorting is in-place.) //
    // See doc and samples at https://docs.nvidia.com/cuda/cusparse/index.html#csrsort
    begin = omp_get_wtime();
    
    // Initialization
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;
    int *P = NULL;
    
    // Step 1: allocate buffer
    CHECK_CUSPARSE_SUCCESS(cusparseXcsrsort_bufferSizeExt(handleCUSPARSE,
                                                          nbPoints,
                                                          nbPoints,
                                                          (*nnzS),
                                                          GPU_csrRowS,
                                                          *GPU_csrColS,
                                                          &pBufferSizeInBytes));
    CHECK_CUDA_SUCCESS(cudaMalloc(&pBuffer, sizeof(char)*pBufferSizeInBytes));
    
    // Step 2: setup permutation vector P to identity
    CHECK_CUDA_SUCCESS(cudaMalloc((void**)&P, sizeof(int)*(*nnzS)));
    CHECK_CUSPARSE_SUCCESS(cusparseCreateIdentityPermutation(handleCUSPARSE, (*nnzS), P));
    
    // Step 3: sort the column indices of CSR format
    cusparseMatDescr_t  DescrSim;
    CHECK_CUSPARSE_SUCCESS(cusparseCreateMatDescr(&DescrSim));
    CHECK_CUSPARSE_SUCCESS(cusparseSetMatIndexBase(DescrSim, CUSPARSE_INDEX_BASE_ZERO)); 
    CHECK_CUSPARSE_SUCCESS(cusparseSetMatType(DescrSim, CUSPARSE_MATRIX_TYPE_GENERAL)); 
    CHECK_CUSPARSE_SUCCESS(cusparseXcsrsort(handleCUSPARSE,   // handle to the cuSPARSE library context
                                            nbPoints,       // number of rows of matrix
                                            nbPoints,       // number of columns of matrix
                                            (*nnzS),            // number of nonzero elements of matrix
                                            DescrSim,
                                            GPU_csrRowS,
                                            *GPU_csrColS,
                                            P,
                                            pBuffer));
    
    // Step 4: gather sorted csrVal
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_csrValS, (sizeof(int)*(*nnzS))));  // Note that "(void**) GPU_csrValS" instead of "(void**) &GPU_csrValS"
    CHECK_CUSPARSE_SUCCESS(CUSPARSE_GTHR(handleCUSPARSE,
                                         (*nnzS),
                                         GPU_csrValS_unsorted,    // in-place sorting is not supported
                                         *GPU_csrValS,             // sorted
                                         P, 
                                         CUSPARSE_INDEX_BASE_ZERO));
    
    // Destroy descriptor & handle
    CHECK_CUSPARSE_SUCCESS(cusparseDestroyMatDescr(DescrSim));
    CHECK_CUSPARSE_SUCCESS(cusparseDestroy(handleCUSPARSE));
    
    // Memory deallocation
    CHECK_CUDA_SUCCESS(cudaFree(pBuffer));
    CHECK_CUDA_SUCCESS(cudaFree(P));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_csrValS_unsorted));
    
    finish = omp_get_wtime();
    Tomp_gpu_CSRSort += (finish - begin);
    
    printf("    Initialization:                    %f s\n", (float)(Tomp_Init));
    printf("    kernel_construct_similarity_chunk: %f s\n", Tcde_gpu_Kernel/1E3f);
    printf("    dense2csr:                         %f s\n", (float)(Tomp_gpu_Dense2CSR));
    printf("    csrsort:                           %f s\n", (float)(Tomp_gpu_CSRSort));
        
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
        // save_file_real(csrValS, (*nnzS),      1, "output/csrValS_s3.txt", "", 0);
        // save_file_int (csrRowS, nbPoints + 1, 1, "output/csrRowS_s3.txt", "", 0);
        // save_file_int (csrColS, (*nnzS),      1, "output/csrColS_s3.txt", "", 0);
        // free(csrValS);
        // free(csrRowS);
        // free(csrColS);
}


__global__ void kernel_compute_nnz_per_row_based_on_csrrow (int nbPoints, 
                                                            int *GPU_csrRowS,
                                                            int *GPU_nnzPerRowS)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nbPoints) {
        GPU_nnzPerRowS[tid] = GPU_csrRowS[tid + 1] - GPU_csrRowS[tid];
    }
}


void compute_nnz_per_row_and_min_max_nnz_row_based_on_csrrow (int nbPoints,
                                                              int *GPU_csrRowS,
                                                              int *GPU_nnzPerRowS, 
                                                              int *minNnzRowS, int *maxNnzRowS)
{
    dim3 Dg, Db;
    
    // Compute nnz per row based on csrRow
    Db.x = BsXN;
    Db.y = 1;
    Dg.x = nbPoints/Db.x + (nbPoints%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    kernel_compute_nnz_per_row_based_on_csrrow<<<Dg, Db>>>(nbPoints, 
                                                           GPU_csrRowS,
                                                           GPU_nnzPerRowS);
    
    // Find minimal and maximal nnz in a row and their corresponding row numbers
    // "The thrust::minmax_element is potentially more efficient than separate calls to thrust::min_element and thrust::max_element."
    thrust::device_ptr<int> d_nnzPerRowS(GPU_nnzPerRowS);
    thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<int>> extrema = thrust::minmax_element(thrust::device, d_nnzPerRowS, d_nnzPerRowS + nbPoints);
    int idxMinNnzRowS = extrema.first - d_nnzPerRowS;
    int idxMaxNnzRowS = extrema.second - d_nnzPerRowS;
    *minNnzRowS = *extrema.first;
    *maxNnzRowS = *extrema.second;
    printf("    Min nnz in one row:  %d (at row %d)\n", *minNnzRowS, idxMinNnzRowS);
    printf("    Max nnz in one row:  %d (at row %d)\n", *maxNnzRowS, idxMaxNnzRowS);
}
