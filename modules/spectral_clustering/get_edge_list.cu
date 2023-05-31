#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cusparse.h>

#include "../../include/config.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/spectral_clustering/get_edge_list.h"


void get_edge_list_from_csr (int nbPoints, int nnz,                                 // input
                             int *GPU_csrRow, int *GPU_csrCol, T_real *GPU_csrVal)  // input
{
    // Declaration
    int *GPU_cooRow;
    int *cooRow;
    int *cooCol;
    T_real *cooVal;
    cusparseHandle_t   handleCUSPARSE;    // Handle for cuSPARSE library
    FILE *fp;                             // File pointer
    
    // Memory allocation
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_cooRow, sizeof(int)*nnz));
    cooRow = (int *) malloc(sizeof(int)*nnz);
    cooCol = (int *) malloc(sizeof(int)*nnz);
    cooVal = (T_real *) malloc(sizeof(T_real)*nnz);
    
    // CSR to COO conversion
    CHECK_CUSPARSE_SUCCESS(cusparseCreate(&handleCUSPARSE));
    CHECK_CUSPARSE_SUCCESS(cusparseXcsr2coo(handleCUSPARSE,
                                            GPU_csrRow,
                                            nnz,
                                            nbPoints,
                                            GPU_cooRow,
                                            CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE_SUCCESS(cusparseDestroy(handleCUSPARSE));
    
    // Copy results from device to host
    CHECK_CUDA_SUCCESS(cudaMemcpy(cooRow, GPU_cooRow, sizeof(int)*nnz, cudaMemcpyDeviceToHost)); 
    CHECK_CUDA_SUCCESS(cudaMemcpy(cooCol, GPU_csrCol, sizeof(int)*nnz, cudaMemcpyDeviceToHost)); 
    CHECK_CUDA_SUCCESS(cudaMemcpy(cooVal, GPU_csrVal, sizeof(T_real)*nnz, cudaMemcpyDeviceToHost)); 
    
    // Save unweighted edge list
    // fp = fopen("output/EdgeList.txt", "w");
    // if (fp == NULL) {
        // printf("    Fail to open file!\n");
        // exit(0);
    // }
    // for (int i = 0; i < nnz; i++) {
        // fprintf(fp, "%d %d\n", cooRow[i], cooCol[i]);
    // }
    // fclose(fp);
    
    // Save weighted edge list
    fp = fopen("output/WeightedEdgeList.csv", "w");
    if (fp == NULL) {
        printf("    Fail to open file!\n");
        exit(0);
    }
    for (int i = 0; i < nnz; i++) {
        fprintf(fp, "%d, %d, %f\n", cooRow[i], cooCol[i], cooVal[i]);
    }
    fclose(fp);
    
    // Save weight list
    // fp = fopen("output/WeightList.txt", "w");
    // if (fp == NULL) {
        // printf("    Fail to open file!\n");
        // exit(0);
    // }
    // for (int i = 0; i < nnz; i++) {
        // fprintf(fp, "%f\n", cooVal[i]);
    // }
    // fclose(fp);
    
    // Memory deallocation
    CHECK_CUDA_SUCCESS(cudaFree(GPU_cooRow));
    free(cooRow);
    free(cooCol);
    free(cooVal);
}