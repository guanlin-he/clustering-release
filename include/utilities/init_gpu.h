#ifndef _INIT_GPU_H
#define _INIT_GPU_H

// #include <stdio.h>  // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
// #include <stdlib.h> // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <nvgraph.h>


extern cublasHandle_t handleCUBLAS;     // Handle for cuBLAS library
extern nvgraphHandle_t handleNVGRAPH;   // Handle for nvGRAPH library
extern nvgraphGraphDescr_t descrG;      // Graph descriptor


// CUDA event used for time measurement
extern cudaEvent_t StartEvent;
extern cudaEvent_t StopEvent;


// Starting address for dynamic allocation of shared memory
extern __shared__ T_real shBuff[];


// Functions
void init_gpu();
void real_data_memory_allocation_gpu (T_real **GPU_data, size_t size);
void float_data_memory_allocation_gpu (float **GPU_data, size_t size);
void double_data_memory_allocation_gpu (double **GPU_data, size_t size);
void int_data_memory_allocation_gpu (int **GPU_data, size_t size); 
void real_data_memory_deallocation_gpu (T_real *GPU_data);
void float_data_memory_deallocation_gpu (float *GPU_data);
void double_data_memory_deallocation_gpu (double *GPU_data);
void int_data_memory_deallocation_gpu (int *GPU_data);
void finalize_gpu();


// Error checking on CUDA APIs and CUDA kernels
#define CHECK_CUDA_SUCCESS(X) {if ((X) != cudaSuccess) { \
    fprintf(stderr, "Error %d (%s) in %s at line %d in %s\n", int(X), cudaGetErrorString(X), __FILE__, __LINE__, __func__); \
    exit(EXIT_FAILURE);}}


// Error checking on cuBLAS APIs
inline const char* cublasGetErrorString(cublasStatus_t error) 
{
    switch (error) {
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "Unknown cuBLAS error";
}
#define CHECK_CUBLAS_SUCCESS(X) {if ((X) != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "Error %d (%s) in %s at line %d in %s\n", int(X), cublasGetErrorString(X), __FILE__, __LINE__, __func__); \
    exit(EXIT_FAILURE);}}


// Error checking on cuSPARSE APIs
inline const char* cusparseGetErrorString(cusparseStatus_t error) 
{
    switch (error) {
        case CUSPARSE_STATUS_NOT_INITIALIZED:           return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:              return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:             return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:             return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_EXECUTION_FAILED:          return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:            return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_NOT_SUPPORTED:             return "CUSPARSE_STATUS_NOT_SUPPORTED";
        //case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:    return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    }
    return "Unknown cuSPARSE error";
}
#define CHECK_CUSPARSE_SUCCESS(X) {if ((X) != CUSPARSE_STATUS_SUCCESS) { \
    fprintf(stderr, "Error %d (%s) in %s at line %d in %s\n", int(X), cusparseGetErrorString(X), __FILE__, __LINE__, __func__); \
    exit(EXIT_FAILURE);}}


// Error checking on cuSOLVER APIs
inline const char* cusolverGetErrorString(cusolverStatus_t error) 
{
    switch (error) {
        case CUSOLVER_STATUS_NOT_INITIALIZED:           return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED:              return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE:             return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH:             return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED:          return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR:            return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
    return "Unknown cuSOLVER error";
}
#define CHECK_CUSOLVER_SUCCESS(X) {if ((X) != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "Error %d (%s) in %s at line %d in %s\n", int(X), cusolverGetErrorString(X), __FILE__, __LINE__, __func__); \
    exit(EXIT_FAILURE);}}


// Error checking on nvGRAPH APIs
inline const char* nvgraphGetErrorString(nvgraphStatus_t error) 
{
    switch (error) {
        case NVGRAPH_STATUS_NOT_INITIALIZED:    return "NVGRAPH_STATUS_NOT_INITIALIZED";
        case NVGRAPH_STATUS_ALLOC_FAILED:       return "NVGRAPH_STATUS_ALLOC_FAILED";
        case NVGRAPH_STATUS_INVALID_VALUE:      return "NVGRAPH_STATUS_INVALID_VALUE";
        case NVGRAPH_STATUS_ARCH_MISMATCH:      return "NVGRAPH_STATUS_ARCH_MISMATCH";
        case NVGRAPH_STATUS_MAPPING_ERROR:      return "NVGRAPH_STATUS_MAPPING_ERROR";
        case NVGRAPH_STATUS_EXECUTION_FAILED:   return "NVGRAPH_STATUS_EXECUTION_FAILED";
        case NVGRAPH_STATUS_INTERNAL_ERROR:     return "NVGRAPH_STATUS_INTERNAL_ERROR";
        case NVGRAPH_STATUS_TYPE_NOT_SUPPORTED: return "NVGRAPH_STATUS_TYPE_NOT_SUPPORTED";
        case NVGRAPH_STATUS_NOT_CONVERGED:      return "NVGRAPH_STATUS_NOT_CONVERGED";
    }
    return "Unknown nvGRAPH error";
}
#define CHECK_NVGRAPH_SUCCESS(X) {if ((X) != NVGRAPH_STATUS_SUCCESS) { \
    fprintf(stderr, "Error %d (%s) in %s at line %d in %s\n", int(X), nvgraphGetErrorString(X), __FILE__, __LINE__, __func__); \
    exit(EXIT_FAILURE);}}

#endif