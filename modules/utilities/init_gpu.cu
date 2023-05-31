#include <cuda.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h> 
#include <nvgraph.h>

#include "../../include/config.h"
#include "../../include/utilities/init_gpu.h"


cublasHandle_t handleCUBLAS;     // Handle for cuBLAS library
nvgraphHandle_t handleNVGRAPH;   // Handle for nvGRAPH library
nvgraphGraphDescr_t descrG;      // Graph descriptor


// CUDA event used for time measurement
cudaEvent_t StartEvent;
cudaEvent_t StopEvent;


// Init GPU device
void init_gpu()
{
    cuInit(0);  // 0.04s ~ 0.05s with CUDA 11.5 (0.2s ~ 0.6s with older CUDA version)
    
    CHECK_CUBLAS_SUCCESS(cublasCreate(&handleCUBLAS));  // 1.11s ~ 1.15s with CUDA 11.5
    
    CHECK_NVGRAPH_SUCCESS(nvgraphCreate(&handleNVGRAPH));  // 0.73s ~ 0.94s with CUDA 11.5
    CHECK_NVGRAPH_SUCCESS(nvgraphCreateGraphDescr(handleNVGRAPH, &descrG)); // 0.00s with CUDA 11.5
    
    // cudaEventBlockingSync: Specifies that event should use blocking synchronization.
    // A host thread that uses cudaEventSynchronize() to wait on an event created with this flag will block until the event actually completes.
    CHECK_CUDA_SUCCESS(cudaEventCreateWithFlags(&StartEvent, cudaEventBlockingSync)); // 0.03s ~ 0.07s with CUDA 11.5
    CHECK_CUDA_SUCCESS(cudaEventCreateWithFlags(&StopEvent, cudaEventBlockingSync));
    
}


// Finalize GPU device
void finalize_gpu()
{
    CHECK_CUBLAS_SUCCESS(cublasDestroy(handleCUBLAS));
    CHECK_NVGRAPH_SUCCESS(nvgraphDestroyGraphDescr(handleNVGRAPH, descrG));
    CHECK_NVGRAPH_SUCCESS(nvgraphDestroy(handleNVGRAPH));
    
    CHECK_CUDA_SUCCESS(cudaEventDestroy(StartEvent));
    CHECK_CUDA_SUCCESS(cudaEventDestroy(StopEvent));
}


// GPU memory allocation for real data
void real_data_memory_allocation_gpu (T_real **GPU_data, size_t size)
{
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_data, size));
}

// GPU memory allocation for data of "float" type
void float_data_memory_allocation_gpu (float **GPU_data, size_t size)
{
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_data, size));
}

// GPU memory allocation for data of "double" type
void double_data_memory_allocation_gpu (double **GPU_data, size_t size)
{
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_data, size));
}


// GPU memory allocation for integer data
void int_data_memory_allocation_gpu (int **GPU_data, size_t size)
{
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) GPU_data, size));
}


// GPU memory deallocation for real data
void real_data_memory_deallocation_gpu (T_real *GPU_data)
{
    CHECK_CUDA_SUCCESS(cudaFree(GPU_data));
}

// GPU memory deallocation for data of "float" type
void float_data_memory_deallocation_gpu (float *GPU_data)
{
    CHECK_CUDA_SUCCESS(cudaFree(GPU_data));
}

// GPU memory deallocation for data of "double" type
void double_data_memory_deallocation_gpu (double *GPU_data)
{
    CHECK_CUDA_SUCCESS(cudaFree(GPU_data));
}


// GPU memory deallocation for integer data
void int_data_memory_deallocation_gpu (int *GPU_data) 
{
    CHECK_CUDA_SUCCESS(cudaFree(GPU_data));
}
