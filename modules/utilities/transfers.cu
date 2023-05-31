#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>

#include "../../include/config.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/transfers.h"


// Transfer real data from host to device
void real_data_register (T_real *data, size_t size)
{
    CHECK_CUDA_SUCCESS(cudaHostRegister(data, size, cudaHostRegisterPortable));
}
void float_data_register (float *data, size_t size)
{
    CHECK_CUDA_SUCCESS(cudaHostRegister(data, size, cudaHostRegisterPortable));
}
void double_data_register (double *data, size_t size)
{
    CHECK_CUDA_SUCCESS(cudaHostRegister(data, size, cudaHostRegisterPortable));
}

void real_data_transfers_cpu_to_gpu (T_real *data, size_t size,
                                     T_real *GPU_data)
{   
    CHECK_CUDA_SUCCESS(cudaMemcpy(GPU_data, data, size, cudaMemcpyHostToDevice));
}
void float_data_transfers_cpu_to_gpu (float *data, size_t size,
                                     float *GPU_data)
{   
    CHECK_CUDA_SUCCESS(cudaMemcpy(GPU_data, data, size, cudaMemcpyHostToDevice));
}
void double_data_transfers_cpu_to_gpu (double *data, size_t size,
                                     double *GPU_data)
{   
    CHECK_CUDA_SUCCESS(cudaMemcpy(GPU_data, data, size, cudaMemcpyHostToDevice));
}



// Transfer integer data from host to device
void int_data_register (int *data, size_t size)
{
    CHECK_CUDA_SUCCESS(cudaHostRegister(data, size, cudaHostRegisterPortable));
}


void int_data_transfers_cpu_to_gpu (int *data, size_t size,
                                    int *GPU_data)
{   
    CHECK_CUDA_SUCCESS(cudaMemcpy(GPU_data, data, size, cudaMemcpyHostToDevice));
}




// Transfer real data from device to host
void real_data_transfers_gpu_to_cpu (T_real *GPU_data, size_t size,
                                     T_real *data)
{
    CHECK_CUDA_SUCCESS(cudaMemcpy(data, GPU_data, size, cudaMemcpyDeviceToHost));
}
void float_data_transfers_gpu_to_cpu (float *GPU_data, size_t size,
                                     float *data)
{
    CHECK_CUDA_SUCCESS(cudaMemcpy(data, GPU_data, size, cudaMemcpyDeviceToHost));
}
void double_data_transfers_gpu_to_cpu (double *GPU_data, size_t size,
                                     double *data)
{
    CHECK_CUDA_SUCCESS(cudaMemcpy(data, GPU_data, size, cudaMemcpyDeviceToHost));
}

void real_data_unregister (T_real *data)
{
    CHECK_CUDA_SUCCESS(cudaHostUnregister(data));
}
void float_data_unregister (float *data)
{
    CHECK_CUDA_SUCCESS(cudaHostUnregister(data));
}
void double_data_unregister (double *data)
{
    CHECK_CUDA_SUCCESS(cudaHostUnregister(data));
}



// Transfer integer data from device to host
void int_data_transfers_gpu_to_cpu (int *GPU_data, size_t size,
                                    int *data)
{
    CHECK_CUDA_SUCCESS(cudaMemcpy(data, GPU_data, size, cudaMemcpyDeviceToHost));
}

void int_data_unregister (int *data)
{
    CHECK_CUDA_SUCCESS(cudaHostUnregister(data));
}