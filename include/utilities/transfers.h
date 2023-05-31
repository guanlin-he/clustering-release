#ifndef _TRANSFERS_H
#define _TRANSFERS_H

void real_data_register (T_real *data, size_t size);
void float_data_register (float *data, size_t size);
void double_data_register (double *data, size_t size);
void real_data_transfers_cpu_to_gpu (T_real *data, size_t size,  // input
                                     T_real *GPU_data);          // output
void float_data_transfers_cpu_to_gpu (float *data, size_t size,  // input
                                      float *GPU_data);          // output
void double_data_transfers_cpu_to_gpu (double *data, size_t size,  // input
                                       double *GPU_data);          // output


void int_data_register (int *data, size_t size);
void int_data_transfers_cpu_to_gpu (int *data, size_t size,  // input
                                    int *GPU_data);          // output


void real_data_transfers_gpu_to_cpu (T_real *GPU_data, size_t size,  // input
                                     T_real *data);                  // output
void float_data_transfers_gpu_to_cpu (float *GPU_data, size_t size,  // input
                                      float *data);                  // output
void double_data_transfers_gpu_to_cpu (double *GPU_data, size_t size,  // input
                                       double *data);                  // output
void real_data_unregister (T_real *data);
void float_data_unregister (float *data);
void double_data_unregister (double *data);


void int_data_transfers_gpu_to_cpu (int *GPU_data, size_t size,  // input
                                    int *data);                  // output
void int_data_unregister (int *data);

#endif