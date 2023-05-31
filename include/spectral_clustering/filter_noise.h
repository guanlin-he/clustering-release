#ifndef _FILTER_NOISE_H
#define _FILTER_NOISE_H

void filter_noise_based_on_nnz_per_row (int nbPoints,                                                      // input
                                        int *GPU_nnzPerRowS, int minNnzRowS, int maxNnzRowS, int nnzS,     // input
                                        int *GPU_csrRowS, int *GPU_csrColS, T_real *GPU_csrValS,           // input
                                        int nbBinsHist, int flagInteractive, T_real tholdNoise,            // input
                                        int *nbPointsNF,                                                   // output
                                        int **GPU_nnzPerRowSNF, int *nnzSNF,                               // output
                                        int **GPU_csrRowSNF, int **GPU_csrColSNF, T_real **GPU_csrValSNF,  // output
                                        int *GPU_isNoise, int *GPU_nbNoiseFront,                           // output
                                        int *GPU_labels, double *timeUserResponse);                        // output
                                                 
void filter_noise_based_on_vertex_degree (int nbPoints,                                                      // input
                                          int *GPU_nnzPerRowS, int nnzS,                                     // input
                                          int *GPU_csrRowS, int *GPU_csrColS, T_real *GPU_csrValS,           // input
                                          int nbBinsHist, int flagInteractive, T_real tholdNoise,            // input
                                          int *nbPointsNF,                                                   // output
                                          int **GPU_nnzPerRowSNF, int *nnzSNF,                               // output
                                          int **GPU_csrRowSNF, int **GPU_csrColSNF, T_real **GPU_csrValSNF,  // output
                                          int *GPU_isNoise, int *GPU_nbNoiseFront,                           // output
                                          int *GPU_labels, double *timeUserResponse);                        // output

void get_noise_free_transposed_data_matrix (int nbDims, int nbPoints, T_real *GPU_dataT,
                                            int *GPU_isNoise, int *GPU_nbNoiseFront,
                                            int nbPointsNF,
                                            T_real *GPU_dataTNF);

void merge_labels (int nbPoints, int nbPointsNF, 
                   int *GPU_isNoise, int *GPU_nbNoiseFront,
                   int *GPU_labelsNF,
                   int *GPU_labels);

#endif