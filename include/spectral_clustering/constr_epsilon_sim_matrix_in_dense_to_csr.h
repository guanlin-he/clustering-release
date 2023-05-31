#ifndef _CONSTR_EPSILON_SIM_MATRIX_IN_DENSE_TO_CSR_H
#define _CONSTR_EPSILON_SIM_MATRIX_IN_DENSE_TO_CSR_H

void algo_CSR3_for_similarity_matrix_construction (int nbPoints, int nbDims, T_real *GPU_dataT,                           // input
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,                     // input
                                                   T_real memUsePercent, T_real maxNzPercent,                             // input
                                                   int *nnzS, T_real **GPU_csrValS, int **GPU_csrColS, int *GPU_csrRowS); // output

void compute_nnz_per_row_and_min_max_nnz_row_based_on_csrrow (int nbPoints,
                                                              int *GPU_csrRowS,
                                                              int *GPU_nnzPerRowS, 
                                                              int *minNnzRowS, int *maxNnzRowS);

#endif