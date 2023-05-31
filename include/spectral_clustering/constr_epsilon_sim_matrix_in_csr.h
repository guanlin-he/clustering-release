#ifndef _CONSTR_EPSILON_SIM_MATRIX_IN_CSR_H
#define _CONSTR_EPSILON_SIM_MATRIX_IN_CSR_H

void algo_CSR1_for_similarity_matrix_construction (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                   int *GPU_nnzPerRowS, int *minNnzRowS, int *maxNnzRowS, int *nnzS,
                                                   T_real **GPU_csrValS, int **GPU_csrColS, int *GPU_csrRowS);

void algo_CSR2_for_similarity_matrix_construction (int nbPoints, int nbDims, T_real *GPU_dataT,
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                   int hypoMaxNnzRow, int pad1, int pad2,
                                                   int *GPU_nnzPerRowS, int *minNnzRowS, int *maxNnzRowS, int *nnzS,
                                                   T_real **GPU_csrValS, int **GPU_csrColS, int *GPU_csrRowS);

#endif