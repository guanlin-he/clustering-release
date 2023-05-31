#ifndef _CONSTR_SIM_LAP_MATRIX_IN_DENSE_H
#define _CONSTR_SIM_LAP_MATRIX_IN_DENSE_H

void construct_similarity_degree_matrix (int nbPoints, int nbDims, T_real *GPU_dataT,
                                         T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                         T_real *GPU_sim, T_real *GPU_deg);

void compute_laplacian_matrix (int nbPoints, T_real *GPU_sim, T_real *GPU_deg,
                               T_real *GPU_lap);

#endif