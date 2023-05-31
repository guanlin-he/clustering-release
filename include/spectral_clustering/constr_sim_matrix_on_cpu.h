#ifndef _CONSTR_SIM_MATRIX_ON_CPU_H
#define _CONSTR_SIM_MATRIX_ON_CPU_H

void constr_similarity_matrix_on_cpu (int nbPoints, int nbDims, T_real *data,
                                      int constrAlgoCPU, int NbThreadsCPU,
                                      T_real sigma, T_real tholdSim, T_real tholdSqDist);

#endif