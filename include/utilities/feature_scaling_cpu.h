#ifndef _FEATURE_SCALING_CPU_H
#define _FEATURE_SCALING_CPU_H

void feature_scaling (int nbPoints, int nbDims,
                      T_real *data, 
                      T_real *dimMax, T_real *dimMin);

void inverse_feature_scaling (T_real *dimMax, T_real *dimMin,
                              int nbPoints, int nbDims,
                              T_real *data);

#endif