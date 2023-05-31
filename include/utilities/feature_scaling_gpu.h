#ifndef _FEATURE_SCALING_GPU_H
#define _FEATURE_SCALING_GPU_H

void find_min_max (int nbDims, int nbPoints,              // input
                   T_real *GPU_dataT,                     // input
                   float *GPU_dimMax, float *GPU_dimMin); // output

void feature_scaling_on_gpu (int nbDims, int nbPoints,       // input
                             T_real *GPU_dataT,              // input & output
                             float *dimMax, float *dimMin);  // output

void inverse_feature_scaling_on_gpu (int nbDims, int nbPoints,       // input
                                     float *dimMax, float *dimMin,   // input
                                     T_real *GPU_dataT);             // input & output

void compute_unscaled_centroids (float *GPU_dimMax, float *GPU_dimMin,
                                 int nbClusters, int nbDims,
                                 T_real *GPU_centroids);

#endif