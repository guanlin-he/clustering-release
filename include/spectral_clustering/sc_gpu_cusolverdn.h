#ifndef _SC_GPU_CUSOLVERDN_H
#define _SC_GPU_CUSOLVERDN_H

void spectral_clustering_on_gpu_involving_cusolverdn (int nbPoints, int nbDims, int nbClusters, T_real *GPU_dataT,                    // input
                                                      T_real sigma, T_real tholdSim, T_real tholdSqDist,                              // input
                                                      int flagAutoTuneNbClusters, int maxNbClusters, int flagInteractive,             // input
                                                      int seedingMethod, unsigned int seedBase, T_real tolKMGPU, int maxNbItersKM,    // input 
                                                      int tholdUsePackages, int nbPackages, int nbStreamsStep1, int nbStreamsStep2,   // input
                                                      int *nbItersKM, int *optNbClusters, int *GPU_count, int *GPU_labels);           // output

#endif