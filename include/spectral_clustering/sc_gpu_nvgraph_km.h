#ifndef _SC_GPU_NVGRAPH_KM_H
#define _SC_GPU_NVGRAPH_KM_H

void spectral_clustering_on_gpu_involving_nvgraph_and_kmeans (int nbPoints, int nbDims, int nbClusters, T_real *GPU_dataT,                   // input
                                                              T_real sigma, T_real tholdSim, T_real tholdSqDist,                             // input
                                                              int csrAlgo, int hypoMaxNnzRow, T_real maxNzPercent,                           // input
                                                              T_real memUsePercent, int pad1, int pad2, int pad3,                            // input
                                                              int filterNoiseApproach, int nbBinsHist, T_real tholdNoise,                    // input
                                                              int nvGraphAlgo, T_real tolEigen, int maxNbItersEigen,                         // input
                                                              int flagAutoTuneNbClusters, int maxNbClusters, int flagInteractive,            // input
                                                              int seedingMethod, unsigned int seedBase, T_real tolKMGPU, int maxNbItersKM,   // input
                                                              int tholdUsePackages, int nbPackages, int nbStreamsStep1, int nbStreamsStep2,  // input 
                                                              float *modularityScore, float *edgeCutScore, float *ratioCutScore,             // input
                                                              int *optNbClusters, int *nbItersKM, int *GPU_count, int *GPU_labels);          // output

#endif