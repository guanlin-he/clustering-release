#ifndef _SC_GPU_CUGRAPH_H
#define _SC_GPU_CUGRAPH_H

void spectral_clustering_on_gpu_involving_cugraph (int nbPoints, int nbDims, int nbClusters, T_real *GPU_dataT,        // input 
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,                  // input 
                                                   int csrAlgo, int hypoMaxNnzRow, T_real maxNzPercent,                // input 
                                                   T_real memUsePercent, int pad1, int pad2, int pad3,                 // input 
                                                   int filterNoiseApproach, int nbBinsHist, T_real tholdNoise,         // input
                                                   int flagAutoTuneNbClusters, int flagInteractive,                    // input
                                                   int cuGraphAlgo, T_real tolEigen, int maxNbItersEigen,              // input 
                                                   T_real tolKMGPU, int maxNbItersKM,                                  // input 
                                                   float *modularityScore, float *edgeCutScore, float *ratioCutScore,  // input 
                                                   int *optNbClusters, int *GPU_labels);                               // output
    
#endif