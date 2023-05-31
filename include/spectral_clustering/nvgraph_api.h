#ifndef _NVGRAPH_API_H
#define _NVGRAPH_API_H

void similarity_graph_partitioning_via_nvgraph (int nbPoints, int nbClusters, int nnzS,                       // input 
                                                int *GPU_csrRowS, int *GPU_csrColS, T_real *GPU_csrValS,      // input
                                                int nvGraphAlgo, T_real tolEigen, int maxNbItersEigen,        // input 
                                                T_real tolKMGPU, int maxNbItersKM,                            // input 
                                                T_real *GPU_eigVals, T_real *GPU_eigVects, int *GPU_labels);  // output

void evaluate_clustering_quality_via_nvgraph (int nbPoints, int nbClusters, int nnzS,                             // input 
                                              int *GPU_csrRowS, int *GPU_csrColS, T_real *GPU_csrValS,            // input
                                              int *GPU_labels,                                                    // input
                                              float *modularityScore, float *edgeCutScore, float *ratioCutScore); // output

#endif