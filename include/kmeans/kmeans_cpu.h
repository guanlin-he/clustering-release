#ifndef _KMEANS_CPU_H
#define _KMEANS_CPU_H

void random_sampling (int nbPoints, int nbDims, int nbClusters,  // input
                      T_real *data,                              // input
                      unsigned int seedbase,                     // input
                      T_real *centroids);                        // output

void d2_sampling (int nbPoints, int nbDims, int nbClusters,  // input
                  T_real *data,                              // input
                  unsigned int seedbase,                     // input
                  T_real *centroids);                        // output

void kmeans_cpu (int nbPoints, int nbDims, int nbClusters, T_real *data,               // input
                 int seedingMethod, unsigned int seedbase,                             // input
                 int tholdUsePackages, int nbPackages,                                 // input
                 T_real tolKMCPU, int maxNbItersKM,                                    // input
                 int *nbIters, int *countPerCluster, T_real *centroids, int *labels);  // output

void kmeans_cpu_for_extracting_representatives (int nbPoints, int nbDims, int nbClusters, T_real *data,  // input
                                                int seedingMethod, unsigned int seedbase,                // input
                                                int tholdUsePackages, int nbPackages,                    // input
                                                T_real tolKMCPU, int maxNbItersKM,                       // input
                                                int *nbIters, T_real *centroids, int *labels);           // output

#endif