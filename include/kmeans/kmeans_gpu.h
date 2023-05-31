#ifndef _KMEANS_GPU_H
#define _KMEANS_GPU_H

void seeding (int nbPoints, int nbDims, int nbClusters,                      // input
              T_real *GPU_dataT, int *GPU_labels,                            // input
              int seedingMethod, unsigned int seedBase,                      // input
              bool flagUsePackages, int nbPackages,                          // input
              int nbStreamsStep1, int nbStreamsStep2,                        // input
              T_real *GPU_centroids);                                        // output

void compute_assign (int nbPoints, int nbDims, int nbClusters,         // input
                     T_real *GPU_dataT, T_real *GPU_centroids,         // input
                     int *GPU_labels, unsigned long long int *track);  // output

void update_centroids (int clustAlgo,                                              // input
                       int nbPoints, int nbDims, int nbClusters,                   // input
                       T_real *GPU_dataT, int *GPU_labels,                         // input
                       bool flagUsePackages, int nbPackages, T_real *GPU_packages, // input
                       int nbStreamsStep1, int nbStreamsStep2,                     // input
                       int *GPU_count, T_real *GPU_centroidsT);                    // output

void kmeans_gpu (int clustAlgo,                                                                 // input (env)
                 int nbPoints, int nbDims, int nbClusters, T_real *GPU_dataT,                   // input
                 int seedingMethod, unsigned int seedBase, T_real tolKMGPU, int maxNbItersKM,   // input 
                 int tholdUsePackages, int nbPackages, int nbStreamsStep1, int nbStreamsStep2,  // input
                 int *nbIters, int *GPU_count, T_real *GPU_centroids, int *GPU_labels);         // output

void gpu_attach_to_representative(int nbPoints, int nbDims, int nbReps,  // input
                                  T_real *GPU_dataT, T_real *GPU_reps,   // input
                                  int *GPU_labels);                      // output

void gpu_membership_attachment (int nbPoints, int *GPU_labelsReps,  // input
                               int *GPU_labels);                    // output

#endif