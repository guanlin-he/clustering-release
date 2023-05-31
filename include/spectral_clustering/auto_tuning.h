#ifndef _AUTO_TUNING_H
#define _AUTO_TUNING_H

void auto_tune_nb_clusters_based_on_eigengaps (int flagInteractive,
                                               int maxNbClusters, T_real *GPU_eigVals,
                                               int *optNbClusters, double *timeUserResponse);

void save_decision_graph_and_determine_nb_clusters (int nbPoints,
                                                    int *GPU_density, T_real *GPU_delta,
                                                    int *optNbClusters, double *timeUserResponse);

#endif