#ifndef _ATTACHMENT_H
#define _ATTACHMENT_H

void attach_to_representative (int nbPoints, int nbDims, int nbClusters,  // input
                               T_real *data, T_real *centroids,           // input
                               int *labels);                              // output

void membership_attachment (int nbPoints, int *labels_reps,  // input
                            int *labels_alldata);            // output

#endif