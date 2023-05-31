#include <stdio.h>  // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h> // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <float.h>  // Library Macros (e.g. FLT_MAX, FLT_MIN)
#include <omp.h>

#include "../../include/config.h"
#include "../../include/utilities/attachment.h"


// Attach each point to its nearest representative
void attach_to_representative (int nbPoints, int nbDims, int nbReps,
                               T_real *data, T_real *reps,
                               int *labels)
{
    #pragma omp parallel for
    for (int i = 0; i < nbPoints; i++) {
        int min = 0;
        T_real sqDist, minDistSq = FLT_MAX;
        for (int k = 0; k < nbReps; k++) {
            // Calculate the square of distance between instance i and centroid k
            sqDist = 0.0f;
            T_real diff;
            index_t idxOffset1 = ((index_t)i)*((index_t)nbDims);
            int     idxOffset2 = k*nbDims;
            for (int j = 0; j < NB_DIMS; j ++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                diff = data[idxOffset1 + (index_t)j] - reps[idxOffset2 + j];
                sqDist += diff*diff;
            }
            // Find and record the nearest centroid to instance i
            bool a = (sqDist < minDistSq);
            min = (a ? k : min);
            minDistSq = (a ? sqDist : minDistSq);
        }
        
        // Change the label if necessary
        if (labels[i] != min) { // TO REMOVE !!!! Unnecessary
            labels[i] = min;
        }
    }
}


void membership_attachment (int nbPoints, int *labels_reps,
                            int *labels_alldata)
{   
    // Sequential version
    // int rid;  // Representative ID
    // for (int i = 0; i < nbPoints; i++) {
        // rid = labels_alldata[i];  
        // labels_alldata[i] = labels_reps[rid];
    // }

    // Parallel version
    #pragma omp parallel for
    for (int i = 0; i < nbPoints; i++) {
        int rid = labels_alldata[i];  // Representative ID
        labels_alldata[i] = labels_reps[rid];
    }
}