#include <stdio.h>  // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h> // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <float.h>  // Library Macros (e.g. FLT_MAX, FLT_MIN)
#include <omp.h>
#include <thrust/reduce.h> // thrust::reduce
#include <thrust/scan.h>   // thrust::exclusive_scan, inclusive_scan
#include <thrust/execution_policy.h>  // using the thrust::host execution policy for parallelization

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/utilities/attachment.h"
#include "../../include/kmeans/kmeans_cpu.h"


// Select nbClusters initial centroids from nbPoints instances using random sampling
void random_sampling (int nbPoints, int nbDims, int nbClusters,
                      T_real *data,
                      unsigned int seedbase,
                      T_real *centroids)
{
    #pragma omp parallel
    {
        // Declare a seed for rand_r() function
        unsigned int seed = seedbase * omp_get_thread_num();
        
        #pragma omp for
        for (int k = 0; k < nbClusters; k++) { 
            // rand_r() is multithread safe. 
            // Like rand(), rand_r() returns a pseudo-random integer in the range [0, RAND_MAX]
            int centerIdx = rand_r(&seed)/(T_real)RAND_MAX * nbPoints;
            index_t idxOffset1 = ((index_t)centerIdx)*((index_t)nbDims);
            int     idxOffset2 = k*nbDims;
            for (int j = 0; j < NB_DIMS; j++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                centroids[idxOffset2 + j] = data[idxOffset1 + (index_t)j];
            }
        }
    }
}


// Select nbClusters initial centroids from nbPoints instances using D² sampling
void d2_sampling (int nbPoints, int nbDims, int nbClusters,
                  T_real *data,
                  unsigned int seedbase,
                  T_real *centroids)
{
    // Declaration
    unsigned int seed = seedbase;  // time(NULL) will be different each time you launch the program
    int centerIdx;
    double randValue;
    double *d2;
    double d2Total;
    double *sampProb;
    double *inScanSum;
    double *exScanSum;
    omp_lock_t lock;
    
    // Memory allocation
    d2        = (double *) malloc(sizeof(double)*nbPoints);
    sampProb  = (double *) malloc(sizeof(double)*nbPoints);
    inScanSum = (double *) malloc(sizeof(double)*nbPoints);
    exScanSum = (double *) malloc(sizeof(double)*nbPoints);

    // Initialization
    omp_init_lock(&lock);
    

    #pragma omp parallel
    {
        for (int k = 0; k < nbClusters; k++) {

            #pragma omp single
            {
                if (k == 0) {
                    centerIdx = rand_r(&seed)/(double)RAND_MAX * nbPoints;
                }
                randValue = rand_r(&seed)/(double)RAND_MAX;
                d2Total = 0.0;
            }

            if (k > 0) {
                // Find the new center
                #pragma omp for
                for (int i = 0; i < nbPoints; i++) {
                    double exSS = exScanSum[i];
                    double inSS = inScanSum[i];
                    if ( (randValue >= exSS) && (randValue < inSS) ) {
                        omp_set_lock(&lock);   // necessary due to the -Ofast
                        centerIdx = i;
                        omp_unset_lock(&lock); // necessary due to the -Ofast
                    }
                }
            }
            
            // Store the new center
            index_t idxOffset1 = ((index_t)centerIdx)*((index_t)nbDims);
            int     idxOffset2 = k*nbDims;
            #pragma omp for
            for (int j = 0; j < NB_DIMS; j++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                centroids[idxOffset2 + j] = data[idxOffset1 + (index_t)j];
            }

            // Update D²(x): the shortest distance from a data point to the closest center that have already been chosen
            // - Solution 1: results vary with the number of threads, and also vary from one run to another.
            // #pragma omp for reduction(+: d2Total)  // reduction order is unknown and uncontrollable --> source of result indeterminacy (with identical arguments) due to rounding errors
            // for (int i = 0; i < nbPoints; i++) {
                // T_real sqDist = 0.0f;
                // T_real minDistSq = (k > 0 ? d2[i] : FLT_MAX);
                // for (int j = 0; j < NB_DIMS; j++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                    // index_t dataIdx1 = ((index_t)i)*((index_t)nbDims) + ((index_t)j);
                    // index_t dataIdx2 = ((index_t)centerIdx)*((index_t)nbDims) + ((index_t)j);
                    // sqDist += (data[dataIdx1] - data[dataIdx2])*(data[dataIdx1] - data[dataIdx2]);
                // }
                // if (sqDist < minDistSq) {
                    // minDistSq = sqDist;
                    // d2[i] = minDistSq;
                // }
                // d2Total += minDistSq;  // source of rounding errors
            // }
            
            // - Solution 2: a little slower than Solution 1 due to more memory accesses, but the result is stable.
            #pragma omp for
            for (int i = 0; i < nbPoints; i++) {
                T_real sqDist = 0.0f;
                T_real minDistSq = (k > 0 ? d2[i] : FLT_MAX);
                index_t idxOffset1 = ((index_t)i)*((index_t)nbDims);
                index_t idxOffset2 = ((index_t)centerIdx)*((index_t)nbDims);
                T_real diff;
                for (int j = 0; j < NB_DIMS; j++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                    diff = data[idxOffset1 + (index_t)j] - data[idxOffset2 + (index_t)j];
                    sqDist += diff*diff;
                }
                if (sqDist < minDistSq) {
                    d2[i] = sqDist;
                }
            }
            #pragma omp single
            {
                d2Total = thrust::reduce(thrust::host, d2, d2 + nbPoints, 0.0);  // Need to use 0.0 rather than 0.0f, otherwise the computations would not be completely in double precision
            }
            
            
            // Calculate sampling probability
            #pragma omp for
            for (int i = 0; i < nbPoints; i++) {
                sampProb[i] = d2[i] / d2Total;
            }
            
                
            #pragma omp single
            {
                // The following two scan functions can obtain wrong results in case of scanning an array with a HUGE number of elements in SINGLE precision!
                // We infer that it is caused by accumulation of rounding errors. 
                // Using double precision can significantly reduce the effect of rounding errors.
                thrust::inclusive_scan(thrust::host, sampProb, sampProb + nbPoints, inScanSum);
                thrust::exclusive_scan(thrust::host, sampProb, sampProb + nbPoints, exScanSum, 0.0); // Need to use 0.0 rather than 0.0f, otherwise the computations would not be completely in double precision.
            }
            
        } // end for 
    } // end #pragma omp parallel
    
    // Destroy lock
    omp_destroy_lock(&lock);
    
    // Memory deallocation
    free(d2);
    free(sampProb);
    free(inScanSum);
    free(exScanSum);
}



void kmeans_cpu (int nbPoints, int nbDims, int nbClusters, T_real *data,
                 int seedingMethod, unsigned int seedbase,
                 int tholdUsePackages, int nbPackages,
                 T_real tolKMCPU, int maxNbItersKM,
                 int *nbIters, int *countPerCluster, T_real *centroids, int *labels)
{
    // Declaration & definition
    unsigned long long int track;  // Number of label changes in two consecutive iterations
    T_real changeRatio;            // changeRatio = track / nbPoints
    printf("    NB_DIMS_LIMITED_BY_NB_CLUSTERS = %ld\n", NB_DIMS_LIMITED_BY_NB_CLUSTERS);
    printf("    NB_DIMS_BATCH_NC               = %ld\n", NB_DIMS_BATCH_NC);
    T_real sumPerClusterDL[NB_CLUSTERS][NB_DIMS_BATCH_NC]; // Array for the sum per cluster
    T_real packagesDL[NB_CLUSTERS][NB_DIMS_BATCH_NC];      // Array for the packages used in UpdateCentroids
    int    count[NB_CLUSTERS];                             // Array for the number of data instances in each cluster
    bool flagUsePackages;

    // Initialization
    track = 0;
    *nbIters = 0;
    flagUsePackages = (nbPoints/nbClusters > tholdUsePackages ? true: false);
    
    // Initialize centroids
    double td_seeding, tf_seeding; 
    td_seeding = omp_get_wtime();
    if (INPUT_INITIAL_CENTROIDS == "") {
        if (seedingMethod == 1) {
            random_sampling(nbPoints, nbDims, nbClusters,  // input
                            data,                          // input
                            seedbase,                      // input
                            centroids);                    // output
        }
        if (seedingMethod == 2) {
            d2_sampling(nbPoints, nbDims, nbClusters,  // input
                        data,                          // input
                        seedbase,                      // input
                        centroids);                    // output
        }
    } else {
        // read_file_real(centroids, nbClusters, nbDims, INPUT_INITIAL_CENTROIDS, "\t", 0, 0);
        read_file_real(centroids, nbClusters, nbDims, INPUT_INITIAL_CENTROIDS, " ", 0, 0);  // " " delimter for InitialCentroids_InputDataset-50million.txt
    }
    tf_seeding = omp_get_wtime();
    Tomp_cpu_seeding += (tf_seeding - td_seeding);


    #pragma omp parallel
    {
        // Declaration for each thread
        double td_compute_assign, tf_compute_assign; 
        double td_update, tf_update; 
        double d_compute_assign = 0.0, d_update = 0.0;

        do {    
            // Compute distances & assign points to clusters
            td_compute_assign = omp_get_wtime();
            #pragma omp for reduction(+: track)
            for (int i = 0; i < nbPoints; i++) {
                int min = 0;
                T_real sqDist, minDistSq = FLT_MAX;
                for (int k = 0; k < NB_CLUSTERS; k++) { // Using the constant "NB_CLUSTERS" instead of the variable "nbClusters" may improve the performance significantly.
                    // Calculate the square of distance between instance i and centroid k
                    sqDist = 0.0f;
                    T_real diff;
                    index_t idxOffset1 = ((index_t)i)*((index_t)nbDims);
                    int     idxOffset2 = k*nbDims;
                    for (int j = 0; j < NB_DIMS; j++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                        diff = data[idxOffset1 + (index_t)j] - centroids[idxOffset2 + j];
                        sqDist += diff*diff;
                    }
                    // Find and record the nearest centroid to instance i
                    bool a = (sqDist < minDistSq);
                    min = (a ? k : min);
                    minDistSq = (a ? sqDist : minDistSq);
                }

                // Change the label if necessary and count this change into track
                if (labels[i] != min) {
                    track++;
                    labels[i] = min;
                }
            }
            tf_compute_assign = omp_get_wtime();
            d_compute_assign += (tf_compute_assign - td_compute_assign);


            // Update centroids
            td_update = omp_get_wtime();
            if (flagUsePackages) {
                int j = 0;
                for ( ; j + NB_DIMS_BATCH_NC < nbDims; j = j + NB_DIMS_BATCH_NC) {
                    #pragma omp for
                    for (int k = 0; k < nbClusters; k++) {
                        count[k] = 0;
                        for (int jj = 0; jj < NB_DIMS_BATCH_NC /* && j + jj < nbDims */; jj++) {
                            sumPerClusterDL[k][jj] = 0.0f;
                        }
                    }

                    // In order to reduce the rounding error which happens when adding numbers of very different magnitudes,
                    // we first divide the dataset into packages, then calculate the sum of points in each packages, finally compute the sum of all packages.
                    int quotient, remainder, offset, length;
                    quotient = nbPoints/nbPackages;
                    remainder = nbPoints%nbPackages;
                    // Sum the contributions to each cluster
                    #pragma omp for private(packagesDL) reduction(+: count, sumPerClusterDL)
                    for (int p = 0; p < nbPackages; p++) {      // Process by packages
                        offset = (p < remainder ? ((quotient + 1) * p) : (quotient * p + remainder));
                        length = (p < remainder ? (quotient + 1) : quotient);
                        // Reset "packages" to zeros
                        for (int k = 0; k < nbClusters; k++) {
                            for (int jj = 0; jj < NB_DIMS_BATCH_NC /* && j + jj < nbDims */; jj++) {
                                packagesDL[k][jj] = 0.0f;
                            }
                        }
                        // 1st step local reduction
                        // - Count nb of instances in OpenMP reduction array
                        // - Reduction in thread private array
                        for (int i = offset; i < offset + length; i++) {
                            int k = labels[i];
                            count[k]++;
                            index_t idxOffset = ((index_t)i)*((index_t)nbDims) + ((index_t)j);
                            for (int jj = 0; jj < NB_DIMS_BATCH_NC /* && j + jj < nbDims */; jj++) {
                                packagesDL[k][jj] += data[idxOffset + (index_t)jj];
                            }
                        }
                        // 2nd step local reduction
                        // - Reduction in local OpenMP reduction array
                        for (int k = 0; k < nbClusters; k++) {
                            for (int jj = 0; jj < NB_DIMS_BATCH_NC /* && j + jj < nbDims */; jj++) {
                                sumPerClusterDL[k][jj] += packagesDL[k][jj];
                            }
                        }
                    }   // 2nd step global reduction: final reduction by OpenMP in global "sumPerClusterDL" array
                
                    // Final averaging to get new centroids
                    #pragma omp for
                    for (int k = 0; k < nbClusters; k++) {  // Process by cluster
                        if (count[k] != 0) {
                            for (int jj = 0; jj < NB_DIMS_BATCH_NC /* && j + jj < nbDims */; jj++) {
                                centroids[k*nbDims + j + jj] = sumPerClusterDL[k][jj]/count[k];    // - Calculate global "centroids" array
                            }
                        }
                    }
                } // End of for ( ; j + NB_DIMS_BATCH_NC < nbDims; j = j + NB_DIMS_MIN)
                
                if (j < nbDims) {
                    #pragma omp for
                    for (int k = 0; k < nbClusters; k++) {
                        count[k] = 0;
                        for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                            sumPerClusterDL[k][jj] = 0.0f;
                        }
                    }

                    int quotient, remainder, offset, length;
                    quotient = nbPoints/nbPackages;
                    remainder = nbPoints%nbPackages;
                    #pragma omp for private(packagesDL) reduction(+: count, sumPerClusterDL)
                    for (int p = 0; p < nbPackages; p++) {
                        offset = (p < remainder ? ((quotient + 1) * p) : (quotient * p + remainder));
                        length = (p < remainder ? (quotient + 1) : quotient);
                        for (int k = 0; k < nbClusters; k++) {
                            for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                                packagesDL[k][jj] = 0.0f;
                            }
                        }
                        for (int i = offset; i < offset + length; i++) {
                            int k = labels[i];
                            count[k]++;
                            index_t idxOffset = ((index_t)i)*((index_t)nbDims) + ((index_t)j);
                            for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                                packagesDL[k][jj] += data[idxOffset + (index_t)jj];
                            }
                        }
                        for (int k = 0; k < nbClusters; k++) {
                            for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                                sumPerClusterDL[k][jj] += packagesDL[k][jj];
                            }
                        }
                    }
                
                    #pragma omp for
                    for (int k = 0; k < nbClusters; k++) {
                        if (count[k] != 0) {
                            for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                                centroids[k*nbDims + j + jj] = sumPerClusterDL[k][jj]/count[k];
                            }
                        }
                    }
                } // End of if (j < nbDims)
                // End of if (nbPoints/nbClusters > tholdUsePackages)
            
            } else { // Without using packages
                int j = 0;
                for ( ; j + NB_DIMS_BATCH_NC < nbDims; j = j + NB_DIMS_BATCH_NC) {
                    #pragma omp for
                    for (int k = 0; k < nbClusters; k++) {
                        count[k] = 0;
                        for (int jj = 0; jj < NB_DIMS_BATCH_NC /* && j + jj < nbDims */; jj++) {
                            sumPerClusterDL[k][jj] = 0.0f;
                        }
                    }

                    // Sum the contributions to each cluster
                    #pragma omp for reduction(+: count, sumPerClusterDL)
                    // 1st step local reduction
                    // - Count nb of instances in OpenMP reduction array
                    // - Reduction in thread private array
                    for (int i = 0; i < nbPoints; i++) {
                        int k = labels[i];
                        count[k]++;
                        index_t idxOffset = ((index_t)i)*((index_t)nbDims) + ((index_t)j);
                        for (int jj = 0; jj < NB_DIMS_BATCH_NC /* && j + jj < nbDims */; jj++) {
                            sumPerClusterDL[k][jj] += data[idxOffset + (index_t)jj];
                        }
                    } // 2nd step global reduction: final reduction by OpenMP in global "sumPerClusterDL" array
                
                    // Final averaging to get new representatives
                    #pragma omp for
                    for (int k = 0; k < nbClusters; k++) {  // Process by cluster
                        if (count[k] != 0) {
                            for (int jj = 0; jj < NB_DIMS_BATCH_NC /* && j + jj < nbDims */; jj++) {
                                centroids[k*nbDims + j + jj] = sumPerClusterDL[k][jj]/count[k];    // - Update global "centroids" array
                            }
                        }
                    }
                } // End of for ( ; j + NB_DIMS_BATCH_NC < nbDims; j = j + NB_DIMS_MIN)
                
                if (j < nbDims) {
                    #pragma omp for
                    for (int k = 0; k < nbClusters; k++) {
                        count[k] = 0;
                        for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                            sumPerClusterDL[k][jj] = 0.0f;
                        }
                    }

                    #pragma omp for reduction(+: count, sumPerClusterDL)
                    for (int i = 0; i < nbPoints; i++) {
                        int k = labels[i];
                        count[k]++;
                        index_t idxOffset = ((index_t)i)*((index_t)nbDims) + ((index_t)j);
                        for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                            sumPerClusterDL[k][jj] += data[idxOffset + (index_t)jj];
                        }
                    }
                
                    #pragma omp for
                    for (int k = 0; k < nbClusters; k++) {
                        if (count[k] != 0) {
                            for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                                centroids[k*nbDims + j + jj] = sumPerClusterDL[k][jj]/count[k];
                            }
                        }
                    }
                } // End of if (j < nbDims)
            } // End of else
            tf_update = omp_get_wtime(); 
            d_update += (tf_update - td_update);


            // Calculate the parameters of iteration control
            #pragma omp single
            {
                changeRatio = (T_real)track / (T_real)nbPoints;
                (*nbIters)++;
                printf("        track = %llu\tchangeRatio = %f\n", track, changeRatio); 
                track = 0; 
            }
            
        } while (changeRatio > tolKMCPU && (*nbIters) < maxNbItersKM);  // Check stopping criteria
        
        // Store the elapsed time in global variables
        #pragma omp single
        {
            Tomp_cpu_computeAssign += d_compute_assign;
            Tomp_cpu_updateCentroids += d_update;
        }
        
        #pragma omp for
        for (int k = 0; k < nbClusters; k++)
            countPerCluster[k] = count[k];
    }
}



void kmeans_cpu_for_extracting_representatives (int nbPoints, int nbDims, int nbReps, T_real *data,
                                                int seedingMethod, unsigned int seedbase,
                                                int tholdUsePackages, int nbPackages,
                                                T_real tolKMCPU, int maxNbItersKM,
                                                int *nbIters, T_real *reps, int *labels)
{
    // Declaration & definition
    unsigned long long int track;  // Number of label changes in two consecutive iterations
    T_real changeRatio;            // changeRatio = track / nbPoints
    printf("    NB_DIMS_LIMITED_BY_NB_REPS = %ld\n", NB_DIMS_LIMITED_BY_NB_REPS);
    printf("    NB_DIMS_BATCH_NR           = %ld\n", NB_DIMS_BATCH_NR);
    #ifdef RS
        T_real sumPerClusterDL[100][2]; // Array for the sum per cluster
        T_real packagesDL[100][2];      // Array for the packages used in UpdateCentroids
        int    count[100];              // Array for the number of data instances in each cluster
    #else
        T_real sumPerClusterDL[NB_REPS][NB_DIMS_BATCH_NR]; // Array for the sum per cluster
        T_real packagesDL[NB_REPS][NB_DIMS_BATCH_NR];      // Array for the packages used in UpdateCentroids
        int    count[NB_REPS];                             // Array for the number of data instances in each cluster
    #endif
    bool flagUsePackages;
    
    // Initialization
    track = 0;
    *nbIters = 0;
    flagUsePackages = (nbPoints/nbReps > tholdUsePackages ? true: false);

    // Initialize representatives
    if (INPUT_INITIAL_CENTROIDS == "") {
        double td_sampling, tf_sampling; 
        td_sampling = omp_get_wtime();
        if (seedingMethod == 1) {
            random_sampling(nbPoints, nbDims, nbReps,  // input
                            data,                      // input
                            seedbase,                  // input
                            reps);                     // output
        }
        if (seedingMethod == 2) {
            d2_sampling(nbPoints, nbDims, nbReps,  // input
                        data,                      // input
                        seedbase,                  // input
                        reps);                     // output
        }
        tf_sampling = omp_get_wtime();
        Tomp_cpu_seeding += (tf_sampling - td_sampling);
    } else {
        read_file_real(reps, nbReps, nbDims, INPUT_INITIAL_CENTROIDS, "\t", 0, 0);
        // read_file_real(reps, nbReps, nbDims, INPUT_INITIAL_CENTROIDS, " ", 0, 0);
    }


    #pragma omp parallel
    {
        // Declaration for each thread
        double td_compute_assign, tf_compute_assign; 
        double td_update, tf_update; 
        double d_compute_assign = 0.0, d_update = 0.0;

        // Iterations
        do {    
            // Compute distances & assign points to clusters
            td_compute_assign = omp_get_wtime();
            #pragma omp for reduction(+: track)
            for (int i = 0; i < nbPoints; i++) {
                // Declaration
                int min = 0;
                T_real sqDist, minDistSq = FLT_MAX;
                
                for (int k = 0; k < NB_REPS; k++) { // Using the constant "NB_REPS" instead of the variable "nbReps" may improve the performance significantly.
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
                } // End of for (int k = 0; k < nbReps; k++)

                // Change the label if necessary and count this change into track
                if (labels[i] != min) {
                    track++;
                    labels[i] = min;
                }
            }  // End of for (int i = 0; i < nbPoints; i++)
            tf_compute_assign = omp_get_wtime();
            d_compute_assign += (tf_compute_assign - td_compute_assign);


            // Update representatives
            td_update = omp_get_wtime();
            if (flagUsePackages) {
                int j = 0;
                for ( ; j + NB_DIMS_BATCH_NR < nbDims; j = j + NB_DIMS_BATCH_NR) {
                    #pragma omp for
                    for (int k = 0; k < nbReps; k++) {
                        count[k] = 0;
                        for (int jj = 0; jj < NB_DIMS_BATCH_NR /* && j + jj < nbDims */; jj++) {
                            sumPerClusterDL[k][jj] = 0.0f;
                        }
                    }

                    // In order to reduce the rounding error which happens when adding numbers of very different magnitudes,
                    // we first divide the dataset into packages, then calculate the sum of points in each packages, finally compute the sum of all packages.
                    int quotient, remainder, offset, length;
                    quotient = nbPoints/nbPackages;
                    remainder = nbPoints%nbPackages;
                    // Sum the contributions to each cluster
                    #pragma omp for private(packagesDL) reduction(+: count, sumPerClusterDL)
                    for (int p = 0; p < nbPackages; p++) {      // Process by packages
                        offset = (p < remainder ? ((quotient + 1) * p) : (quotient * p + remainder));
                        length = (p < remainder ? (quotient + 1) : quotient);
                        // Reset "packages" to zeros
                        for (int k = 0; k < nbReps; k++) {
                            for (int jj = 0; jj < NB_DIMS_BATCH_NR /* && j + jj < nbDims */; jj++) {
                                packagesDL[k][jj] = 0.0f;
                            }
                        }
                        // 1st step local reduction
                        // - Count nb of instances in OpenMP reduction array
                        // - Reduction in thread private array
                        for (int i = offset; i < offset + length; i++) {
                            int k = labels[i];
                            count[k]++;
                            index_t idxOffset = ((index_t)i)*((index_t)nbDims) + ((index_t)j);
                            for (int jj = 0; jj < NB_DIMS_BATCH_NR /* && j + jj < nbDims */; jj++) {
                                packagesDL[k][jj] += data[idxOffset + (index_t)jj];
                            }
                        }
                        // 2nd step local reduction
                        // - Reduction in local OpenMP reduction array
                        for (int k = 0; k < nbReps; k++) {
                            for (int jj = 0; jj < NB_DIMS_BATCH_NR /* && j + jj < nbDims */; jj++) {
                                sumPerClusterDL[k][jj] += packagesDL[k][jj];
                            }
                        }
                    }   // 2nd step global reduction: final reduction by OpenMP in global "sumPerClusterDL" array
                
                    // Final averaging to get new representatives
                    #pragma omp for
                    for (int k = 0; k < nbReps; k++) {  // Process by cluster
                        if (count[k] != 0) {
                            for (int jj = 0; jj < NB_DIMS_BATCH_NR /* && j + jj < nbDims */; jj++) {
                                reps[k*nbDims + j + jj] = sumPerClusterDL[k][jj]/count[k];    // - Calculate global "reps" array
                            }
                        }
                    }
                } // End of for ( ; j + NB_DIMS_BATCH_NR < nbDims; j = j + NB_DIMS_BATCH_NR)
                
                if (j < nbDims) {
                    #pragma omp for
                    for (int k = 0; k < nbReps; k++) {
                        count[k] = 0;
                        for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                            sumPerClusterDL[k][jj] = 0.0f;
                        }
                    }

                    int quotient, remainder, offset, length;
                    quotient = nbPoints/nbPackages;
                    remainder = nbPoints%nbPackages;
                    #pragma omp for private(packagesDL) reduction(+: count, sumPerClusterDL)
                    for (int p = 0; p < nbPackages; p++) {
                        offset = (p < remainder ? ((quotient + 1) * p) : (quotient * p + remainder));
                        length = (p < remainder ? (quotient + 1) : quotient);
                        for (int k = 0; k < nbReps; k++) {
                            for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                                packagesDL[k][jj] = 0.0f;
                            }
                        }
                        for (int i = offset; i < offset + length; i++) {
                            int k = labels[i];
                            count[k]++;
                            index_t idxOffset = ((index_t)i)*((index_t)nbDims) + ((index_t)j);
                            for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                                packagesDL[k][jj] += data[idxOffset + (index_t)jj];
                            }
                        }
                        for (int k = 0; k < nbReps; k++) {
                            for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                                sumPerClusterDL[k][jj] += packagesDL[k][jj];
                            }
                        }
                    }
                
                    #pragma omp for
                    for (int k = 0; k < nbReps; k++) {
                        if (count[k] != 0) {
                            for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                                reps[k*nbDims + j + jj] = sumPerClusterDL[k][jj]/count[k];
                            }
                        }
                    }
                } // End of if (j < nbDims)
                // End of if (nbPoints/nbReps > tholdUsePackages)
            
            } else { // Without using packages
                int j = 0;
                for ( ; j + NB_DIMS_BATCH_NR < nbDims; j = j + NB_DIMS_BATCH_NR) {
                    #pragma omp for
                    for (int k = 0; k < nbReps; k++) {
                        count[k] = 0;
                        for (int jj = 0; jj < NB_DIMS_BATCH_NR /* && j + jj < nbDims */; jj++) {
                            sumPerClusterDL[k][jj] = 0.0f;
                        }
                    }

                    // Sum the contributions to each cluster
                    #pragma omp for reduction(+: count, sumPerClusterDL)
                    // 1st step local reduction
                    // - Count nb of instances in OpenMP reduction array
                    // - Reduction in thread private array
                    for (int i = 0; i < nbPoints; i++) {
                        int k = labels[i];
                        count[k]++;
                        index_t idxOffset = ((index_t)i)*((index_t)nbDims) + ((index_t)j);
                        for (int jj = 0; jj < NB_DIMS_BATCH_NR /* && j + jj < nbDims */; jj++) {
                            sumPerClusterDL[k][jj] += data[idxOffset + (index_t)jj];
                        }
                    } // 2nd step global reduction: final reduction by OpenMP in global "sumPerClusterDL" array
                
                    // Final averaging to get new representatives
                    #pragma omp for
                    for (int k = 0; k < nbReps; k++) {  // Process by cluster
                        if (count[k] != 0) {
                            for (int jj = 0; jj < NB_DIMS_BATCH_NR /* && j + jj < nbDims */; jj++) {
                                reps[k*nbDims + j + jj] = sumPerClusterDL[k][jj]/count[k];    // - Update global "reps" array
                            }
                        }
                    }
                } // End of for ( ; j + NB_DIMS_BATCH_NR < nbDims; j = j + NB_DIMS_BATCH_NR)
                
                if (j < nbDims) {
                    #pragma omp for
                    for (int k = 0; k < nbReps; k++) {
                        count[k] = 0;
                        for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                            sumPerClusterDL[k][jj] = 0.0f;
                        }
                    }

                    #pragma omp for reduction(+: count, sumPerClusterDL)
                    for (int i = 0; i < nbPoints; i++) {
                        int k = labels[i];
                        count[k]++;
                        index_t idxOffset = ((index_t)i)*((index_t)nbDims) + ((index_t)j);
                        for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                            sumPerClusterDL[k][jj] += data[idxOffset + (index_t)jj];
                        }
                    }
                
                    #pragma omp for
                    for (int k = 0; k < nbReps; k++) {
                        if (count[k] != 0) {
                            for (int jj = 0; j + jj < NB_DIMS; jj++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                                reps[k*nbDims + j + jj] = sumPerClusterDL[k][jj]/count[k];
                            }
                        }
                    }
                } // End of if (j < nbDims)
            } // End of else
            tf_update = omp_get_wtime(); 
            d_update += (tf_update - td_update);


            // Calculate the parameters of iteration control
            #pragma omp single
            {
                changeRatio = (T_real)track / (T_real)nbPoints;
                (*nbIters)++;
                printf("        track = %llu  changeRatio = %f\n", track, changeRatio); 
                track = 0; 
            }
            
        } while (changeRatio > tolKMCPU && (*nbIters) < maxNbItersKM);  // Check stopping criteria
        
        // Store the elapsed time in global variables
        #pragma omp single
        {
            Tomp_cpu_computeAssign += d_compute_assign;
            Tomp_cpu_updateCentroids += d_update;
        }
    }
    
    // Save results of count
    // save_file_int(&count[0], nbReps, 1, "output/CountAttachedPointsPerRep.txt", "", 0);
}
