#include <stdio.h>   // FILE, stderr, stdin, stdout, fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush
#include <stdlib.h>  // EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX, atof, atoi, malloc, free, exit, rand
#include <float.h>   // FLT_MAX, FLT_MIN
#include <math.h>    // exp, expf, pow, powf, log, logf, sqrt, sqrtf, ceil
#include <omp.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <thrust/device_vector.h>   // thrust::device_ptr
#include <thrust/extrema.h>         // thrust::max_element
#include <thrust/sort.h>            // thrust::sort
#include <thrust/functional.h>      // thrust::greater<int>()

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/utilities/feature_scaling_gpu.h"
#include "../../include/spectral_clustering/auto_tuning.h"



/* Auto-tune the number of clusters */

__global__ void kernel_compute_eigengap_vector (int maxNbClusters, T_real *GPU_eigVals,
                                                T_real *GPU_eigGaps)
{
    // 1D block in x-axis, 1D grid in x-axis
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < (maxNbClusters - 1)) {
        GPU_eigGaps[tid] = GPU_eigVals[tid + 1] - GPU_eigVals[tid];
    }
}


void auto_tune_nb_clusters_based_on_eigengaps (int flagInteractive,
                                               int maxNbClusters, T_real *GPU_eigVals,
                                               int *optNbClusters, double *timeUserResponse)
{
    dim3 Dg, Db;
    T_real *GPU_eigGaps;
    
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_eigGaps, sizeof(T_real)*(maxNbClusters - 1)));
    
    // Compute eigengaps
    Db.x = BsXC;
    Db.y = 1;
    Dg.x = (maxNbClusters-1)/Db.x + ((maxNbClusters-1)%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    kernel_compute_eigengap_vector<<<Dg,Db>>>(maxNbClusters, GPU_eigVals, GPU_eigGaps);
    CHECK_CUDA_SUCCESS(cudaPeekAtLastError());
    
    T_real *eigVals;
    eigVals = (T_real *) malloc(sizeof(T_real)*maxNbClusters);
    CHECK_CUDA_SUCCESS(cudaMemcpy(eigVals, GPU_eigVals, sizeof(T_real)*maxNbClusters, cudaMemcpyDeviceToHost));
    
    // Estimate the optimal number of clusters by times between consecutive eigenvalues
    T_real times;
    T_real maxTimes = 1.0f;
    int optNbClustersByTimes;
    for (int i = maxNbClusters - 1; i > 0; i--) {
        if (eigVals[i - 1] > 1E-6f) {
            times = eigVals[i] / eigVals[i - 1];
            if (times > maxTimes) {
                maxTimes = times;
                optNbClustersByTimes = i;
            }
        } else {
            break;
        }
    }
    
    // Estimate the optimal number of clusters by maximal eigGap
    thrust::device_ptr<T_real> d_eigGaps(GPU_eigGaps);
    thrust::device_ptr<T_real> max_ptr = thrust::max_element(d_eigGaps, d_eigGaps + maxNbClusters - 1);
    unsigned int position = max_ptr - d_eigGaps;
    int optNbClustersByMaxEigGap = position + 1;
    CHECK_CUDA_SUCCESS(cudaFree(GPU_eigGaps));
    
    // Visualize the comparison of eigenvalues
    printf("    The first %d eigenvalues of graph Laplacian:\n", maxNbClusters);
    T_real maxEigVal = eigVals[maxNbClusters - 1];
    for (int i = 0; i < maxNbClusters; i++) {
        int count = (eigVals[i] / maxEigVal) * 20.0f;
        if (i < 9) {
            printf("    eigenvale %d  |", i + 1);
        } else {
            printf("    eigenvale %d |", i + 1);
        }
        for (int j = 0; j < count; j++) {
            printf("*");
        }
        printf(" %.2e\n", eigVals[i]);
    }

    if (flagInteractive) {
        printf("    Please determine the number of clusters (recommend %d or %d): ", optNbClustersByTimes, optNbClustersByMaxEigGap);
        double begin, finish;
        begin = omp_get_wtime();
        scanf("%d", optNbClusters);
        while (*optNbClusters < 1 || *optNbClusters > maxNbClusters) {
            fprintf(stderr,"    Error: the number of clusters has to be in [1, %d]!\n", maxNbClusters);
            printf("    Please enter a valid number: ");
            scanf("%d", optNbClusters);
        }
        finish = omp_get_wtime();
        *timeUserResponse += (finish - begin);
    } else {
        *optNbClusters = optNbClustersByMaxEigGap;
        printf("    Auto-determined number of clusters based on eigengaps: %d\n", *optNbClusters);
    }
    
    free(eigVals);
}


void save_decision_graph (int nbPoints, int *GPU_density, T_real *GPU_delta)
{
    int *density;
    T_real *delta;
    density = (int *) malloc(sizeof(int)*nbPoints);
    delta = (T_real *) malloc(sizeof(T_real)*nbPoints);
    CHECK_CUDA_SUCCESS(cudaMemcpy(density, GPU_density, sizeof(int)*nbPoints, cudaMemcpyDeviceToHost));
    CHECK_CUDA_SUCCESS(cudaMemcpy(delta, GPU_delta, sizeof(T_real)*nbPoints, cudaMemcpyDeviceToHost));
    FILE *fp; // File pointer
    fp = fopen("output/DecisionGraph.txt", "w");
    if (fp == NULL) {
        printf("    Fail to open file!\n");
        exit(0);
    }
    for (int i = 0; i < nbPoints; i++) {
        fprintf(fp, "%d\t%f\n", density[i], delta[i]);
    }
    fclose(fp);
    free(density);
    free(delta);
}


void save_decision_graph_and_determine_nb_clusters (int nbPoints,
                                                    int *GPU_density, T_real *GPU_delta,
                                                    int *optNbClusters, double *timeUserResponse)
{
    int *density;
    T_real *delta;
    density = (int *) malloc(sizeof(int)*nbPoints);
    delta = (T_real *) malloc(sizeof(T_real)*nbPoints);
    CHECK_CUDA_SUCCESS(cudaMemcpy(density, GPU_density, sizeof(int)*nbPoints, cudaMemcpyDeviceToHost));
    CHECK_CUDA_SUCCESS(cudaMemcpy(delta, GPU_delta, sizeof(T_real)*nbPoints, cudaMemcpyDeviceToHost));
    FILE *fp;
    fp = fopen("output/DecisionGraph.txt", "w");
    if (fp == NULL) {
        printf("    Fail to open file!\n");
        exit(0);
    }
    for (int i = 0; i < nbPoints; i++) {
        fprintf(fp, "%d\t%f\n", density[i], delta[i]);
    }
    fclose(fp);
    free(density);
    free(delta);
    
    // Get the number of clusters by user input
    printf("    The decision graph has been saved into a text file.\n");
    printf("    Please determine the number of clusters based on decision graph: ");
    double begin, finish;
    begin = omp_get_wtime();
    scanf("%d", optNbClusters);
    while (*optNbClusters < 1 || *optNbClusters > nbPoints) {
        fprintf(stderr,"    Error: the number of clusters has to be in [1, %d]!\n", nbPoints);
        printf("    Please enter a valid number: ");
        scanf("%d", optNbClusters);
    }
    finish = omp_get_wtime();
    *timeUserResponse += (finish - begin);
}
