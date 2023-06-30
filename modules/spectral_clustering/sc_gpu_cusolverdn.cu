#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cusolverDn.h>   // CUSOLVERDN_SYEVDX
#include <omp.h>          // omp_get_wtime

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/kmeans/kmeans_gpu.h"
#include "../../include/spectral_clustering/constr_sim_lap_matrix_in_dense.h"
#include "../../include/spectral_clustering/normalize_eigvect_mat.h"
#include "../../include/spectral_clustering/auto_tuning.h"
#include "../../include/spectral_clustering/sc_gpu_cusolverdn.h"


void spectral_clustering_on_gpu_involving_cusolverdn (int nbPoints, int nbDims, int nbClusters, T_real *GPU_dataT,
                                                      T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                      int flagAutoTuneNbClusters, int maxNbClusters, int flagInteractive,
                                                      int seedingMethod, unsigned int seedBase, T_real tolKMGPU, int maxNbItersKM,
                                                      int tholdUsePackages, int nbPackages, int nbStreamsStep1, int nbStreamsStep2,
                                                      int *nbItersKM, int *optNbClusters, int *GPU_count, int *GPU_labels)
{
    // Declaration
    double begin, finish;
    
    // Construct affinity matrix, degree matrix and Laplacian matrix
    T_real *GPU_sim;       // Array for similarity matrix
    T_real *GPU_deg;       // Array for the diagonal degree matrix
    T_real *GPU_lap;       // Array for Laplacian matrix
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_sim, (sizeof(T_real)*nbPoints)*nbPoints));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_deg, sizeof(T_real)*nbPoints));
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_deg, 0, sizeof(T_real)*nbPoints));
    GPU_lap = GPU_sim;     // Use the same memory space for storing similarity matrix and Laplacian matrix
    printf("    Similarity & Laplacian matrix construction begins ...\n");
    begin = omp_get_wtime();
    construct_similarity_degree_matrix(nbPoints, nbDims, GPU_dataT,    // input
                                       sigma, tholdSim, tholdSqDist,   // input
                                       GPU_sim, GPU_deg);              // output
    compute_laplacian_matrix(nbPoints, GPU_sim, GPU_deg,    // input
                             GPU_lap);                      // output
    finish = omp_get_wtime();
    Tomp_gpu_constructSimLapMatrix += (finish - begin);
    printf("    Similarity & Laplacian matrix construction completed!\n");

    // Memory deallocation
    CHECK_CUDA_SUCCESS(cudaFree(GPU_deg));
    
    
    // Compute eigenvalues & eigenvectors of Laplacian matrix using cuSolverDN library
    printf("    Eigenvectors calculation by cuSOLVERDN begins ...\n");
    begin = omp_get_wtime();
    
    // Declaration & initialization
    cusolverDnHandle_t handleCUSOLVERDN;  // Handle for cuSolverDN library
    CHECK_CUSOLVER_SUCCESS(cusolverDnCreate(&handleCUSOLVERDN));
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
    cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;    // the il-th through iu-th eigenvalues/eigenvectors will be found
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    T_real vl = 0.0f;  // if range = CUSOLVER_EIG_RANGE_V, then all eigenvalues/eigenvectors in the half-open interval (vl,vu] will be found
    T_real vu = 1.0f;  // Not referenced if range = CUSOLVER_EIG_RANGE_ALL or range = CUSOLVER_EIG_RANGE_I.
    int il = 1;
    int iu = nbClusters;
    int *h_meig;          // Output the number of eigenvalues/eigenvectors computed by the routine (h_meig = iu - il + 1)
    int *lwork;
    T_real *GPU_work;     // Array for working space of cuSolverDN
    int    *GPU_devInfo;  // Array for devInfo when using cuSolverDN
    T_real *GPU_eigVals;  // Array for eigenvalues calculated by cuSolverDN
    h_meig = (int *)malloc(sizeof(int));
    lwork = (int *)malloc(sizeof(int));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_eigVals, sizeof(T_real)*nbPoints)); // Must be a real array of dimension n (see the function doc)
    *lwork = 0;
    
    // Query working space of syevd
    CHECK_CUSOLVER_SUCCESS(CUSOLVERDN_SYEVDX_BUFFERSIZE(
                           handleCUSOLVERDN,
                           jobz,
                           range,
                           uplo,
                           nbPoints,
                           GPU_lap,
                           nbPoints,
                           vl, vu,
                           il, iu,
                           h_meig,
                           GPU_eigVals,
                           lwork));
    
    // printf("    lwork = %d\n", *lwork);    // Attention! There might be integer overflow for lwork.
    
    // Memory allocation
    CHECK_CUDA_SUCCESS(cudaMalloc((void**)&GPU_work, sizeof(T_real)*(*lwork)));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_devInfo, sizeof(int)));
    
    // The syevdx function computes all or selection of the eigenvalues and optionally eigenvectors of a symmetric (Hermitian) n×n matrix A.
    // On exit, if jobz = CUSOLVER_EIG_MODE_VECTOR, and GPU_devInfo = 0, A contains the orthonormal eigenvectors of the matrix A.
    CHECK_CUSOLVER_SUCCESS(CUSOLVERDN_SYEVDX(  // The dense matrices are assumed to be stored in column-major order in memory.
                           handleCUSOLVERDN,
                           jobz,
                           range,
                           uplo,
                           nbPoints,
                           GPU_lap,
                           nbPoints,
                           vl, vu,
                           il, iu,
                           h_meig,
                           GPU_eigVals,
                           GPU_work, 
                           *lwork,
                           GPU_devInfo));
    
    // printf("    h_meig = %d\n", *h_meig);
    // If devInfo = 0, the operation is successful.
    // If devInfo = -i, the i-th parameter is wrong (not counting handle).
    // If devInfo = i (> 0), devInfo indicates i off-diagonal elements of an intermediate tridiagonal form did not converge to zero.
    // int devInfo;
    // CHECK_CUDA_SUCCESS(cudaMemcpy(&devInfo, GPU_devInfo, sizeof(int), cudaMemcpyDeviceToHost)); 
    // printf("    devInfo = %d\n", devInfo);
    
    // Destroy handle
    CHECK_CUSOLVER_SUCCESS(cusolverDnDestroy(handleCUSOLVERDN));
    
    // Memory deallocation
    CHECK_CUDA_SUCCESS(cudaFree(GPU_work));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_devInfo));
    free(h_meig);
    free(lwork);
    
    finish = omp_get_wtime();
    Tomp_gpu_cuSolverDNsyevdx += (finish - begin);
    printf("    Eigenvectors calculation by cuSOLVERDN completed!\n");
    
        // Save eigenvalues and eigenvectors of Laplacian matrix found by cuSolverDN
        // T_real *eigval_dn;
        // T_real *eigvect_dn;
        // eigval_dn = (T_real *) malloc(sizeof(T_real)*nbClusters);
        // eigvect_dn = (T_real *) malloc((sizeof(T_real)*nbClusters)*nbPoints);
        // CHECK_CUDA_SUCCESS(cudaMemcpy(eigval_dn, GPU_eigVals, sizeof(T_real)*nbClusters, cudaMemcpyDeviceToHost));   // Transfer nbClusters eigenvalues back to host
        // CHECK_CUDA_SUCCESS(cudaMemcpy(eigvect_dn, GPU_lap, (sizeof(T_real)*nbClusters)*nbPoints, cudaMemcpyDeviceToHost));  // Transfer nbClusters eigenvectors back to host
        // save_file_real(eigval_dn,  nbClusters, 1,        "output/Eigenvalues.txt",  "");
        // save_file_real(eigvect_dn, nbPoints, nbClusters, "output/Eigenvectors.txt", "\t");  // the order of "nbClusters" and "nbPoints" is changed, since there will be buffer overflow due to too long length of row (nbPoints)
        // free(eigval_dn);
        // free(eigvect_dn);
    
    // Auto-tuning of the nb of clusters based on eigengaps
    if (flagAutoTuneNbClusters == 1) {
        printf("    Auto-tuning of the nb of clusters begins ...\n");
        begin = omp_get_wtime();
        double timeUserResponse = 0.0;
        auto_tune_nb_clusters_based_on_eigengaps(flagInteractive,
                                                 maxNbClusters, GPU_eigVals,         // input
                                                 optNbClusters, &timeUserResponse);  // output
        nbClusters = *optNbClusters;
        finish = omp_get_wtime();
        Tomp_gpu_autoTuneNbClusters += (finish - begin - timeUserResponse);
        printf("    Auto-tuning of the nb of clusters completed!\n");
    }
    
    // Memory deallocation
    CHECK_CUDA_SUCCESS(cudaFree(GPU_eigVals));
    
    
    // Normalize eigenvector matrix
    T_real *GPU_eigVects;    // Array for eigenvector matrix
    GPU_eigVects = GPU_lap;  // Use the same memory space for storing Laplacian matrix and eigenvector matrix
    printf("    Eigenvector matrix normalization begins ...\n");
    begin = omp_get_wtime();
    normalize_eigenvector_matrix(nbPoints, nbClusters,  // input
                                 GPU_eigVects);         // input & output
    finish = omp_get_wtime();
    Tomp_gpu_normalizeEigenvectorMatrix += (finish - begin);
    printf("    Eigenvector matrix normalization completed!\n");
    
    
    // Final k-means(++) clustering
    printf("    Final k-means(++) clustering begins ...\n");
    begin = omp_get_wtime();
    
    // Set the # of dims equal to the # of clusters
    nbDims = nbClusters;
    
    // Declaration & memory allocation
    T_real *GPU_centroidsEigMat;
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centroidsEigMat, (sizeof(T_real)*nbClusters)*nbDims));
    
    // Call GPU k-means(++) function
    kmeans_gpu(SC_GPU,                                                        // input (env)
               nbPoints, nbDims, nbClusters, GPU_eigVects,                    // input
               seedingMethod, seedBase, tolKMGPU, maxNbItersKM,               // input 
               tholdUsePackages, nbPackages, nbStreamsStep1, nbStreamsStep2,  // input
               nbItersKM, GPU_count, GPU_centroidsEigMat, GPU_labels);        // output
    
    // Memory deallocation
    CHECK_CUDA_SUCCESS(cudaFree(GPU_centroidsEigMat));
    CHECK_CUDA_SUCCESS(cudaFree(GPU_eigVects));  // equal to cudaFree(GPU_sim) and cudaFree(GPU_lap)
    
    finish = omp_get_wtime();
    Tomp_gpu_finalKmeansForSC += (finish - begin);
    printf("    Final k-means(++) clustering completed!\n");
}