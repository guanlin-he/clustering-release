#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>
#include <omp.h>     // omp_get_wtime
#include <thrust/device_vector.h>     // thrust::device_ptr
#include <thrust/execution_policy.h>  // thrust::device
#include <thrust/extrema.h>           // thrust::max_element, thrust::min_element, thrust::minmax_element
#include <thrust/fill.h>              // thrust::fill

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/kmeans/kmeans_gpu.h"
#include "../../include/spectral_clustering/constr_epsilon_sim_matrix_in_csr.h"
#include "../../include/spectral_clustering/constr_epsilon_sim_matrix_in_dense_to_csr.h"
#include "../../include/spectral_clustering/get_edge_list.h"
#include "../../include/spectral_clustering/filter_noise.h"
#include "../../include/spectral_clustering/nvgraph_api.h"
#include "../../include/spectral_clustering/normalize_eigvect_mat.h"
#include "../../include/spectral_clustering/auto_tuning.h"
#include "../../include/spectral_clustering/sc_gpu_nvgraph_km.h"



void spectral_clustering_on_gpu_involving_nvgraph_and_kmeans (int nbPoints, int nbDims, int nbClusters, T_real *GPU_dataT,
                                                              T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                              int csrAlgo, int hypoMaxNnzRow, T_real maxNzPercent,
                                                              T_real memUsePercent, int pad1, int pad2, int pad3,
                                                              int filterNoiseApproach, int nbBinsHist, T_real tholdNoise,
                                                              int nvGraphAlgo, T_real tolEigen, int maxNbItersEigen,
                                                              int flagAutoTuneNbClusters, int maxNbClusters, int flagInteractive,
                                                              int seedingMethod, unsigned int seedBase, T_real tolKMGPU, int maxNbItersKM,
                                                              int tholdUsePackages, int nbPackages, int nbStreamsStep1, int nbStreamsStep2,
                                                              float *modularityScore, float *edgeCutScore, float *ratioCutScore,
                                                              int *optNbClusters, int *nbItersKM, int *GPU_count, int *GPU_labels)
{
    // Declaration
    double begin, finish;
    int nnzS;               // Total number of nonzero elements in similarity matrix
    int minNnzRowS;         // Minimal number of nonzero elements in a row of similarity matrix
    int maxNnzRowS;         // Maximal number of nonzero elements in a row of similarity matrix
    int    *GPU_nnzPerRowS; // Array for nnzPerRow of similarity matrix
    T_real *GPU_csrValS;    // Array for csrVal of similarity matrix
    int    *GPU_csrRowS;    // Array for csrRow of similarity matrix
    int    *GPU_csrColS;    // Array for csrCol of similarity matrix
    
    // Memory allocation
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_nnzPerRowS, sizeof(int)*(nbPoints + 1)));
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_csrRowS, sizeof(int)*(nbPoints + 1)));
    
    // Construct affinity matrix directly in CSR format
    printf("    Similarity matrix construction in CSR format begins ...\n");
    begin = omp_get_wtime();
    switch (csrAlgo) {
        case 1 :  
            algo_CSR1_for_similarity_matrix_construction(nbPoints, nbDims, GPU_dataT,                      // input
                                                         sigma, tholdSim, tholdSqDist,                     // input
                                                         GPU_nnzPerRowS, &minNnzRowS, &maxNnzRowS, &nnzS,  // output
                                                         &GPU_csrValS, &GPU_csrColS, GPU_csrRowS);         // output
            break;
            
        case 2 :  
            algo_CSR2_for_similarity_matrix_construction(nbPoints, nbDims, GPU_dataT,                      // input
                                                         sigma, tholdSim, tholdSqDist,                     // input
                                                         hypoMaxNnzRow, pad1, pad2,                        // input
                                                         GPU_nnzPerRowS, &minNnzRowS, &maxNnzRowS, &nnzS,  // output
                                                         &GPU_csrValS, &GPU_csrColS, GPU_csrRowS);         // output
            break;
            
        case 3 :  
            algo_CSR3_for_similarity_matrix_construction(nbPoints, nbDims, GPU_dataT,                      // input
                                                         sigma, tholdSim, tholdSqDist,                     // input
                                                         memUsePercent, maxNzPercent,                      // input
                                                         &nnzS, &GPU_csrValS, &GPU_csrColS, GPU_csrRowS);  // output
            if (filterNoiseApproach != 0) {
                compute_nnz_per_row_and_min_max_nnz_row_based_on_csrrow(nbPoints,
                                                                        GPU_csrRowS,
                                                                        GPU_nnzPerRowS, 
                                                                        &minNnzRowS, &maxNnzRowS);
            }
            break;
            
        default: 
            fprintf(stderr, "Unknown algorithm for constructing similarity matrix in CSR format!\n");
            exit(EXIT_FAILURE);
    }
    finish = omp_get_wtime();
    Tomp_gpu_constructSimMatrixInCSR += (finish - begin);
    printf("    Similarity matrix construction in CSR format completed!\n");
    
    
    // Get the edge list
    // printf("    Edge list output begins ...\n");
    // get_edge_list_from_csr(nbPoints, nnzS,                          // input
                           // GPU_csrRowS, GPU_csrColS, GPU_csrValS);  // input
    // printf("    Edge list output completed!\n");


    if (filterNoiseApproach != 0) {  // With noise filtering
    
        // Filter noise based on sparse similarity matrix
        printf("    Noise filtering begins ...\n");
        begin = omp_get_wtime();
        
        double timeUserResponse = 0.0;
        int nbPointsNF;
        int nnzSNF;
        int *GPU_nnzPerRowSNF;
        int *GPU_csrRowSNF;
        int *GPU_csrColSNF;
        T_real *GPU_csrValSNF;
        int *GPU_isNoise;
        int *GPU_nbNoiseFront;
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_isNoise, sizeof(int)*nbPoints));
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_nbNoiseFront, sizeof(int)*nbPoints));
        CHECK_CUDA_SUCCESS(cudaMemset(GPU_labels, 0, sizeof(int)*nbPoints));
        
        switch (filterNoiseApproach) {
            case 1 : 
                filter_noise_based_on_nnz_per_row(nbPoints, 
                                                                  GPU_nnzPerRowS, minNnzRowS, maxNnzRowS, nnzS,
                                                                  GPU_csrRowS, GPU_csrColS, GPU_csrValS,
                                                                  nbBinsHist, flagInteractive, tholdNoise,
                                                                  &nbPointsNF, 
                                                                  &GPU_nnzPerRowSNF, &nnzSNF, 
                                                                  &GPU_csrRowSNF, &GPU_csrColSNF, &GPU_csrValSNF,
                                                                  GPU_isNoise, GPU_nbNoiseFront,
                                                                  GPU_labels, &timeUserResponse);
                break;
                
            case 2 : 
                filter_noise_based_on_vertex_degree(nbPoints,
                                                                  GPU_nnzPerRowS, nnzS,
                                                                  GPU_csrRowS, GPU_csrColS, GPU_csrValS,
                                                                  nbBinsHist, flagInteractive, tholdNoise,
                                                                  &nbPointsNF, 
                                                                  &GPU_nnzPerRowSNF, &nnzSNF, 
                                                                  &GPU_csrRowSNF, &GPU_csrColSNF, &GPU_csrValSNF,
                                                                  GPU_isNoise, GPU_nbNoiseFront,
                                                                  GPU_labels, &timeUserResponse);
                break;
                
            default: 
                fprintf(stderr, "Unknown approach for filtering noise!\n");
                exit(EXIT_FAILURE);
        }
        
        int nbClustersNF = nbClusters - 1;
        int maxNbClustersNF = maxNbClusters - 1;
        
        finish = omp_get_wtime();
        Tomp_gpu_filterNoise += (finish - begin - timeUserResponse);
        printf("    Noise filtering completed!\n");
        
        CHECK_CUDA_SUCCESS(cudaFree(GPU_nnzPerRowSNF));  // equal to cudaFree(GPU_nnzPerRowS)
        
        
        // nvGRAPH spectral graph partitioning on noise-free points
        printf("    nvGRAPH spectral graph partitioning begins ...\n");
        begin = omp_get_wtime();
        int *GPU_labelsNF;
        T_real *GPU_eigValsNF;
        T_real *GPU_eigVectsNF;
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_labelsNF, sizeof(int)*nbPointsNF));
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_eigValsNF, sizeof(T_real)*nbClustersNF));
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_eigVectsNF, (sizeof(T_real)*nbPointsNF)*nbClustersNF));
        similarity_graph_partitioning_via_nvgraph(nbPointsNF, nbClustersNF, nnzSNF,              // input 
                                                  GPU_csrRowSNF, GPU_csrColSNF, GPU_csrValSNF,   // input
                                                  nvGraphAlgo, tolEigen, maxNbItersEigen,        // input 
                                                  tolKMGPU, maxNbItersKM,                        // input 
                                                  GPU_eigValsNF, GPU_eigVectsNF, GPU_labelsNF);  // output
        finish = omp_get_wtime();
        Tomp_gpu_nvGRAPHSpectralClusteringAPI += (finish - begin);
        printf("    nvGRAPH spectral graph partitioning completed!\n");
        
        // Auto-tuning of the nb of clusters based on eigengaps
        if (flagAutoTuneNbClusters == 1) {
            printf("    Auto-tuning of the nb of clusters begins ...\n");
            begin = omp_get_wtime();
            double timeUserResponse = 0.0;
            int optNbClustersNF;
            auto_tune_nb_clusters_based_on_eigengaps(flagInteractive, 
                                                     maxNbClustersNF, GPU_eigValsNF,        // input
                                                     &optNbClustersNF, &timeUserResponse);  // output
            nbClustersNF = optNbClustersNF;
            *optNbClusters = optNbClustersNF + 1;
            finish = omp_get_wtime();
            Tomp_gpu_autoTuneNbClusters += (finish - begin - timeUserResponse);
            printf("    Auto-tuning of the nb of clusters completed!\n");
        }
        
        // Memory deallocation
        CHECK_CUDA_SUCCESS(cudaFree(GPU_eigValsNF));


        // Normalize eigenvector matrix
        printf("    Eigenvector matrix normalization begins ...\n");
        begin = omp_get_wtime();
        T_real *GPU_eigVectsTruncT;   // Array for eigenvectors truncated and transposed
        T_real alpha = 1.0f;
        T_real beta = 0.0f;
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_eigVectsTruncT, (sizeof(T_real)*nbPointsNF)*nbClustersNF));
        
        // Transpose GPU_eigVects into GPU_eigVectsTruncT
        CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(handleCUBLAS,                
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             nbPointsNF, nbClustersNF,
                             &alpha, GPU_eigVectsNF, maxNbClustersNF,
                             &beta, NULL, nbPointsNF,
                             GPU_eigVectsTruncT, nbPointsNF)); 
        
        // Memory deallocation
        CHECK_CUDA_SUCCESS(cudaFree(GPU_eigVectsNF));
        
        // Normalize eigenvector matrix
        normalize_eigenvector_matrix(nbPointsNF, nbClustersNF,  // input
                                     GPU_eigVectsTruncT);       // input & output
        
        finish = omp_get_wtime();
        Tomp_gpu_normalizeEigenvectorMatrix += (finish - begin);
        printf("    Eigenvector matrix normalization completed!\n");
        
        
        // Final k-means(++) clustering
        printf("    Final k-means(++) clustering begins ...\n");
        begin = omp_get_wtime();
        
        // Declaration
        T_real *GPU_centroidsEigMat;
        
        // Memory allocation
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centroidsEigMat, (sizeof(T_real)*nbClustersNF)*nbClustersNF));
        
        // Call GPU k-means(++) function
        kmeans_gpu(SC_GPU,                                                        // input (env)
                   nbPointsNF, nbDims, nbClustersNF, GPU_eigVectsTruncT,          // input
                   seedingMethod, seedBase, tolKMGPU, maxNbItersKM,               // input 
                   tholdUsePackages, nbPackages, nbStreamsStep1, nbStreamsStep2,  // input
                   nbItersKM, GPU_count, GPU_centroidsEigMat, GPU_labelsNF);      // output
        
        // Memory deallocation
        CHECK_CUDA_SUCCESS(cudaFree(GPU_centroidsEigMat));
        CHECK_CUDA_SUCCESS(cudaFree(GPU_eigVectsTruncT));
        
        finish = omp_get_wtime();
        Tomp_gpu_finalKmeansForSC += (finish - begin);
        printf("    Final k-means(++) clustering completed!\n");
        
        
        // nvGRAPH clustering quality evaluation
        printf("    nvGRAPH clustering analysis begins ...\n");
        evaluate_clustering_quality_via_nvgraph(nbPointsNF, nbClustersNF, nnzSNF,               // input 
                                                GPU_csrRowSNF, GPU_csrColSNF, GPU_csrValSNF,    // input
                                                GPU_labelsNF,                                   // input
                                                modularityScore, edgeCutScore, ratioCutScore);  // output
        printf("    nvGRAPH clustering analysis completed!\n");
        
        // Memory deallocation
        CHECK_CUDA_SUCCESS(cudaFree(GPU_csrRowSNF));  // equal to cudaFree(GPU_csrRowS)
        CHECK_CUDA_SUCCESS(cudaFree(GPU_csrColSNF));  // equal to cudaFree(GPU_csrColS)
        CHECK_CUDA_SUCCESS(cudaFree(GPU_csrValSNF));  // equal to cudaFree(GPU_csrValS)
        
        // Merge noise-free labels and noise labels
        merge_labels(nbPoints, nbPointsNF, 
                     GPU_isNoise, GPU_nbNoiseFront,
                     GPU_labelsNF,
                     GPU_labels);
        
        CHECK_CUDA_SUCCESS(cudaFree(GPU_isNoise)); 
        CHECK_CUDA_SUCCESS(cudaFree(GPU_nbNoiseFront)); 
        CHECK_CUDA_SUCCESS(cudaFree(GPU_labelsNF));
        
    } else { // Without noise filtering
        
        CHECK_CUDA_SUCCESS(cudaFree(GPU_nnzPerRowS));
        
        // nvGRAPH spectral graph partitioning
        printf("    nvGRAPH spectral graph partitioning begins ...\n");
        begin = omp_get_wtime();
        // Declaration
        T_real *GPU_eigVals;
        T_real *GPU_eigVects;
        
        // Memory allocation
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_eigVals, sizeof(T_real)*nbClusters));
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_eigVects, (sizeof(T_real)*nbPoints)*nbClusters));
        
        // Function call
        similarity_graph_partitioning_via_nvgraph(nbPoints, nbClusters, nnzS,                    // input 
                                                  GPU_csrRowS, GPU_csrColS, GPU_csrValS,         // input
                                                  nvGraphAlgo, tolEigen, maxNbItersEigen,        // input 
                                                  tolKMGPU, 1,                                   // input 
                                                  GPU_eigVals, GPU_eigVects, GPU_labels);        // output
        finish = omp_get_wtime();
        Tomp_gpu_nvGRAPHSpectralClusteringAPI += (finish - begin);
        printf("    nvGRAPH spectral graph partitioning completed!\n");
        
        
        printf("    nvGRAPH clustering analysis begins ...\n");
        evaluate_clustering_quality_via_nvgraph(nbPoints, nbClusters, nnzS,                    // input 
                                                GPU_csrRowS, GPU_csrColS, GPU_csrValS,         // input
                                                GPU_labels,                                    // input
                                                modularityScore, edgeCutScore, ratioCutScore); // output
        printf("    nvGRAPH clustering analysis completed!\n");
        
        
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
        printf("    Eigenvector matrix normalization begins ...\n");
        begin = omp_get_wtime();
        
        // Declaration & initialization
        T_real *GPU_eigVectsTruncT;   // Array for eigenvectors truncated and transposed
        T_real alpha = 1.0f;
        T_real beta = 0.0f;
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_eigVectsTruncT, (sizeof(T_real)*nbPoints)*nbClusters));

        // Transpose GPU_eigVects into GPU_eigVectsTruncT
        CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(handleCUBLAS,                
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             nbPoints, nbClusters,
                             &alpha, GPU_eigVects, maxNbClusters,
                             &beta, NULL, nbPoints,
                             GPU_eigVectsTruncT, nbPoints)); 
        
        // Memory deallocation
        CHECK_CUDA_SUCCESS(cudaFree(GPU_eigVects));
        
        // Normalize eigenvector matrix
        normalize_eigenvector_matrix(nbPoints, nbClusters,  // input
                                     GPU_eigVectsTruncT);   // input & output
        
        finish = omp_get_wtime();
        Tomp_gpu_normalizeEigenvectorMatrix += (finish - begin);
        printf("    Eigenvector matrix normalization completed!\n");
        
        
        // Final k-means(++) clustering
        printf("    Final k-means(++) clustering begins ...\n");
        begin = omp_get_wtime();
        
        // Declaration
        T_real *GPU_centroidsEigMat;
        
        // Memory allocation
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centroidsEigMat, (sizeof(T_real)*nbClusters)*nbClusters));
        
        // Call GPU k-means(++) function
        kmeans_gpu(SC_GPU,                                        // input (env)
                   nbPoints, nbDims, nbClusters, GPU_eigVectsTruncT,   // input
                   seedingMethod, seedBase, tolKMGPU, maxNbItersKM,                   // input 
                   tholdUsePackages, nbPackages, nbStreamsStep1, nbStreamsStep2,      // input
                   nbItersKM, GPU_count, GPU_centroidsEigMat, GPU_labels);            // output
        
        // Memory deallocation
        CHECK_CUDA_SUCCESS(cudaFree(GPU_centroidsEigMat));
        CHECK_CUDA_SUCCESS(cudaFree(GPU_eigVectsTruncT));
        
        finish = omp_get_wtime();
        Tomp_gpu_finalKmeansForSC += (finish - begin);
        printf("    Final k-means(++) clustering completed!\n");
        
        
        // nvGRAPH clustering quality evaluation
        printf("    nvGRAPH clustering analysis begins ...\n");
        evaluate_clustering_quality_via_nvgraph(nbPoints, nbClusters, nnzS,                    // input 
                                                GPU_csrRowS, GPU_csrColS, GPU_csrValS,         // input
                                                GPU_labels,                                    // input
                                                modularityScore, edgeCutScore, ratioCutScore); // output
        printf("    nvGRAPH clustering analysis completed!\n");
        
        // Memory deallocation
        CHECK_CUDA_SUCCESS(cudaFree(GPU_csrRowS));
        CHECK_CUDA_SUCCESS(cudaFree(GPU_csrColS));
        CHECK_CUDA_SUCCESS(cudaFree(GPU_csrValS));
    }
}
