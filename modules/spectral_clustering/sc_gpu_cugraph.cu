#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>
#include <omp.h>                      // omp_get_wtime
#include <thrust/device_vector.h>     // thrust::device_ptr
#include <thrust/execution_policy.h>  // thrust::device
#include <thrust/extrema.h>           // thrust::max_element, thrust::min_element, thrust::minmax_element
#include <thrust/fill.h>              // thrust::fill
#include <cugraph/algorithms.hpp> 

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/spectral_clustering/constr_epsilon_sim_matrix_in_csr.h"
#include "../../include/spectral_clustering/constr_epsilon_sim_matrix_in_dense_to_csr.h"
#include "../../include/spectral_clustering/get_edge_list.h"
#include "../../include/spectral_clustering/filter_noise.h"
#include "../../include/spectral_clustering/auto_tuning.h"
#include "../../include/spectral_clustering/sc_gpu_cugraph.h"


void spectral_clustering_on_gpu_involving_cugraph (int nbPoints, int nbDims, int nbClusters, T_real *GPU_dataT,
                                                   T_real sigma, T_real tholdSim, T_real tholdSqDist,
                                                   int csrAlgo, int hypoMaxNnzRow, T_real maxNzPercent,
                                                   T_real memUsePercent, int pad1, int pad2, int pad3,
                                                   int filterNoiseApproach, int nbBinsHist, T_real tholdNoise,
                                                   int flagAutoTuneNbClusters, int flagInteractive,
                                                   int cuGraphAlgo, T_real tolEigen, int maxNbItersEigen,
                                                   T_real tolKMGPU, int maxNbItersKM,
                                                   float *modularityScore, float *edgeCutScore, float *ratioCutScore,
                                                   int *optNbClusters, int *GPU_labels)
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
    
    
    // Filter noise and outliers
    if (filterNoiseApproach != 0) {
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
                fprintf(stderr, "Unknown approach to filtering noise!\n");
                exit(EXIT_FAILURE);
        }
        
        int nbClustersNF = nbClusters - 1;
        
        finish = omp_get_wtime();
        Tomp_gpu_filterNoise += (finish - begin - timeUserResponse);
        printf("    Noise filtering completed!\n");
        
        CHECK_CUDA_SUCCESS(cudaFree(GPU_nnzPerRowSNF));  // equal to cudaFree(GPU_nnzPerRowS)
        
        
        // cuGraph spectral graph partitioning on noise-free points
        // See the example at https://github.com/rapidsai/cugraph/blob/branch-22.02/cpp/tests/community/balanced_edge_test.cpp
        printf("    cuGraph spectral graph partitioning begins ...\n");
        begin = omp_get_wtime();
        
        int num_verts = nbPointsNF;
        int num_edges = nnzSNF;
        int num_clusters = nbClustersNF;
        int num_eigenvectors = nbClustersNF;
        T_real evs_tolerance = tolEigen;
        T_real kmean_tolerance = tolKMGPU;
        int evs_max_iter = maxNbItersEigen;
        int kmean_max_iter = maxNbItersKM;
        int *GPU_labelsNF;
        CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_labelsNF, sizeof(int)*nbPointsNF));
        
        cugraph::legacy::GraphCSRView<int, int, T_real> G(GPU_csrRowSNF, 
                                                          GPU_csrColSNF, 
                                                          GPU_csrValSNF, 
                                                          num_verts, 
                                                          num_edges);
        switch (cuGraphAlgo) {
            case 1 :
                cugraph::ext_raft::spectralModularityMaximization(G,
                                                                  num_clusters,
                                                                  num_eigenvectors,
                                                                  evs_tolerance,
                                                                  evs_max_iter,
                                                                  kmean_tolerance,
                                                                  kmean_max_iter,
                                                                  GPU_labelsNF);
                break;
            
            case 2: 
                cugraph::ext_raft::balancedCutClustering(G,
                                                         num_clusters,
                                                         num_eigenvectors,
                                                         evs_tolerance,
                                                         evs_max_iter,
                                                         kmean_tolerance,
                                                         kmean_max_iter,
                                                         GPU_labelsNF);
                break;
            
            default: 
                fprintf(stderr, "Unknown cuGraph algorithm for graph partitioning!\n");
                exit(EXIT_FAILURE);
        }
        
        finish = omp_get_wtime();
        Tomp_gpu_cuGraphSpectralClusteringAPI += (finish - begin);
        printf("    cuGraph spectral graph partitioning completed!\n");

        // cuGraph clustering analysis
        printf("    cuGraph clustering analysis begins ...\n");
        // modularityScore: modularity clustering score telling how good the clustering is compared to random assignments.
        // edgeCutScore  : total number of edges between clusters.
        // ratioCutScore : sum for all clusters of the number of edges going outside of the cluster divided by the number of vertices inside the cluster.
        cugraph::ext_raft::analyzeClustering_modularity(G, num_clusters, GPU_labelsNF, (T_real *)modularityScore);
        cugraph::ext_raft::analyzeClustering_edge_cut(G, num_clusters, GPU_labelsNF, (T_real *)edgeCutScore);
        cugraph::ext_raft::analyzeClustering_ratio_cut(G, num_clusters, GPU_labelsNF, (T_real *)ratioCutScore);
        printf("    cuGraph clustering analysis completed!\n");
        
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
        
    } else {
        
        CHECK_CUDA_SUCCESS(cudaFree(GPU_nnzPerRowS));

        // cuGraph spectral graph partitioning
        // See the example at https://github.com/rapidsai/cugraph/blob/branch-22.02/cpp/tests/community/balanced_edge_test.cpp
        printf("    cuGraph spectral graph partitioning begins ...\n");
        begin = omp_get_wtime();
        
        int num_verts = nbPoints;
        int num_edges = nnzS;
        int num_clusters = nbClusters;
        int num_eigenvectors = nbClusters;
        T_real evs_tolerance = tolEigen;
        T_real kmean_tolerance = tolKMGPU;
        int evs_max_iter = maxNbItersEigen;
        int kmean_max_iter = maxNbItersKM;
        
        cugraph::legacy::GraphCSRView<int, int, T_real> G(GPU_csrRowS, 
                                                          GPU_csrColS, 
                                                          GPU_csrValS, 
                                                          num_verts, 
                                                          num_edges);
        switch (cuGraphAlgo) {
            case 1 :
                cugraph::ext_raft::spectralModularityMaximization(G,
                                                                  num_clusters,
                                                                  num_eigenvectors,
                                                                  evs_tolerance,
                                                                  evs_max_iter,
                                                                  kmean_tolerance,
                                                                  kmean_max_iter,
                                                                  GPU_labels);
                break;
            
            case 2: 
                cugraph::ext_raft::balancedCutClustering(G,
                                                         num_clusters,
                                                         num_eigenvectors,
                                                         evs_tolerance,
                                                         evs_max_iter,
                                                         kmean_tolerance,
                                                         kmean_max_iter,
                                                         GPU_labels);
                break;
            
            default: 
                fprintf(stderr, "Unknown cuGraph algorithm for graph partitioning!\n");
                exit(EXIT_FAILURE);
        }
        
        finish = omp_get_wtime();
        Tomp_gpu_cuGraphSpectralClusteringAPI += (finish - begin);
        printf("    cuGraph spectral graph partitioning completed!\n");

        // cuGraph clustering analysis
        printf("    cuGraph clustering analysis begins ...\n");
        // modularityScore: modularity clustering score telling how good the clustering is compared to random assignments.
        // edgeCutScore  : total number of edges between clusters.
        // ratioCutScore : sum for all clusters of the number of edges going outside of the cluster divided by the number of vertices inside the cluster.
        cugraph::ext_raft::analyzeClustering_modularity(G, num_clusters, GPU_labels, (T_real *)modularityScore);
        cugraph::ext_raft::analyzeClustering_edge_cut(G, num_clusters, GPU_labels, (T_real *)edgeCutScore);
        cugraph::ext_raft::analyzeClustering_ratio_cut(G, num_clusters, GPU_labels, (T_real *)ratioCutScore);
        printf("    cuGraph clustering analysis completed!\n");
        
        // Memory deallocation
        CHECK_CUDA_SUCCESS(cudaFree(GPU_csrRowS));
        CHECK_CUDA_SUCCESS(cudaFree(GPU_csrColS));
        CHECK_CUDA_SUCCESS(cudaFree(GPU_csrValS));
    }
}