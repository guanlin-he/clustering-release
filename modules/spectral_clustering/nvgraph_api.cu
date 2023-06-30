#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <cuda.h> 
#include <cuda_runtime.h>
#include <omp.h>     // omp_get_wtime
#include <nvgraph.h> // nvgraphSpectralClustering, nvgraphAnalyzeClustering

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/init_gpu.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/spectral_clustering/nvgraph_api.h"


void similarity_graph_partitioning_via_nvgraph (int nbPoints, int nbClusters, int nnzS,                             // input 
                                                int *GPU_csrRowS, int *GPU_csrColS, T_real *GPU_csrValS,            // input
                                                int nvGraphAlgo, T_real tolEigen, int maxNbItersEigen,                 // input 
                                                T_real tolKMGPU, int maxNbItersKM,                               // input 
                                                T_real *GPU_eigVals, T_real *GPU_eigVects, int *GPU_labels)         // output
{   
    // Declaration & initialization
    int edge_numsets = 1;
    int weight_index = 0; 
    cudaDataType_t edge_dimT = T_REAL;
    nvgraphCSRTopology32I_st CSR_input = {nbPoints, nnzS, GPU_csrRowS, GPU_csrColS};
    // nvgraphCSRTopology32I_t CSR_input;
    // CSR_input = (nvgraphCSRTopology32I_st *) malloc(sizeof(struct nvgraphCSRTopology32I_st));
    // CSR_input->nvertices = nbPoints;
    // CSR_input->nedges = nnzS;
    // CSR_input->source_offsets = GPU_csrRowS;
    // CSR_input->destination_indices = GPU_csrColS;

    CHECK_NVGRAPH_SUCCESS(nvgraphSetGraphStructure(handleNVGRAPH, descrG, (void*)&CSR_input, NVGRAPH_CSR_32));
    // CHECK_NVGRAPH_SUCCESS(nvgraphSetGraphStructure(handleNVGRAPH, descrG, (void*)CSR_input, NVGRAPH_CSR_32));
    CHECK_NVGRAPH_SUCCESS(nvgraphAllocateEdgeData(handleNVGRAPH, descrG, edge_numsets, &edge_dimT));
    CHECK_NVGRAPH_SUCCESS(nvgraphSetEdgeData(handleNVGRAPH, descrG, (void*)GPU_csrValS, 0));
    struct SpectralClusteringParameter clustering_params;
    clustering_params.n_clusters = nbClusters;
    clustering_params.n_eig_vects = nbClusters;
    switch (nvGraphAlgo) {
        case 1 : 
            clustering_params.algorithm = NVGRAPH_MODULARITY_MAXIMIZATION; 
            break;
        
        case 2 : 
            clustering_params.algorithm = NVGRAPH_BALANCED_CUT_LANCZOS; 
            break;
        
        case 3 : 
            clustering_params.algorithm = NVGRAPH_BALANCED_CUT_LOBPCG; 
            break;
        
        default: 
            fprintf(stderr, "Unknown nvGRAPH algorithm for graph partitioning!\n");
            exit(EXIT_FAILURE);
    }
    clustering_params.evs_tolerance = tolEigen;
    clustering_params.evs_max_iter = maxNbItersEigen;
    clustering_params.kmean_tolerance = tolKMGPU;
    clustering_params.kmean_max_iter = maxNbItersKM;
    
    // Call nvgraphSpectralClustering function
    double begin = omp_get_wtime();
    CHECK_NVGRAPH_SUCCESS(nvgraphSpectralClustering(handleNVGRAPH, 
                                                    descrG, 
                                                    weight_index, 
                                                    &clustering_params, 
                                                    GPU_labels, 
                                                    GPU_eigVals, 
                                                    GPU_eigVects)); 
    double finish = omp_get_wtime();
    double Tomp_nvg = finish - begin;
    printf("    nvGRAPH API: %f ms\n", (float)Tomp_nvg*1.0E3f);
    
        // Save eigenvalues and eigenvectors of Laplacian matrix found by nvgraph
        // T_real *eigval_nvg;         
        // T_real *eigvect_nvg;
        // eigval_nvg = (T_real *) malloc(sizeof(T_real)*nbClusters);
        // eigvect_nvg = (T_real *) malloc((sizeof(T_real)*nbClusters)*nbPoints);
        // CHECK_CUDA_SUCCESS(cudaMemcpy(eigval_nvg, GPU_eigVals, sizeof(T_real)*nbClusters, cudaMemcpyDeviceToHost));   // Transfer nbClusters eigenvalues back to host
        // CHECK_CUDA_SUCCESS(cudaMemcpy(eigvect_nvg, GPU_eigVects, (sizeof(T_real)*nbClusters)*nbPoints, cudaMemcpyDeviceToHost));  // Transfer nbClusters eigenvectors back to host
        // save_file_real(eigval_nvg,  nbClusters, 1,          "output/Eigenvalues.txt",  "");
        // save_file_real(eigvect_nvg, nbPoints,   nbClusters, "output/Eigenvectors.txt", "\t");
        // free(eigval_nvg);
        // free(eigvect_nvg);
    

    // free(CSR_input);
}



void evaluate_clustering_quality_via_nvgraph (int nbPoints, int nbClusters, int nnzS,                             // input 
                                              int *GPU_csrRowS, int *GPU_csrColS, T_real *GPU_csrValS,            // input
                                              int *GPU_labels,                                                    // input
                                              float *modularityScore, float *edgeCutScore, float *ratioCutScore)  // output
{   
    // Declaration & initialization
    // int edge_numsets = 1;
    int weight_index = 0; 
    // cudaDataType_t edge_dimT = T_REAL;
    // nvgraphCSRTopology32I_st CSR_input = {nbPoints, nnzS, GPU_csrRowS, GPU_csrColS};
    // nvgraphCSRTopology32I_t CSR_input;
    // CSR_input = (nvgraphCSRTopology32I_st *) malloc(sizeof(struct nvgraphCSRTopology32I_st));
    // CSR_input->nvertices = nbPoints;
    // CSR_input->nedges = nnzS;
    // CSR_input->source_offsets = GPU_csrRowS;
    // CSR_input->destination_indices = GPU_csrColS;
    
    // CHECK_NVGRAPH_SUCCESS(nvgraphSetGraphStructure(handleNVGRAPH, descrG, (void*)&CSR_input, NVGRAPH_CSR_32));
    // CHECK_NVGRAPH_SUCCESS(nvgraphSetGraphStructure(handleNVGRAPH, descrG, (void*)CSR_input, NVGRAPH_CSR_32));
    // CHECK_NVGRAPH_SUCCESS(nvgraphAllocateEdgeData(handleNVGRAPH, descrG, edge_numsets, &edge_dimT));
    // CHECK_NVGRAPH_SUCCESS(nvgraphSetEdgeData(handleNVGRAPH, descrG, (void*)GPU_csrValS, 0));

    // NVGRAPH_MODULARITY: modularity clustering score telling how good the clustering is compared to random assignments.
    // NVGRAPH_EDGE_CUT  : total number of edges between clusters.
    // NVGRAPH_RATIO_CUT : sum for all clusters of the number of edges going outside of the cluster divided by the number of vertices inside the cluster.
    CHECK_NVGRAPH_SUCCESS(nvgraphAnalyzeClustering(handleNVGRAPH, descrG, weight_index, nbClusters, GPU_labels, NVGRAPH_MODULARITY, modularityScore));
    CHECK_NVGRAPH_SUCCESS(nvgraphAnalyzeClustering(handleNVGRAPH, descrG, weight_index, nbClusters, GPU_labels, NVGRAPH_EDGE_CUT, edgeCutScore));
    CHECK_NVGRAPH_SUCCESS(nvgraphAnalyzeClustering(handleNVGRAPH, descrG, weight_index, nbClusters, GPU_labels, NVGRAPH_RATIO_CUT, ratioCutScore));
    
    // free(CSR_input);
}