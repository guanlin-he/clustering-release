#include <stdio.h>  // printf, fprintf, stderr, stdin, stdout
#include <stdlib.h> // EXIT_FAILURE, EXIT_SUCCESS, malloc, free, exit
#include <omp.h>    // omp_get_wtime()
#include <sys/types.h> // mkdir
#include <sys/stat.h>  // mkdir
#include <unistd.h>    // access

#include "include/config.h"
#include "include/vars.h"
#include "include/utilities/arguments.h"
#include "include/utilities/dataIO.h"
#include "include/utilities/init_gpu.h"
#include "include/utilities/transfers.h"
#include "include/utilities/transpose.h"
#include "include/utilities/feature_scaling_cpu.h"
#include "include/utilities/feature_scaling_gpu.h"
#include "include/utilities/attachment.h"
#include "include/utilities/print.h"
#include "include/kmeans/kmeans_cpu.h"
#include "include/kmeans/kmeans_gpu.h"
#include "include/spectral_clustering/auto_tuning.h"
#include "include/spectral_clustering/sc_gpu_cusolverdn.h"
#include "include/spectral_clustering/sc_gpu_nvgraph.h"
#include "include/spectral_clustering/sc_gpu_nvgraph_km.h"
#include "include/spectral_clustering/sc_gpu_cugraph.h"
#include "include/spectral_clustering/constr_sim_matrix_on_cpu.h"


// Declare functions
void create_output_directory();
void init_global_variables();
void kmeans_clustering_on_cpu();
void kmeans_clustering_on_gpu();
void spectral_clustering_on_cpu();
void spectral_clustering_on_gpu();
void rep_based_sc_on_gpu();
void rep_based_sc_on_cpu();
void rep_based_sc_on_cpu_gpu();
void rep_based_sc();


int main (int argc, char *argv[])
{
    // Make some preparation
    init_global_variables();             // Initialize global variables
    command_line_parsing(argc, argv);    // Parse the command line
    print_configuration();               // Print configurations
    create_output_directory();           // Create a directory to store output files
    
    printf("- Program starts ...\n");

    // Get the beginning time of the application
    double beginApp, finishApp;
    beginApp = omp_get_wtime();   // omp_get_wtime() returns elapsed wall clock time in seconds

    // Perform clustering with one of the following algorithms
    switch(ClustAlgo) {
        case KM_CPU:  // Case 1: parallel k-means(++) clustering on CPU
            kmeans_clustering_on_cpu();
            break;
            
        case KM_GPU:  // Case 2: parallel k-means(++) clustering on GPU
            kmeans_clustering_on_gpu();
            break;
            
        case SC_CPU:  // Case 3: parallel spectral clustering on CPU (imcomplete due to the need of calling an API of scikit-learn)
            spectral_clustering_on_cpu();
            break;
            
        case SC_GPU:  // Case 4: parallel spectral clustering on GPU
            spectral_clustering_on_gpu();
            break;
            
        case SC_REPS: // Case 5: parallel representative-based spectral clustering on CPU / on GPU / on CPU+GPU
            rep_based_sc();
            break;
            
        default :
            fprintf(stderr, "Unknown clustering algorithm!\n");
            exit(EXIT_FAILURE);
    }
    
    // Get the ending time of the application
    finishApp = omp_get_wtime();
    
    // Compute the elapsed time of the application
    Tomp_application += (finishApp - beginApp);
    
    // Print results and performance
    print_results_performance();

    return(EXIT_SUCCESS);
}


// Create a directory to store output files
void create_output_directory()
{
    char *dir;
    dir = (char*)"./output";
    mode_t mode = 0744;  // let the directory readable, writable, and executable by me, while everybody else can only read the directory
    int status;
    
    if (access(dir, 0) == -1) { // access(): determine accessibility of a file
        printf("    %s is missing, now make it.\n", dir);
        status = mkdir(dir, mode);  // make a directory
        if (status == 0) {
            printf("    %s is created successfully.\n", dir);
        } else {
            printf("    Failed to create %s.\n", dir);
        }
    }
}


// Initialize global variables
void init_global_variables()
{
    // Related to global application (regulable by users)
    ClustAlgo               = DEFAULT_CLUSTERING_ALGO;             // regulable by the "-algo" argument
    NbThreadsCPU            = DEFAULT_NB_THREADS_CPU;              // regulable by the "-cpu-nt" argument
    FlagFeatureScaling      = DEFAULT_FLAG_FEATURE_SCALING;        // regulable by the "-fs" argument
    
    // Related to k-means(++) clustering (regulable by users)
    // - related to both k-means(++) on CPU and k-means(++) on GPU
    SeedBase                = DEFAULT_SEED_BASE;                   // regulable by the "-seedbase" argument
    TholdUsePackages        = DEFAULT_THOLD_USE_PACKAGES;          // regulable by the "-thold-use-pkgs" argument
    NbPackages              = DEFAULT_NB_PACKAGES;                 // regulable by the "-np" argument
    MaxNbItersKM            = DEFAULT_MAX_NB_ITERS_KMEANS;         // regulable by the "-max-iters-km" argument
    // - only related to k-means(++) on CPU
    SeedingKMCPU            = DEFAULT_SEEDING_KMEANS_CPU;          // regulable by the "-seeding-km-cpu" argument
    TolKMCPU                = DEFAULT_TOL_KMEANS_CPU;              // regulable by the "-tol-km-cpu" argument
    // - only related to k-means(++) on GPU
    SeedingKMGPU            = DEFAULT_SEEDING_KMEANS_GPU;          // regulable by the "-seeding-km-gpu" argument
    TolKMGPU                = DEFAULT_TOL_KMEANS_GPU;              // regulable by the "-tol-km-gpu" argument
    NbStreamsStep1          = DEFAULT_NB_STREAMS_UPDATE_S1;        // regulable by the "-ns1" argument
    NbStreamsStep2          = DEFAULT_NB_STREAMS_UPDATE_S2;        // regulable by the "-ns2" argument
    
    // Related to spectral clustering (regulable by users)
    SCImpGPU                = DEFAULT_SC_IMPLEMENTATION_ON_GPU;    // regulable by the "-sc-imp" argument
    // - related to similarity matrix construction
    CSRAlgo                 = DEFAULT_CSR_ALGO;                    // regulable by the "-csr-algo" argument
    Sigma                   = DEFAULT_SIGMA;                       // regulable by the "-sigma" argument
    TholdSim                = DEFAULT_THOLD_SIM;                   // regulable by the "-thold-sim" argument
    TholdDistSq             = DEFAULT_THOLD_DIST_SQ;               // regulable by the "-thold-dist-sq" argument
    HypoMaxNnzRow           = DEFAULT_HYPO_MAX_NNZ_ROW;            // regulable by the "-hypo-max-nnz-row" argument
    Pad1                    = DEFAULT_PAD1;                        // regulable by the "-pad1" argument
    Pad2                    = DEFAULT_PAD2;                        // regulable by the "-pad2" argument
    Pad3                    = DEFAULT_PAD3;                        // regulable by the "-pad3" argument
    MemUsePercent           = DEFAULT_MEM_USE_PERCENT;             // regulable by the "-mem-use-percent" argument
    MaxNzPercent            = DEFAULT_MAX_NZ_PERCENT;              // regulable by the "-max-nz-percent" argument
    SimConstrAlgoCPU        = DEFAULT_SIM_CONSTR_ALGO_CPU;         // regulable by the "-sim-constr-algo-cpu" argument
    // - related to nvGRAPH & cuGraph libraries
    NVGraphAlgo             = DEFAULT_NVGRAPH_SC_ALGO;             // regulable by the "-nvg-algo" argument
    CUGraphAlgo             = DEFAULT_CUGRAPH_SC_ALGO;             // regulable by the "-cug-algo" argument
    MaxNbItersEigen         = DEFAULT_MAX_NB_ITERS_EIGEN;          // regulable by the "-max-iters-eig" argument
    TolEigen                = DEFAULT_TOL_EIGEN;                   // regulable by the "-tol-eig" argument
    // - related to noise filtering
    FilterNoiseApproach     = DEFAULT_FILTER_NOISE_APPROACH;       // regulable by the "-filter-noise" argument
    NbBinsHist              = DEFAULT_NB_BINS_HIST;                // regulable by the "-nb-bins-hist" argument
    TholdNoise              = DEFAULT_THOLD_NOISE;                 // regulable by the "-thold-noise" argument
    // - related to auto-tuning of the number of clusters
    FlagAutoTuneNbClusters  = DEFAULT_FLAG_AUTO_TUNE_NB_CLUSTERS;  // regulable by the "-auto-tune-nc" argument
    // - related to both noise filtering and auto-tuning
    FlagInteractive         = DEFAULT_FLAG_INTERACTIVE;            // regulable by the "-interactive" argument

    // Related to representative-based spectral clustering (regulable by users)
    MethodToExtractReps     = DEFAULT_METHOD_TO_EXTRACT_REPS;      // regulable by the "-er" argument
    Chain                   = DEFAULT_CHAIN_OF_SC_USING_REPS;      // regulable by the "-chain" argument
    
    // Related to block size configuration for CUDA kernels (regulable by users)
    BsXN                    = DEFAULT_BLOCK_SIZE_X_N;              // regulable by the "-bsxn" argument
    BsXP                    = DEFAULT_BLOCK_SIZE_X_P;              // regulable by the "-bsxp" argument
    BsXD                    = DEFAULT_BLOCK_SIZE_X_D;              // regulable by the "-bsxd" argument
    BsXC                    = DEFAULT_BLOCK_SIZE_X_C;              // regulable by the "-bsxc" argument
    BsYN                    = DEFAULT_BLOCK_SIZE_Y_N;              // regulable by the "-bsyn" argument
    BsXK1                   = DEFAULT_BLOCK_SIZE_X_N;              // regulable by the "-bsxk1" argument
    BsXK2                   = DEFAULT_BLOCK_SIZE_X_N;              // regulable by the "-bsxk2" argument
    BsXK3                   = DEFAULT_BLOCK_SIZE_X_N;              // regulable by the "-bsxk3" argument
    BsXK4                   = DEFAULT_BLOCK_SIZE_X_N;              // regulable by the "-bsxk4" argument
    BsXK5                   = DEFAULT_BLOCK_SIZE_X_N;              // regulable by the "-bsxk5" argument
    BsXK6                   = DEFAULT_BLOCK_SIZE_X_N;              // regulable by the "-bsxk6" argument
    BsYK1                   = DEFAULT_BLOCK_SIZE_Y_N;              // regulable by the "-bsyk1" argument
    BsYK2                   = DEFAULT_BLOCK_SIZE_Y_N;              // regulable by the "-bsyk2" argument              
    BsYK3                   = DEFAULT_BLOCK_SIZE_Y_N;              // regulable by the "-bsyk3" argument
    BsYK4                   = DEFAULT_BLOCK_SIZE_Y_N;              // regulable by the "-bsyk4" argument
    BsYK5                   = DEFAULT_BLOCK_SIZE_Y_N;              // regulable by the "-bsyk5" argument
    BsYK6                   = DEFAULT_BLOCK_SIZE_Y_N;              // regulable by the "-bsyk6" argument
    
    // Initialization of global timing variables
    Tomp_application = 0.0;
    // - on CPU
    Tomp_cpu_readData = 0.0;
    Tomp_cpu_featureScaling = 0.0;
    Tomp_cpu_randomSampling = 0.0; Tomp_cpu_d2Sampling = 0.0; Tomp_cpu_attach = 0.0;
    Tomp_cpu_seeding = 0.0; Tomp_cpu_computeAssign = 0.0; Tomp_cpu_updateCentroids = 0.0; Tomp_cpu_kmeans = 0.0; 
    Tomp_cpu_constructSimMatrix = 0.0;
    Tomp_cpu_membershipAttach = 0.0;
    Tomp_cpu_saveResults = 0.0;
    Tomp_cpu_saveSimMatrix = 0.0;
    // - on GPU
    Tomp_gpu_randomSampling = 0.0, Tomp_gpu_attach = 0.0;
    Tomp_gpu_cuInit = 0.0;
    Tomp_gpu_computeUnscaledCentroids = 0.0;
    Tomp_gpu_transposeReps = 0.0;
    Tomp_gpu_featureScaling = 0.0;
    Tomp_gpu_seeding = 0.0; Tomp_gpu_computeAssign = 0.0; Tomp_gpu_updateCentroids = 0.0; Tomp_gpu_kmeans = 0.0; Tomp_gpu_kmeanspp = 0.0;
    Tomp_gpu_spectralClustering = 0.0;
    Tomp_gpu_constructSimLapMatrix = 0.0; Tomp_gpu_constructSimMatrixInCSR = 0.0;
    Tomp_gpu_filterNoise = 0.0;
    Tomp_gpu_cuSolverDNsyevdx = 0.0; Tomp_gpu_nvGRAPHSpectralClusteringAPI = 0.0; Tomp_gpu_cuGraphSpectralClusteringAPI = 0.0;
    Tomp_gpu_autoTuneNbClusters = 0.0;
    Tomp_gpu_normalizeEigenvectorMatrix = 0.0;
    Tomp_gpu_finalKmeansForSC = 0.0;
    Tomp_gpu_membershipAttach = 0.0;
    // - on CPU-GPU
    Tomp_cpu_gpu_transfers = 0.0; Tomp_gpu_cpu_transfers = 0.0;
}


// Parallel k-means(++) clustering on CPU
void kmeans_clustering_on_cpu()
{
    // Declare variables
    double begin, finish;
    int nbPoints   = NB_POINTS;
    int nbDims     = NB_DIMS;
    int nbClusters = NB_CLUSTERS;
    T_real *data;            // Array for the matrix of data instances
    T_real *centroids;       // Array for the matrix of centroids
    int *labels;             // Array for cluster labels of data instances
    int *countPerCluster;    // Array for the number of data instances in each cluster
    T_real *dimMax;          // Array for the maximal value in each dimension
    T_real *dimMin;          // Array for the minimal value in each dimension
    
    // Allocate memory for arrays
    data = (T_real *) malloc((sizeof(T_real)*nbPoints)*nbDims);
    centroids = (T_real *) malloc((sizeof(T_real)*nbClusters)*nbDims);
    labels = (int *) malloc(sizeof(int)*nbPoints);
    countPerCluster = (int *) malloc(sizeof(int)*nbClusters);
    
    // Set the number of OpenMP threads on CPU
    omp_set_num_threads(NbThreadsCPU);
    
    // Read the data file
    printf("    Data file reading begins ...\n");
    begin = omp_get_wtime();
    if (DATASET_NAME == "Clouds4D_5E7") {
        read_file_real(data, nbPoints, nbDims, INPUT_DATA, " ", 0, 0);  // " " delimter for InputDataset-50million.txt
    } else {
        read_file_real(data, nbPoints, nbDims, INPUT_DATA, "\t", 0, 0);
    }
    finish = omp_get_wtime();
    Tomp_cpu_readData += (finish - begin);
    printf("    Data file reading completed!\n");

    // Perform feature scaling on CPU (if needed)
    if (FlagFeatureScaling) {
        printf("    Feature scaling begins ...\n");
        dimMax = (T_real *) malloc(sizeof(T_real)*nbDims);
        dimMin = (T_real *) malloc(sizeof(T_real)*nbDims);
        begin = omp_get_wtime();
        feature_scaling(nbPoints, nbDims,  // input
                        data,              // input & output
                        dimMax, dimMin);
        finish = omp_get_wtime();
        Tomp_cpu_featureScaling += (finish - begin);
        printf("    Feature scaling completed!\n");
        // save_file_real(data, nbPoints, nbDims, "output/Data_feature_scaled.txt", "\t", 0);
    }
    
    // Perform k-means(++) clustering on CPU
    printf("    k-means(++) clustering on CPU begins ...\n");
    begin = omp_get_wtime();
    kmeans_cpu(nbPoints, nbDims, nbClusters, data,  // input
               SeedingKMCPU, SeedBase,              // input
               TholdUsePackages, NbPackages,        // input
               TolKMCPU, MaxNbItersKM,              // input
               &NbItersKMCPU, countPerCluster, centroids, labels);   // output
    finish = omp_get_wtime();
    Tomp_cpu_kmeans += (finish - begin);
    printf("    k-means(++) clustering on CPU completed!\n");
    
    // Perform inverse feature scaling on GPU to obtain cluster centroids on the initial scale (if needed)
    if (FlagFeatureScaling) {
        inverse_feature_scaling(dimMax, dimMin,      // input
                                nbClusters, nbDims,  // input
                                centroids);          // input & output
        free(dimMax);
        free(dimMin);
    }
    
    // Save results into .txt files
    printf("    Result saving begins ...\n");
    begin = omp_get_wtime();
    save_file_int(labels, nbPoints, 1, "output/Labels.txt", "", 0);
    save_file_int(countPerCluster, nbClusters, 1, "output/CountPerCluster.txt", "", 0);
    save_file_real(centroids, nbClusters, nbDims, "output/FinalCentroids.txt", "\t", 0);
    finish = omp_get_wtime();
    Tomp_cpu_saveResults += (finish - begin);
    printf("    Result saving completed!\n");
    
    // Deallocate memory
    free(data);
    free(centroids);
    free(countPerCluster);
    free(labels);
}


// Parallel k-means(++) clustering on GPU
void kmeans_clustering_on_gpu()
{
    // Declare variables
    double begin, finish;
    int nbPoints   = NB_POINTS;
    int nbDims     = NB_DIMS;
    int nbClusters = NB_CLUSTERS;
    T_real *dataT;                // Array for the transposed matrix of data instances
    T_real *centroids;            // Array for the matrix of centroids
    int    *countPerCluster;      // Array for the nb of data instances in each cluster
    int    *labels;               // Array for cluster labels of data instances
    T_real *GPU_dataT;            // GPU array for the transposed matrix of data instances
    T_real *GPU_centroids;        // GPU array for the matrix of centroids
    int    *GPU_countPerCluster;  // GPU array for the nb of data instances in each cluster
    int    *GPU_labels;           // GPU array for cluster labels of data instances
    float *GPU_dimMax;
    float *GPU_dimMin;
    
    // Initialize the GPU device and some CUDA libraries
    begin = omp_get_wtime();
    init_gpu();
    finish = omp_get_wtime();
    Tomp_gpu_cuInit += (finish - begin);
    
    // Allocate memory for arrays
    dataT = (T_real *) malloc((sizeof(T_real)*nbDims)*nbPoints);
    centroids = (T_real *) malloc((sizeof(T_real)*nbClusters)*nbDims);
    countPerCluster = (int *) malloc(sizeof(int)*nbClusters);
    labels = (int *) malloc(sizeof(int)*nbPoints);
    real_data_memory_allocation_gpu(&GPU_dataT, (sizeof(T_real)*nbDims)*nbPoints);
    real_data_memory_allocation_gpu(&GPU_centroids, (sizeof(T_real)*nbClusters)*nbDims);
    int_data_memory_allocation_gpu(&GPU_countPerCluster, sizeof(int)*nbClusters);
    int_data_memory_allocation_gpu(&GPU_labels, sizeof(int)*nbPoints);
    
    // Read the data file
    printf("    Data file reading begins ...\n");
    begin = omp_get_wtime();
    if (DATASET_NAME == "Clouds4D_5E7") {
        read_file_real(dataT, nbPoints, nbDims, INPUT_DATA, " ", 0, 1);  // " " delimter for InputDataset-50million.txt
    } else {
        read_file_real(dataT, nbPoints, nbDims, INPUT_DATA, "\t", 0, 1);
    }
    finish = omp_get_wtime();
    Tomp_cpu_readData += (finish - begin);
    printf("    Data file reading completed!\n");
    
    // Transfer data from host (CPU) to device (GPU)
    printf("    Host-to-device data transfers begins ...\n");
    begin = omp_get_wtime();
    real_data_register(dataT, (sizeof(T_real)*nbDims)*nbPoints);
    real_data_transfers_cpu_to_gpu(dataT, (sizeof(T_real)*nbDims)*nbPoints,   // input
                                   GPU_dataT);                                // output
    real_data_unregister(dataT);
    finish = omp_get_wtime();
    Tomp_cpu_gpu_transfers += (finish - begin);
    printf("    Host-to-device data transfers completed!\n");

    // Perform feature scaling on GPU (if needed)
    if (FlagFeatureScaling) {
        printf("    Feature scaling begins ...\n");
        float_data_memory_allocation_gpu(&GPU_dimMax, sizeof(float)*nbDims);
        float_data_memory_allocation_gpu(&GPU_dimMin, sizeof(float)*nbDims);
        begin = omp_get_wtime();
        feature_scaling_on_gpu(nbDims, nbPoints,          // input
                               GPU_dataT,                 // input & output
                               GPU_dimMax, GPU_dimMin);   // output
        finish = omp_get_wtime();
        Tomp_gpu_featureScaling += (finish - begin);
        printf("    Feature scaling completed!\n");
        // real_data_register(dataT, (sizeof(T_real)*nbDims)*nbPoints);
        // real_data_transfers_gpu_to_cpu(GPU_dataT, (sizeof(T_real)*nbDims)*nbPoints,  // input
                                       // dataT);                                       // output
        // real_data_unregister(dataT);
        // save_file_real(dataT, nbDims, nbPoints, "output/DataT_feature_scaled.txt", "\t", 0);
    }
    
    // Perform k-means(++) clustering on GPU
    printf("    k-means(++) clustering on GPU begins ...\n");
    begin = omp_get_wtime();
    kmeans_gpu(KM_GPU,                                                          // input (env)
               nbPoints, nbDims, nbClusters, GPU_dataT,                         // input
               SeedingKMGPU, SeedBase, TolKMGPU, MaxNbItersKM,                  // input 
               TholdUsePackages, NbPackages, NbStreamsStep1, NbStreamsStep2,    // input
               &NbItersKMGPU, GPU_countPerCluster, GPU_centroids, GPU_labels);  // output
    finish = omp_get_wtime();
    Tomp_gpu_kmeans += (finish - begin);
    printf("    k-means(++) clustering on GPU completed!\n");

    // Perform inverse feature scaling on GPU to obtain cluster centroids on the initial scale (if needed)
    if (FlagFeatureScaling) {
        begin = omp_get_wtime();
        compute_unscaled_centroids(GPU_dimMax, GPU_dimMin,
                                   nbClusters, nbDims,
                                   GPU_centroids);
        finish = omp_get_wtime();
        Tomp_gpu_computeUnscaledCentroids += (finish - begin);
        float_data_memory_deallocation_gpu(GPU_dimMax);
        float_data_memory_deallocation_gpu(GPU_dimMin);
    }
    
    // Transfer results from device (GPU) to host (CPU)
    printf("    Device-to-host result transfers begins ...\n");
    begin = omp_get_wtime();
    int_data_register(labels, sizeof(int)*nbPoints);
    int_data_register(countPerCluster, sizeof(int)*nbClusters);
    real_data_register(centroids, (sizeof(T_real)*nbClusters)*nbDims);
    int_data_transfers_gpu_to_cpu(GPU_labels, sizeof(int)*nbPoints,  // input
                                  labels);                           // output
    int_data_transfers_gpu_to_cpu(GPU_countPerCluster, sizeof(int)*nbClusters,  // input
                                  countPerCluster);                             // output
    real_data_transfers_gpu_to_cpu(GPU_centroids, (sizeof(T_real)*nbClusters)*nbDims,  // input
                                   centroids);                                         // output
    int_data_unregister(labels);
    int_data_unregister(countPerCluster);
    real_data_unregister(centroids);
    finish = omp_get_wtime();
    Tomp_gpu_cpu_transfers += (finish - begin);
    printf("    Device-to-host result transfers completed!\n");
    
    // Save results into .txt files
    printf("    Result saving begins ...\n");
    begin = omp_get_wtime();
    save_file_int(labels, nbPoints, 1, "output/Labels.txt", "", 0);
    save_file_int(countPerCluster, nbClusters, 1, "output/CountPerCluster.txt", "", 0);
    save_file_real(centroids, nbClusters, nbDims, "output/FinalCentroids.txt", "\t", 0);
    finish = omp_get_wtime();
    Tomp_cpu_saveResults += (finish - begin);
    printf("    Result saving completed!\n");
    
    // Deallocate memory
    real_data_memory_deallocation_gpu(GPU_dataT);
    real_data_memory_deallocation_gpu(GPU_centroids);
    int_data_memory_deallocation_gpu(GPU_countPerCluster);
    int_data_memory_deallocation_gpu(GPU_labels);
    finalize_gpu();
    free(dataT);
    free(centroids);
    free(countPerCluster);
    free(labels);
}



// Parallel spectral clustering on CPU (imcomplete due to the need of calling an API of scikit-learn)
void spectral_clustering_on_cpu()
{
    // Declare variables
    double begin, finish;
    int nbPoints   = NB_POINTS;
    int nbDims     = NB_DIMS;
    int nbClusters = NB_CLUSTERS;
    T_real *data;            // Array for the matrix of data instances
    T_real *centroids;       // Array for the matrix of centroids
    int *labels;             // Array for cluster labels of data instances
    
    // Allocate memory for arrays
    data = (T_real *) malloc((sizeof(T_real)*nbPoints)*nbDims);
    centroids = (T_real *) malloc((sizeof(T_real)*nbClusters)*nbDims);
    labels = (int *) malloc(sizeof(int)*nbPoints);
    
    // Set the number of OpenMP threads on CPU
    omp_set_num_threads(NbThreadsCPU);
    
    // Read the data file
    printf("    Data file reading begins ...\n");
    begin = omp_get_wtime();
    if (DATASET_NAME == "Clouds4D_5E7") {
        read_file_real(data, nbPoints, nbDims, INPUT_DATA, " ", 0, 0);  // " " delimter for InputDataset-50million.txt
    } else {
        read_file_real(data, nbPoints, nbDims, INPUT_DATA, "\t", 0, 0);
    }
    finish = omp_get_wtime();
    Tomp_cpu_readData += (finish - begin);
    printf("    Data file reading completed!\n");

    // Perform feature scaling on CPU (if needed)
    if (FlagFeatureScaling) {
        printf("    Feature scaling begins ...\n");
        T_real *dimMax;
        T_real *dimMin;
        dimMax = (T_real *) malloc(sizeof(T_real)*nbDims);
        dimMin = (T_real *) malloc(sizeof(T_real)*nbDims);
        begin = omp_get_wtime();
        feature_scaling(nbPoints, nbDims,  // input
                        data,              // input & output
                        dimMax, dimMin);   // output
        finish = omp_get_wtime();
        Tomp_cpu_featureScaling += (finish - begin);
        free(dimMax);
        free(dimMin);
        printf("    Feature scaling completed!\n");
        // save_file_real(data, nbPoints, nbDims, "output/Data_feature_scaled.txt", "\t", 0);
    }
    
    // Construct the similarity matrix in dense/CSR format on CPU
    printf("    Similarity matrix construction on CPU begins ...\n");
    begin = omp_get_wtime();
    constr_similarity_matrix_on_cpu(nbPoints, nbDims, data,         // input
                                    SimConstrAlgoCPU, NbThreadsCPU, // input
                                    Sigma, TholdSim, TholdDistSq);  // input
    finish = omp_get_wtime();
    Tomp_cpu_constructSimMatrix += (finish - begin) - Tomp_cpu_saveSimMatrix;
    printf("    Similarity matrix construction on CPU completed!\n");
    printf("    Similarity matrix construction on CPU: %.2f s (not including the time of saving results into .txt files)\n", (float)Tomp_cpu_constructSimMatrix);
    
    // Call the Spectral Clustering API of scikit-learn
    // ...
    printf("    It remains to import the precomputed similarity matrix and call the Spectral Clustering API of scikit-learn!\n");
    
    // Save results into .txt files
    // printf("    Result saving begins ...\n");
    // begin = omp_get_wtime();
    // save_file_int(labels, nbPoints, 1, "output/Labels.txt", "", 0);
    // save_file_real(centroids, nbClusters, nbDims, "output/FinalCentroids.txt", "\t", 0);
    // finish = omp_get_wtime();
    // Tomp_cpu_saveResults += (finish - begin);
    // printf("    Result saving completed!\n");
    
    // Deallocate memory
    free(data);
    free(centroids);
    free(labels);
}


// Parallel spectral clustering on GPU
void spectral_clustering_on_gpu()
{
    // Declare variables
    double begin, finish;
    int nbPoints, nbDims, nbClusters;
    int maxNbClusters, optNbClusters;
    T_real *dataT;                // Array for the transposed matrix of data instances
    int    *countPerCluster;      // Array for the nb of data instances in each cluster
    int    *labels;               // Array for cluster labels of data instances
    T_real *GPU_dataT;            // GPU array for the transposed matrix of data instances
    int    *GPU_countPerCluster;  // GPU array for the nb of data instances in each cluster
    int    *GPU_labels;           // GPU array for cluster labels of data instances
    
    // Initialize some basic variables
    nbPoints = NB_POINTS;
    nbDims = NB_DIMS;
    if (FlagAutoTuneNbClusters == 1 && (SCImpGPU == DN_CUS || SCImpGPU == SP_NVG_KM)) {
        nbClusters = MAX_NB_CLUSTERS;
    } else {
        nbClusters = NB_CLUSTERS;
    }
    maxNbClusters = nbClusters;
    optNbClusters = nbClusters;
    
    // Initialize the GPU device and some CUDA libraries
    begin = omp_get_wtime();
    init_gpu();
    finish = omp_get_wtime();
    Tomp_gpu_cuInit += (finish - begin);
    
    // Allocate memory for arrays
    dataT = (T_real *) malloc((sizeof(T_real)*nbDims)*nbPoints);
    countPerCluster = (int *) malloc(sizeof(int)*nbClusters);
    labels = (int *) malloc(sizeof(int)*nbPoints);
    real_data_memory_allocation_gpu(&GPU_dataT, (sizeof(T_real)*nbDims)*nbPoints);
    int_data_memory_allocation_gpu(&GPU_countPerCluster, sizeof(int)*nbClusters);
    int_data_memory_allocation_gpu(&GPU_labels, sizeof(int)*nbPoints);
    
    // Read the data file
    printf("    Data file reading begins ...\n");
    begin = omp_get_wtime();
    if (DATASET_NAME == "Clouds4D_5E7") {
        read_file_real(dataT, nbPoints, nbDims, INPUT_DATA, " ", 0, 1);  // " " delimter for InputDataset-50million.txt
    } else {
        read_file_real(dataT, nbPoints, nbDims, INPUT_DATA, "\t", 0, 1);
    }
    finish = omp_get_wtime();
    Tomp_cpu_readData += (finish - begin);
    printf("    Data file reading completed!\n");

    // Transfer data from host (CPU) to device (GPU)
    printf("    Host-to-device data transfers begins ...\n");
    begin = omp_get_wtime();
    real_data_register(dataT, (sizeof(T_real)*nbDims)*nbPoints);
    real_data_transfers_cpu_to_gpu(dataT, (sizeof(T_real)*nbDims)*nbPoints,   // input
                                   GPU_dataT);                                // output
    real_data_unregister(dataT);
    finish = omp_get_wtime();
    Tomp_cpu_gpu_transfers += (finish - begin);
    printf("    Host-to-device data transfers completed!\n");
    
    // Perform feature scaling on GPU (if needed)
    if (FlagFeatureScaling) {
        printf("    Feature scaling begins ...\n");
        float *GPU_dimMax;
        float *GPU_dimMin;
        float_data_memory_allocation_gpu(&GPU_dimMax, sizeof(float)*nbDims);
        float_data_memory_allocation_gpu(&GPU_dimMin, sizeof(float)*nbDims);
        begin = omp_get_wtime();
        feature_scaling_on_gpu(nbDims, nbPoints,         // input
                               GPU_dataT,                // input & output
                               GPU_dimMax, GPU_dimMin);  // output
        finish = omp_get_wtime();
        Tomp_gpu_featureScaling += (finish - begin);
        float_data_memory_deallocation_gpu(GPU_dimMax);
        float_data_memory_deallocation_gpu(GPU_dimMin);
        printf("    Feature scaling completed!\n");
        // real_data_register(dataT, (sizeof(T_real)*nbDims)*nbPoints);
        // real_data_transfers_gpu_to_cpu(GPU_dataT, (sizeof(T_real)*nbDims)*nbPoints,  // input
                                       // dataT);                                       // output
        // real_data_unregister(dataT);
        // save_file_real(dataT, nbDims, nbPoints, "output/DataT_feature_scaled.txt", "\t", 0);
    }
    
    // Perform spectral clustering on GPU with one of the following implementations
    switch (SCImpGPU) {
        case DN_CUS :  // Case 1: spectral clustering in dense storage format involving cuSolverDN library
            printf("    Spectral clustering (involving cuSolverDN) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_cusolverdn(nbPoints, nbDims, nbClusters, GPU_dataT,                         // input
                                                            Sigma, TholdSim, TholdDistSq,                                    // input 
                                                            FlagAutoTuneNbClusters, maxNbClusters, FlagInteractive,          // input 
                                                            SeedingKMGPU, SeedBase, TolKMGPU, MaxNbItersKM,                  // input
                                                            TholdUsePackages, NbPackages, NbStreamsStep1, NbStreamsStep2,    // input
                                                            &NbItersKMGPU, &optNbClusters, GPU_countPerCluster, GPU_labels); // output
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving cuSolverDN) completed!\n");
            break;

        case SP_NVG :  // Case 2: spectral clustering in sparse storage format involving nvGRAPH library
            printf("    Spectral clustering (involving nvGRAPH) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_nvgraph(nbPoints, nbDims, nbClusters, GPU_dataT,          // input
                                                         Sigma, TholdSim, TholdDistSq,                     // input
                                                         CSRAlgo, HypoMaxNnzRow, MaxNzPercent,             // input
                                                         MemUsePercent, Pad1, Pad2, Pad3,                  // input
                                                         FilterNoiseApproach, NbBinsHist, TholdNoise,      // input
                                                         FlagAutoTuneNbClusters, FlagInteractive,          // input
                                                         NVGraphAlgo, TolEigen, MaxNbItersEigen,           // input
                                                         TolKMGPU, MaxNbItersKM,                           // input
                                                         &ModularityScore, &EdgeCutScore, &RatioCutScore,  // input
                                                         &optNbClusters, GPU_labels);                      // output
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving nvGRAPH) completed!\n");
            break;
        
        case SP_NVG_KM :  // Case 3: spectral clustering in sparse storage format involving nvGRAPH library + our k-means(++) implementation
            printf("    Spectral clustering (involving nvGRAPH & our k-means(++)) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_nvgraph_and_kmeans(nbPoints, nbDims, nbClusters, GPU_dataT,                         // input
                                                                    Sigma, TholdSim, TholdDistSq,                                    // input
                                                                    CSRAlgo, HypoMaxNnzRow, MaxNzPercent,                            // input
                                                                    MemUsePercent, Pad1, Pad2, Pad3,                                 // input
                                                                    FilterNoiseApproach, NbBinsHist, TholdNoise,                     // input
                                                                    NVGraphAlgo, TolEigen, MaxNbItersEigen,                          // input
                                                                    FlagAutoTuneNbClusters, maxNbClusters, FlagInteractive,          // input
                                                                    SeedingKMGPU, SeedBase, TolKMGPU, MaxNbItersKM,                  // input
                                                                    TholdUsePackages, NbPackages, NbStreamsStep1, NbStreamsStep2,    // input
                                                                    &ModularityScore, &EdgeCutScore, &RatioCutScore,                 // input
                                                                    &optNbClusters, &NbItersKMGPU, GPU_countPerCluster, GPU_labels); // output
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving nvGRAPH & our k-means(++)) completed!\n");
            break;
            
        case SP_CUG :  // Case 4: spectral clustering in sparse storage format involving cuGraph library
            printf("    Spectral clustering (involving cuGraph) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_cugraph(nbPoints, nbDims, nbClusters, GPU_dataT,          // input
                                                         Sigma, TholdSim, TholdDistSq,                     // input
                                                         CSRAlgo, HypoMaxNnzRow, MaxNzPercent,             // input
                                                         MemUsePercent, Pad1, Pad2, Pad3,                  // input
                                                         FilterNoiseApproach, NbBinsHist, TholdNoise,      // input
                                                         FlagAutoTuneNbClusters, FlagInteractive,          // input
                                                         CUGraphAlgo, TolEigen, MaxNbItersEigen,           // input
                                                         TolKMGPU, MaxNbItersKM,                           // input
                                                         &ModularityScore, &EdgeCutScore, &RatioCutScore,  // input
                                                         &optNbClusters, GPU_labels);                      // output
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving cuGraph) completed!\n");
            break;
        
        default :
            fprintf(stderr, "Unknown GPU implementation of spectral clustering!\n");
            exit(EXIT_FAILURE);
    }

    // Update nbClusters if the auto-tuning mechanism is enabled
    if (FlagAutoTuneNbClusters == 1) {
        nbClusters = optNbClusters;
    }
    
    // Transfer results from device (GPU) to host (CPU)
    printf("    Device-to-host result transfers begins ...\n");
    begin = omp_get_wtime();
    int_data_register(labels, sizeof(int)*nbPoints);
    int_data_transfers_gpu_to_cpu(GPU_labels, sizeof(int)*nbPoints,  // input
                                  labels);                           // output
    int_data_unregister(labels);
    if (SCImpGPU == DN_CUS || SCImpGPU == SP_NVG_KM) {
        int_data_register(countPerCluster, sizeof(int)*nbClusters);
        int_data_transfers_gpu_to_cpu(GPU_countPerCluster, sizeof(int)*nbClusters,  // input
                                      countPerCluster);                             // output
        int_data_unregister(countPerCluster);
    }
    finish = omp_get_wtime();
    Tomp_gpu_cpu_transfers += (finish - begin);
    printf("    Device-to-host result transfers completed!\n");
    
    // Save results into .txt files
    printf("    Result saving begins ...\n");
    begin = omp_get_wtime();
    save_file_int(labels, nbPoints, 1, "output/Labels.txt", "", 0);
    if (SCImpGPU == DN_CUS || SCImpGPU == SP_NVG_KM) {
        save_file_int(countPerCluster, nbClusters, 1, "output/CountPerCluster.txt", "", 0);
    }
    finish = omp_get_wtime();
    Tomp_cpu_saveResults += (finish - begin);
    printf("    Result saving completed!\n");
    
    // Deallocate memory
    real_data_memory_deallocation_gpu(GPU_dataT);
    int_data_memory_deallocation_gpu(GPU_countPerCluster);
    int_data_memory_deallocation_gpu(GPU_labels);
    finalize_gpu();
    free(dataT);
    free(countPerCluster);
    free(labels);
}


// Parallel representative-based spectral clustering on GPU
void rep_based_sc_on_gpu()
{
    // Declare variables
    double begin, finish;
    int nbPoints = NB_POINTS;
    int nbDims = NB_DIMS;
    int nbClusters = NB_CLUSTERS;
    int maxNbClusters = nbClusters;
    int optNbClusters = nbClusters;
    int nbReps = NB_REPS;              // Nb of representatives
    T_real *dataT;                     // Array for the transposed matrix of data instances
    T_real *reps;                      // Array for representatives
    int    *countRepsPerCluster;       // Array for the nb of representatives in each cluster
    int    *labels;                    // Array for cluster labels of data instances
    int    *labelsReps;                // Array for cluster labels of data instances
    T_real *GPU_dataT;                 // GPU array for the transposed matrix of data instances
    T_real *GPU_centroids;             // GPU array for the matrix of centroids
    int    *GPU_countPerRep;           // GPU array for the nb of data instances attached to each representative
    int    *GPU_labels;                // GPU array for cluster labels of data instances
    int    *GPU_labelsReps;            // GPU array for cluster labels of data instances
    int    *GPU_countRepsPerCluster;   // GPU array for the nb of representatives in each cluster
    T_real *GPU_reps;                  // GPU array for the matrix of representatives
    T_real *GPU_repsT;                 // GPU array for the transposed matrix of representatives
    
    // Initialize the GPU device and some CUDA libraries
    begin = omp_get_wtime();
    init_gpu();
    finish = omp_get_wtime();
    Tomp_gpu_cuInit += (finish - begin);
    
    // Allocate memory for arrays
    dataT = (T_real *) malloc((sizeof(T_real)*nbDims)*nbPoints);
    reps = (T_real *) malloc((sizeof(T_real)*nbReps)*nbDims);
    countRepsPerCluster = (int *) malloc(sizeof(int)*nbClusters);
    labels = (int *) malloc(sizeof(int)*nbPoints);
    labelsReps = (int *) malloc(sizeof(int)*nbReps);
    real_data_memory_allocation_gpu(&GPU_dataT, (sizeof(T_real)*nbDims)*nbPoints);
    real_data_memory_allocation_gpu(&GPU_centroids, (sizeof(T_real)*nbClusters)*nbDims);
    real_data_memory_allocation_gpu(&GPU_reps, (sizeof(T_real)*nbReps)*nbDims);
    real_data_memory_allocation_gpu(&GPU_repsT, (sizeof(T_real)*nbDims)*nbReps);
    int_data_memory_allocation_gpu(&GPU_countPerRep, sizeof(int)*nbReps);
    int_data_memory_allocation_gpu(&GPU_labels, sizeof(int)*nbPoints);
    int_data_memory_allocation_gpu(&GPU_labelsReps, sizeof(int)*nbReps);
    int_data_memory_allocation_gpu(&GPU_countRepsPerCluster, sizeof(int)*nbClusters);
    
    // Read the data file
    printf("    Data file reading begins ...\n");
    begin = omp_get_wtime();
    if (DATASET_NAME == "Clouds4D_5E7") {
        read_file_real(dataT, nbPoints, nbDims, INPUT_DATA, " ", 0, 1);  // " " delimter for InputDataset-50million.txt
    } else {
        read_file_real(dataT, nbPoints, nbDims, INPUT_DATA, "\t", 0, 1);
    }
    finish = omp_get_wtime();
    Tomp_cpu_readData += (finish - begin);
    printf("    Data file reading completed!\n");
    
    // Transfer data from host (CPU) to device (GPU)
    printf("    Host-to-device data transfers begins ...\n");
    begin = omp_get_wtime();
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    real_data_register(dataT, (sizeof(T_real)*nbDims)*nbPoints);
    real_data_transfers_cpu_to_gpu(dataT, (sizeof(T_real)*nbDims)*nbPoints,   // input
                                   GPU_dataT);                                // output
    real_data_unregister(dataT);
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
    finish = omp_get_wtime();
    Tomp_cpu_gpu_transfers += (finish - begin);
    printf("    Host-to-device data transfers completed!\n");
    
    // Perform feature scaling on GPU (if needed)
    if (FlagFeatureScaling) {
        printf("    Feature scaling begins ...\n");
        float *GPU_dimMax;
        float *GPU_dimMin;
        float_data_memory_allocation_gpu(&GPU_dimMax, sizeof(float)*nbDims);
        float_data_memory_allocation_gpu(&GPU_dimMin, sizeof(float)*nbDims);
        begin = omp_get_wtime();
        feature_scaling_on_gpu(nbDims, nbPoints,          // input
                               GPU_dataT,                 // input & output
                               GPU_dimMax, GPU_dimMin);   // output
        finish = omp_get_wtime();
        Tomp_gpu_featureScaling += (finish - begin);

        float_data_memory_deallocation_gpu(GPU_dimMax);
        float_data_memory_deallocation_gpu(GPU_dimMin);
        printf("    Feature scaling completed!\n");
        // real_data_register(dataT, (sizeof(T_real)*nbDims)*nbPoints);
        // real_data_transfers_gpu_to_cpu(GPU_dataT, (sizeof(T_real)*nbDims)*nbPoints,  // input
                                       // dataT);                                       // output
        // real_data_unregister(dataT);
        // save_file_real(dataT, nbDims, nbPoints, "output/DataT_feature_scaled.txt", "\t", 0);
    }

    // Extract representatives on GPU with one of the following algorithms
    switch(MethodToExtractReps) {
        case ER_RS :  // Case 1: extract representatives on GPU using random sampling
            printf("    Random sampling on the CPU begins ...\n");
            begin = omp_get_wtime();
            seeding(nbPoints, nbDims, nbReps,        // input
                    GPU_dataT, GPU_labels,           // input
                    1, SeedBase,                     // input
                    0, NbPackages,                   // input
                    NbStreamsStep1, NbStreamsStep2,  // input
                    GPU_reps);                       // output
            finish = omp_get_wtime();
            Tomp_gpu_randomSampling += (finish - begin);
            
            begin = omp_get_wtime();
            gpu_attach_to_representative(nbPoints, nbDims, nbReps,  // input
                                     GPU_dataT, GPU_reps,           // input
                                     GPU_labels);                   // output
            finish = omp_get_wtime();
            Tomp_gpu_attach += (finish - begin);
            printf("    Random sampling on the CPU completed!\n");
            break;

        case ER_KM :  // Case 2: extract representatives on GPU using k-means algorithm
            printf("    k-means clustering on GPU begins ...\n");
            begin = omp_get_wtime();
            CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
            kmeans_gpu(KM_GPU,                                                       // input (env)
                       nbPoints, nbDims, nbReps, GPU_dataT,                          // input
                       1, SeedBase, TolKMGPU, MaxNbItersKM,                          // input 
                       TholdUsePackages, NbPackages, NbStreamsStep1, NbStreamsStep2, // input
                       &NbItersKMGPU, GPU_countPerRep, GPU_reps, GPU_labels);        // output
            CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
            CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
            finish = omp_get_wtime();
            Tomp_gpu_kmeans += (finish - begin);
            printf("    k-means clustering on GPU completed!\n");
            break;
            
        case ER_KMPP :  // Case 3: extract representatives on GPU using k-means++ algorithm
            printf("    k-means++ clustering on GPU begins ...\n");
            begin = omp_get_wtime();
            CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
            kmeans_gpu(KM_GPU,                                                        // input (env)
                       nbPoints, nbDims, nbReps, GPU_dataT,                           // input
                       2, SeedBase, TolKMGPU, MaxNbItersKM,                           // input 
                       TholdUsePackages, NbPackages, NbStreamsStep1, NbStreamsStep2,  // input
                       &NbItersKMGPU, GPU_countPerRep, GPU_reps, GPU_labels);         // output
            CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
            CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
            finish = omp_get_wtime();
            Tomp_gpu_kmeanspp += (finish - begin);
            printf("    k-means++ clustering on GPU completed!\n");
            break;
            
        default : 
            fprintf(stderr, "Unknown method for extracting representatives!\n");
            exit(EXIT_FAILURE);
    }
    
    // Save extracted representatives into a .txt file
    // real_data_register(reps, (sizeof(T_real)*nbReps)*nbDims);
    // real_data_transfers_gpu_to_cpu(GPU_reps, sizeof(T_real)*nbReps*nbDims,  // input
                                   // reps);                                   // output
    // real_data_unregister(reps);
    // save_file_real(reps, nbReps, nbDims, "output/Representatives.txt", "\t", 0);
    
    // Transpose the matrix of representatives on GPU
    printf("    Transposition of representative matrix begins ...\n");
    begin = omp_get_wtime();
    CHECK_CUDA_SUCCESS(cudaEventRecord(StartEvent, 0));
    transpose_data(nbReps, nbDims,  // input
                   GPU_reps,        // input
                   GPU_repsT);      // output
    CHECK_CUDA_SUCCESS(cudaEventRecord(StopEvent, 0));
    CHECK_CUDA_SUCCESS(cudaEventSynchronize(StopEvent));
    finish = omp_get_wtime();
    Tomp_gpu_transposeReps += (finish - begin);
    printf("    Transposition of representative matrix completed!\n");
    printf("    Transposition of representative matrix: %f s\n", (float)Tomp_gpu_transposeReps);
    
    // Perform spectral clustering on GPU on the extracted representatives with one of the following implementations
    switch (SCImpGPU) {
        case DN_CUS :  // Case 1: spectral clustering in sparse storage format involving cuSolverDN library
            printf("    Spectral clustering (involving cuSolverDN) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_cusolverdn(nbReps, nbDims, nbClusters, GPU_repsT,                                   // input
                                                            Sigma, TholdSim, TholdDistSq,                                            // input
                                                            FlagAutoTuneNbClusters, maxNbClusters, FlagInteractive,                  // input 
                                                            SeedingKMGPU, SeedBase, TolKMGPU, MaxNbItersKM,                          // input 
                                                            TholdUsePackages, NbPackages, NbStreamsStep1, NbStreamsStep2,            // input
                                                            &NbItersKMGPU, &optNbClusters, GPU_countRepsPerCluster, GPU_labelsReps); // output 
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving cuSolverDN) completed!\n");
            break;
        
        case SP_NVG :  // Case 2: spectral clustering in sparse storage format involving nvGRAPH library
            printf("    Spectral clustering (involving nvGRAPH) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_nvgraph(nbReps, nbDims, nbClusters, GPU_repsT,            // input
                                                         Sigma, TholdSim, TholdDistSq,                     // input
                                                         CSRAlgo, HypoMaxNnzRow, MaxNzPercent,             // input
                                                         MemUsePercent, Pad1, Pad2, Pad3,                  // input
                                                         FilterNoiseApproach, NbBinsHist, TholdNoise,      // input
                                                         FlagAutoTuneNbClusters, FlagInteractive,          // input
                                                         NVGraphAlgo, TolEigen, MaxNbItersEigen,           // input
                                                         TolKMGPU, MaxNbItersKM,                           // input
                                                         &ModularityScore, &EdgeCutScore, &RatioCutScore,  // input
                                                         &optNbClusters, GPU_labelsReps);                  // output
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving nvGRAPH) completed!\n");
            break;
            
        case SP_NVG_KM :  // Case 3: spectral clustering in sparse storage format involving nvGRAPH library + our k-means(++) implementation
            printf("    Spectral clustering (involving nvGRAPH & our k-means(++)) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_nvgraph_and_kmeans(nbReps, nbDims, nbClusters, GPU_repsT,                                   // input
                                                                    Sigma, TholdSim, TholdDistSq,                                            // input
                                                                    CSRAlgo, HypoMaxNnzRow, MaxNzPercent,                                    // input
                                                                    MemUsePercent, Pad1, Pad2, Pad3,                                         // input
                                                                    FilterNoiseApproach, NbBinsHist, TholdNoise,                             // input
                                                                    NVGraphAlgo, TolEigen, MaxNbItersEigen,                                  // input
                                                                    FlagAutoTuneNbClusters, maxNbClusters, FlagInteractive,                  // input
                                                                    SeedingKMGPU, SeedBase, TolKMGPU, MaxNbItersKM,                          // input
                                                                    TholdUsePackages, NbPackages, NbStreamsStep1, NbStreamsStep2,            // input
                                                                    &ModularityScore, &EdgeCutScore, &RatioCutScore,                         // input
                                                                    &optNbClusters, &NbItersKMGPU, GPU_countRepsPerCluster, GPU_labelsReps); // output
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving nvGRAPH & our k-means(++)) completed!\n");
            break;
        
        case SP_CUG :  // Case 4: spectral clustering in sparse storage format involving cuGraph library
            printf("    Spectral clustering (involving cuGraph) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_cugraph(nbReps, nbDims, nbClusters, GPU_repsT,            // input
                                                         Sigma, TholdSim, TholdDistSq,                     // input
                                                         CSRAlgo, HypoMaxNnzRow, MaxNzPercent,             // input
                                                         MemUsePercent, Pad1, Pad2, Pad3,                  // input
                                                         FilterNoiseApproach, NbBinsHist, TholdNoise,      // input
                                                         FlagAutoTuneNbClusters, FlagInteractive,          // input
                                                         CUGraphAlgo, TolEigen, MaxNbItersEigen,           // input
                                                         TolKMGPU, MaxNbItersKM,                           // input
                                                         &ModularityScore, &EdgeCutScore, &RatioCutScore,  // input
                                                         &optNbClusters, GPU_labelsReps);                  // output
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving cuGraph) completed!\n");
            break;
        
        default :
            fprintf(stderr, "Unknown GPU implementation of spectral clustering!\n");
            exit(EXIT_FAILURE);
    }

    // Update nbClusters if the auto-tuning mechanism is enabled
    if (FlagAutoTuneNbClusters == 1) {
        nbClusters = optNbClusters;
    }
    
    // Save the cluster labels of representatives
    // int_data_register(labelsReps, sizeof(int)*nbReps);
    // int_data_transfers_gpu_to_cpu(GPU_labelsReps, sizeof(int)*nbReps,  // input
                                  // labelsReps);                         // output
    // int_data_unregister(labelsReps);
    // save_file_int(labelsReps, nbReps, 1, "output/LabelsReps.txt", "\t", 0);
    
    // Attach each data instance to its nearest representative on GPU
    printf("    Membership attachment of input data begins ...\n");
    begin = omp_get_wtime();
    gpu_membership_attachment(nbPoints, GPU_labelsReps,  // input
                              GPU_labels);               // output
    finish = omp_get_wtime();
    Tomp_gpu_membershipAttach += (finish - begin);
    printf("    Membership attachment of input data completed!\n");
    
    // Transfer results from device (GPU) to host (CPU)
    printf("    Device-to-host result transfers begins ...\n");
    begin = omp_get_wtime();
    int_data_register(labels, sizeof(int)*nbPoints);
    int_data_transfers_gpu_to_cpu(GPU_labels, sizeof(int)*nbPoints,  // input
                                  labels);                           // output
    int_data_unregister(labels);
    if (SCImpGPU == DN_CUS || SCImpGPU == SP_NVG_KM) {
        int_data_register(countRepsPerCluster, sizeof(int)*nbClusters);
        int_data_transfers_gpu_to_cpu(GPU_countRepsPerCluster, sizeof(int)*nbClusters,  // input
                                      countRepsPerCluster);                             // output
        int_data_unregister(countRepsPerCluster);
    }
    finish = omp_get_wtime();
    Tomp_gpu_cpu_transfers += (finish - begin);
    printf("    Device-to-host result transfers completed!\n");

    // Save results into .txt files
    printf("    Result saving begins ...\n");
    begin = omp_get_wtime();
    save_file_int(labels, nbPoints, 1, "output/Labels.txt", "", 0);
    if (SCImpGPU == DN_CUS || SCImpGPU == SP_NVG_KM) {
        save_file_int(countRepsPerCluster, nbClusters, 1, "output/CountRepsPerCluster.txt", "", 0);
    }
    finish = omp_get_wtime();
    Tomp_cpu_saveResults += (finish - begin);
    printf("    Result saving completed!\n");
    
    // Deallocate memory
    real_data_memory_deallocation_gpu(GPU_dataT);
    real_data_memory_deallocation_gpu(GPU_centroids);
    real_data_memory_deallocation_gpu(GPU_reps);
    real_data_memory_deallocation_gpu(GPU_repsT);
    int_data_memory_deallocation_gpu(GPU_countPerRep);
    int_data_memory_deallocation_gpu(GPU_labels);
    int_data_memory_deallocation_gpu(GPU_labelsReps);
    int_data_memory_deallocation_gpu(GPU_countRepsPerCluster);
    finalize_gpu();
    free(dataT);
    free(reps);
    free(countRepsPerCluster);
    free(labels);
    free(labelsReps);
}


// Parallel representative-based spectral clustering on CPU (imcomplete due to the need of calling an API of scikit-learn)
void rep_based_sc_on_cpu()
{
    // Declare variables
    double begin, finish;
    int nbPoints = NB_POINTS;
    int nbDims = NB_DIMS;
    int nbClusters = (FlagAutoTuneNbClusters == 1 ? MAX_NB_CLUSTERS : NB_CLUSTERS);
    int maxNbClusters = nbClusters;
    int optNbClusters = nbClusters;
    int nbReps = NB_REPS;             // Nb of representatives
    T_real *data;                     // Array for the matrix of data instances
    T_real *reps;                     // Array for the matrix of representatives
    int    *countRepsPerCluster;      // Array for the nb of representatives in each cluster
    int    *labels;                   // Array for cluster labels of data points
    int    *labelsReps;               // Array for cluster labels of representatives
    
    // Set the number of OpenMP threads on CPU
    omp_set_num_threads(NbThreadsCPU);
    
    // Allocate memory for arrays
    data = (T_real *) malloc((sizeof(T_real)*nbPoints)*nbDims);
    reps = (T_real *) malloc((sizeof(T_real)*nbReps)*nbDims);
    countRepsPerCluster = (int *) malloc(sizeof(int)*nbClusters);
    labels = (int *) malloc(sizeof(int)*nbPoints);
    labelsReps = (int *) malloc(sizeof(int)*nbReps);
    
    // Read the data file
    printf("    Data file reading begins ...\n");
    begin = omp_get_wtime();
    if (DATASET_NAME == "Clouds4D_5E7") {
        read_file_real(data, nbPoints, nbDims, INPUT_DATA, " ", 0, 0);  // " " delimter for InputDataset-50million.txt
    } else {
        read_file_real(data, nbPoints, nbDims, INPUT_DATA, "\t", 0, 0);
    }
    finish = omp_get_wtime();
    Tomp_cpu_readData += (finish - begin);
    printf("    Data file reading completed!\n");

    // Perform feature scaling on CPU (if needed)
    if (FlagFeatureScaling) {
        printf("    Feature scaling on the CPU begins ...\n");
        T_real *dimMax;
        T_real *dimMin;
        dimMax = (T_real *) malloc(sizeof(T_real)*nbDims);
        dimMin = (T_real *) malloc(sizeof(T_real)*nbDims);
        begin = omp_get_wtime();
        feature_scaling(nbPoints, nbDims,  // input
                        data,              // input & output
                        dimMax, dimMin);
        finish = omp_get_wtime();
        Tomp_cpu_featureScaling += (finish - begin);
        free(dimMax);
        free(dimMin);
        printf("    Feature scaling on the CPU completed!\n");
        // save_file_real(data, nbPoints, nbDims, "output/Data_feature_scaled.txt", "\t", 0);
    }

    // Extract representatives on CPU with one of the following algorithms
    switch(MethodToExtractReps) {
        case ER_RS :  // Case 1: extract representatives on CPU using random sampling
            printf("    Random sampling on the CPU begins ...\n");
            begin = omp_get_wtime();
            random_sampling(nbPoints, nbDims, nbReps,  // input
                            data,                      // input
                            SeedBase,                  // input
                            reps);                     // output
            finish = omp_get_wtime();
            Tomp_cpu_randomSampling += (finish - begin);
            
            begin = omp_get_wtime();
            attach_to_representative(nbPoints, nbDims, nbReps,  // input
                                     data, reps,                // input
                                     labels);                   // output
            finish = omp_get_wtime();
            Tomp_cpu_attach += (finish - begin);
            printf("    Random sampling on the CPU completed!\n");
            break;

        case ER_KM :  // Case 2: extract representatives on CPU using k-means algorithm
            printf("    k-means clustering on the CPU begins ...\n");
            begin = omp_get_wtime();
            SeedingKMCPU = 1;
            kmeans_cpu_for_extracting_representatives(nbPoints, nbDims, nbReps, data,  // input
                                                      SeedingKMCPU, SeedBase,          // input
                                                      TholdUsePackages, NbPackages,    // input
                                                      TolKMCPU, MaxNbItersKM,          // input
                                                      &NbItersKMCPU, reps, labels);    // output
            finish = omp_get_wtime();
            Tomp_cpu_kmeans += (finish - begin);
            printf("    k-means clustering on the CPU completed!\n");
            break;
        
        case ER_KMPP :  // Case 3: extract representatives on CPU using k-means++ algorithm
            printf("    k-means++ clustering on the CPU begins ...\n");
            begin = omp_get_wtime();
            SeedingKMCPU = 2;
            kmeans_cpu_for_extracting_representatives(nbPoints, nbDims, nbReps, data,  // input
                                                      SeedingKMCPU, SeedBase,          // input
                                                      TholdUsePackages, NbPackages,    // input
                                                      TolKMCPU, MaxNbItersKM,          // input
                                                      &NbItersKMCPU, reps, labels);    // output
            finish = omp_get_wtime();
            Tomp_cpu_kmeans += (finish - begin);
            printf("    k-means++ clustering on the CPU completed!\n");
            break;
            
        default : 
            fprintf(stderr, "Unknown method for extracting representatives!\n");
            exit(EXIT_FAILURE);
    }
    
    // Save extracted representatives and temporary cluster labels into .txt files
    save_file_real(reps, nbReps, nbDims, "output/Representatives.txt", "\t", 0);
    save_file_int(labels, nbPoints, 1, "output/LabelsToReps.txt", "", 0);
    
    // Call the Spectral Clustering API of scikit-learn and then perform input data attachment
    // ...
    printf("    It remains to:\n");
    printf("       1. import the pre-extracted representatives\n");
    printf("       2. call the Spectral Clustering API of scikit-learn on the representatives\n");
    printf("       3. attach each input data instance to its nearest representative\n");
    printf("    Please refer to the file 'sc_on_reps.py' situated in the 'python' folder.\n");
    
    // Save results into .txt files
    // printf("    Result saving begins ...\n");
    // begin = omp_get_wtime();
    // save_file_int(labels, nbPoints, 1, "output/Labels.txt", "", 0);
    // finish = omp_get_wtime();
    // Tomp_cpu_saveResults += (finish - begin);
    // printf("    Result saving completed!\n");
    
    // Deallocate memory
    free(data);
    free(reps);
    free(countRepsPerCluster);
    free(labels);
    free(labelsReps);
}


// Parallel representative-based spectral clustering on CPU+GPU
void rep_based_sc_on_cpu_gpu()
{
    // Declare variables
    double begin, finish;
    int nbPoints = NB_POINTS;
    int nbDims = NB_DIMS;
    int nbClusters = (FlagAutoTuneNbClusters == 1 ? MAX_NB_CLUSTERS : NB_CLUSTERS);
    int maxNbClusters = nbClusters;
    int optNbClusters = nbClusters;
    int nbReps = NB_REPS;             // Nb of representatives
    T_real *data;                     // Array for the matrix of data instances
    T_real *reps;                     // Array for the matrix of representatives
    int    *countRepsPerCluster;      // Array for the nb of representatives in each cluster
    int    *labels;                   // Array for cluster labels of data points
    int    *labelsReps;               // Array for cluster labels of representatives
    T_real *GPU_reps;                 // GPU array for the matrix of representatives
    T_real *GPU_repsT;                // GPU array for the transposed matrix of representatives
    int    *GPU_countRepsPerCluster;  // GPU array for the nb of representatives in each cluster
    int    *GPU_labelsReps;           // GPU array for cluster labels of representatives
    
    // Set the number of OpenMP threads on CPU
    omp_set_num_threads(NbThreadsCPU);
    
    // Initialize the GPU device and some CUDA libraries
    begin = omp_get_wtime();
    init_gpu();
    finish = omp_get_wtime();
    Tomp_gpu_cuInit += (finish - begin);
    
    // Allocate memory for arrays
    data = (T_real *) malloc((sizeof(T_real)*nbPoints)*nbDims);
    reps = (T_real *) malloc((sizeof(T_real)*nbReps)*nbDims);
    countRepsPerCluster = (int *) malloc(sizeof(int)*nbClusters);
    labels = (int *) malloc(sizeof(int)*nbPoints);
    labelsReps = (int *) malloc(sizeof(int)*nbReps);
    real_data_memory_allocation_gpu(&GPU_reps, (sizeof(T_real)*nbReps)*nbDims);
    real_data_memory_allocation_gpu(&GPU_repsT, (sizeof(T_real)*nbDims)*nbReps);
    int_data_memory_allocation_gpu(&GPU_countRepsPerCluster, sizeof(int)*nbClusters);
    int_data_memory_allocation_gpu(&GPU_labelsReps, sizeof(int)*nbReps);
    
    // Read the data file
    printf("    Data file reading begins ...\n");
    begin = omp_get_wtime();
    if (DATASET_NAME == "Clouds4D_5E7") {
        read_file_real(data, nbPoints, nbDims, INPUT_DATA, " ", 0, 0);  // " " delimter for InputDataset-50million.txt
    } else {
        read_file_real(data, nbPoints, nbDims, INPUT_DATA, "\t", 0, 0);
    }
    finish = omp_get_wtime();
    Tomp_cpu_readData += (finish - begin);
    printf("    Data file reading completed!\n");

    // Perform feature scaling on CPU (if needed)
    if (FlagFeatureScaling) {
        printf("    Feature scaling on the CPU begins ...\n");
        T_real *dimMax;
        T_real *dimMin;
        dimMax = (T_real *) malloc(sizeof(T_real)*nbDims);
        dimMin = (T_real *) malloc(sizeof(T_real)*nbDims);
        begin = omp_get_wtime();
        feature_scaling(nbPoints, nbDims,  // input
                        data,              // input & output
                        dimMax, dimMin);
        finish = omp_get_wtime();
        Tomp_cpu_featureScaling += (finish - begin);
        free(dimMax);
        free(dimMin);
        printf("    Feature scaling on the CPU completed!\n");
        // save_file_real(data, nbPoints, nbDims, "output/Data_feature_scaled.txt", "\t", 0);
    }

    // Extract representatives on CPU with one of the following algorithms
    switch(MethodToExtractReps) {
        case ER_RS :  // Case 1: extract representatives on CPU using random sampling
            printf("    Random sampling on the CPU begins ...\n");
            begin = omp_get_wtime();
            random_sampling(nbPoints, nbDims, nbReps,  // input
                            data,                      // input
                            SeedBase,                  // input
                            reps);                     // output
            finish = omp_get_wtime();
            Tomp_cpu_randomSampling += (finish - begin);
            
            begin = omp_get_wtime();
            attach_to_representative(nbPoints, nbDims, nbReps,  // input
                                     data, reps,                // input
                                     labels);                   // output
            finish = omp_get_wtime();
            Tomp_cpu_attach += (finish - begin);
            printf("    Random sampling on the CPU completed!\n");
            break;

        case ER_KM :  // Case 2: extract representatives on CPU using k-means algorithm
            printf("    k-means clustering on the CPU begins ...\n");
            begin = omp_get_wtime();
            SeedingKMCPU = 1;
            kmeans_cpu_for_extracting_representatives(nbPoints, nbDims, nbReps, data,  // input
                                                      SeedingKMCPU, SeedBase,          // input
                                                      TholdUsePackages, NbPackages,    // input
                                                      TolKMCPU, MaxNbItersKM,          // input
                                                      &NbItersKMCPU, reps, labels);    // output
            finish = omp_get_wtime();
            Tomp_cpu_kmeans += (finish - begin);
            printf("    k-means clustering on the CPU completed!\n");
            break;
        
        case ER_KMPP :  // Case 3: extract representatives on CPU using k-means++ algorithm
            printf("    k-means++ clustering on the CPU begins ...\n");
            begin = omp_get_wtime();
            SeedingKMCPU = 2;
            kmeans_cpu_for_extracting_representatives(nbPoints, nbDims, nbReps, data,  // input
                                                      SeedingKMCPU, SeedBase,          // input
                                                      TholdUsePackages, NbPackages,    // input
                                                      TolKMCPU, MaxNbItersKM,          // input
                                                      &NbItersKMCPU, reps, labels);    // output
            finish = omp_get_wtime();
            Tomp_cpu_kmeans += (finish - begin);
            printf("    k-means++ clustering on the CPU completed!\n");
            break;
            
        default : 
            fprintf(stderr, "Unknown method for extracting representatives!\n");
            exit(EXIT_FAILURE);
    }
    
    // Save extracted representatives and temporary cluster labels into .txt files
    // save_file_real(reps, nbReps, nbDims, "output/Representatives.txt", "\t", 0);
    // save_file_int(labels, nbPoints, 1, "output/LabelsToReps.txt", "", 0);
    
    // Transfer representatives from host (CPU) to device (GPU)
    printf("    Host-to-device representatives transfers begins ...\n");
    begin = omp_get_wtime();
    real_data_register(reps, (sizeof(T_real)*nbReps)*nbDims);
    real_data_transfers_cpu_to_gpu(reps, (sizeof(T_real)*nbReps)*nbDims,  // input
                                   GPU_reps);                             // output
    real_data_unregister(reps);
    finish = omp_get_wtime();
    Tomp_cpu_gpu_transfers += (finish - begin);
    printf("    Host-to-device representatives transfers completed!\n");
    
    // Transpose the matrix of representatives on GPU
    printf("    Transposition of representative matrix begins ...\n");
    begin = omp_get_wtime();
    transpose_data(nbReps, nbDims,  // input
                   GPU_reps,        // input
                   GPU_repsT);      // output
    finish = omp_get_wtime();
    Tomp_gpu_transposeReps += (finish - begin);
    printf("    Transposition of representative matrix completed!\n");
    
    // Perform spectral clustering on GPU on the extracted representatives with one of the following implementations
    switch (SCImpGPU) {
        case DN_CUS :  // Case 1: spectral clustering in sparse storage format involving cuSolverDN library
            printf("    Spectral clustering (involving cuSolverDN) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_cusolverdn(nbReps, nbDims, nbClusters, GPU_repsT,                                    // input
                                                            Sigma, TholdSim, TholdDistSq,                                             // input
                                                            FlagAutoTuneNbClusters, maxNbClusters, FlagInteractive,                   // input 
                                                            SeedingKMGPU, SeedBase, TolKMGPU, MaxNbItersKM,                           // input 
                                                            TholdUsePackages, NbPackages, NbStreamsStep1, NbStreamsStep2,             // input
                                                            &NbItersKMGPU, &optNbClusters, GPU_countRepsPerCluster, GPU_labelsReps);  // output 
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving cuSolverDN) completed!\n");
            break;
        
        case SP_NVG :  // Case 2: spectral clustering in sparse storage format involving nvGRAPH library
            printf("    Spectral clustering (involving nvGRAPH) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_nvgraph(nbReps, nbDims, nbClusters, GPU_repsT,            // input
                                                         Sigma, TholdSim, TholdDistSq,                     // input
                                                         CSRAlgo, HypoMaxNnzRow, MaxNzPercent,             // input
                                                         MemUsePercent, Pad1, Pad2, Pad3,                  // input
                                                         FilterNoiseApproach, NbBinsHist, TholdNoise,      // input
                                                         FlagAutoTuneNbClusters, FlagInteractive,          // input
                                                         NVGraphAlgo, TolEigen, MaxNbItersEigen,           // input
                                                         TolKMGPU, MaxNbItersKM,                           // input
                                                         &ModularityScore, &EdgeCutScore, &RatioCutScore,  // input
                                                         &optNbClusters, GPU_labelsReps);                  // output
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving nvGRAPH) completed!\n");
            break;
        
        case SP_NVG_KM :  // Case 3: spectral clustering in sparse storage format involving nvGRAPH library + our k-means(++) implementation
            printf("    Spectral clustering (involving nvGRAPH & our k-means(++)) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_nvgraph_and_kmeans(nbReps, nbDims, nbClusters, GPU_repsT,                                   // input
                                                                    Sigma, TholdSim, TholdDistSq,                                            // input
                                                                    CSRAlgo, HypoMaxNnzRow, MaxNzPercent,                                    // input
                                                                    MemUsePercent, Pad1, Pad2, Pad3,                                         // input
                                                                    FilterNoiseApproach, NbBinsHist, TholdNoise,                             // input
                                                                    NVGraphAlgo, TolEigen, MaxNbItersEigen,                                  // input
                                                                    FlagAutoTuneNbClusters, maxNbClusters, FlagInteractive,                  // input
                                                                    SeedingKMGPU, SeedBase, TolKMGPU, MaxNbItersKM,                          // input
                                                                    TholdUsePackages, NbPackages, NbStreamsStep1, NbStreamsStep2,            // input
                                                                    &ModularityScore, &EdgeCutScore, &RatioCutScore,                         // input
                                                                    &optNbClusters, &NbItersKMGPU, GPU_countRepsPerCluster, GPU_labelsReps); // output
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving nvGRAPH & our k-means(++)) completed!\n");
            break;
        
        case SP_CUG :  // Case 4: spectral clustering in sparse storage format involving cuGraph library
            printf("    Spectral clustering (involving cuGraph) begins ...\n");
            begin = omp_get_wtime();
            spectral_clustering_on_gpu_involving_cugraph(nbReps, nbDims, nbClusters, GPU_repsT,            // input
                                                         Sigma, TholdSim, TholdDistSq,                     // input
                                                         CSRAlgo, HypoMaxNnzRow, MaxNzPercent,             // input
                                                         MemUsePercent, Pad1, Pad2, Pad3,                  // input
                                                         FilterNoiseApproach, NbBinsHist, TholdNoise,      // input
                                                         FlagAutoTuneNbClusters, FlagInteractive,          // input
                                                         CUGraphAlgo, TolEigen, MaxNbItersEigen,           // input
                                                         TolKMGPU, MaxNbItersKM,                           // input
                                                         &ModularityScore, &EdgeCutScore, &RatioCutScore,  // input
                                                         &optNbClusters, GPU_labelsReps);                  // output
            finish = omp_get_wtime();
            Tomp_gpu_spectralClustering += (finish - begin);
            printf("    Spectral clustering (involving cuGraph) completed!\n");
            break;
        
        default :
            fprintf(stderr, "Unknown GPU implementation of spectral clustering!\n");
            exit(EXIT_FAILURE);
    }

    // Update nbClusters if the auto-tuning mechanism is enabled
    if (FlagAutoTuneNbClusters == 1) {
        nbClusters = optNbClusters;
    }
    
    // Transfer results from device (GPU) to host (CPU)
    printf("    Device-to-host result transfers begins ...\n");
    begin = omp_get_wtime();
    int_data_register(labelsReps, sizeof(int)*nbReps);
    int_data_transfers_gpu_to_cpu(GPU_labelsReps, sizeof(int)*nbReps,  // input
                                  labelsReps);                         // output
    int_data_unregister(labelsReps);
    if (SCImpGPU == DN_CUS || SCImpGPU == SP_NVG_KM) {
        int_data_register(countRepsPerCluster, sizeof(int)*nbClusters);
        int_data_transfers_gpu_to_cpu(GPU_countRepsPerCluster, sizeof(int)*nbClusters,  // input
                                      countRepsPerCluster);                             // output
        int_data_unregister(countRepsPerCluster);
    }
    finish = omp_get_wtime();
    Tomp_gpu_cpu_transfers += (finish - begin);
    printf("    Device-to-host result transfers completed!\n");

    // Attach each data instance to its nearest representative on CPU
    printf("    Membership attachment of input data begins ...\n");
    begin = omp_get_wtime();
    membership_attachment(nbPoints, labelsReps,  // input
                          labels);               // output
    finish = omp_get_wtime();
    Tomp_cpu_membershipAttach += (finish - begin);
    printf("    Membership attachment of input data completed!\n");
    
    // Save results into .txt files
    printf("    Result saving begins ...\n");
    begin = omp_get_wtime();
    save_file_int(labels, nbPoints, 1, "output/Labels.txt", "", 0);
    save_file_real(reps, nbReps, nbDims, "output/Representatives.txt", "\t", 0);
    if (SCImpGPU == DN_CUS || SCImpGPU == SP_NVG_KM) {
        save_file_int(countRepsPerCluster, nbClusters, 1, "output/CountRepsPerCluster.txt", "", 0);
    }
    finish = omp_get_wtime();
    Tomp_cpu_saveResults += (finish - begin);
    printf("    Result saving completed!\n");
    
    // Deallocate memory
    real_data_memory_deallocation_gpu(GPU_reps);
    real_data_memory_deallocation_gpu(GPU_repsT);
    int_data_memory_deallocation_gpu(GPU_countRepsPerCluster);
    int_data_memory_deallocation_gpu(GPU_labelsReps);
    finalize_gpu();
    free(data);
    free(reps);
    free(countRepsPerCluster);
    free(labels);
    free(labelsReps);
}


// Parallel representative-based spectral clustering on CPU / on GPU / on CPU+GPU
void rep_based_sc()
{
    // Perform representative-based spectral clustering with one of the following implementations
    switch(Chain) {
        case CHAIN_ON_GPU:  // Case 1: parallel representative-based spectral clustering on GPU
            rep_based_sc_on_gpu();
            break;
            
        case CHAIN_ON_CPU:  // Case 2: parallel representative-based spectral clustering on CPU (imcomplete due to the need of calling an API of scikit-learn)
            rep_based_sc_on_cpu();
            break;
            
        case CHAIN_ON_CPU_GPU: // Case 3: parallel representative-based spectral clustering on CPU+GPU
            rep_based_sc_on_cpu_gpu();
            break;
            
        default :
            fprintf(stderr, "Unknown processing chain of representative-based spectral clustering!\n");
            exit(EXIT_FAILURE);
    }
}

