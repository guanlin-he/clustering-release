#ifndef _VARS_H
#define _VARS_H

// Related to global application
extern algo_t  ClustAlgo;
extern int NbThreadsCPU;        // Number of OpenMP threads
extern int FlagFeatureScaling;  // Flag indicating feature scaling


// Related to k-means(++) clustering
// - related to both k-means(++) on CPU and k-means(++) on GPU
extern unsigned int SeedBase;   // Seed base for random number generator
extern int TholdUsePackages;    // Threshold for using multiple packages in UpdateCentroids
extern int NbPackages;          // Number of packages used in UpdateCentroids
extern int MaxNbItersKM;        // Maximal number of k-means(++) iterations
// - only related to k-means(++) on CPU
extern int SeedingKMCPU;        // Seeding method for k-means(++) on CPU
extern T_real TolKMCPU;         // Tolerance (i.e. convergence criterion) for k-means(++) on CPU
extern int NbItersKMCPU;        // Number of k-means(++) iterations on CPU
// - only related to k-means(++) on GPU
extern int SeedingKMGPU;        // Seeding method for k-means(++) on GPU
extern T_real TolKMGPU;         // Tolerance (i.e. convergence criterion) for k-means(++) on GPU
extern int NbItersKMGPU;        // Number of k-means(++) iterations on GPU
extern int NbStreamsStep1;      // Number of streams for Step 1 of UpdateCentroids
extern int NbStreamsStep2;      // Number of streams for Step 2 of UpdateCentroids


// Related to spectral clustering
extern scimp_t SCImpGPU;
// - related to similarity matrix construction
extern int CSRAlgo;             // Algorithm for constructing similarity matrix in CSR format
extern T_real Sigma;            // Parameter controling the width of neighborhood in the Gaussian similarity function
extern T_real TholdSim;         // Threshold for similarity
extern T_real TholdDistSq;      // Threshold for distance (epsilon for eps-neighborhood graph)
extern int HypoMaxNnzRow;       // Hypothetical maximal nnz in one row of similarity matrix
extern int Pad1;                // Padding 1 in shared memory
extern int Pad2;                // Padding 2 in shared memory
extern int Pad3;                // Padding 3 in shared memory
extern T_real MemUsePercent;    // Percentage of GPU free memory usage
extern T_real MaxNzPercent;     // Maximal percentage of nonzeros in the similarity matrix
extern int SimConstrAlgoCPU;    // Algorithm for constructing similarity matrix on CPU
// - related to nvGRAPH & cuGraph libraries
extern int NVGraphAlgo;         // Algorithm number of nvGRAPH Spectral Clustering API
extern int CUGraphAlgo;         // Algorithm number of cuGraph Spectral Clustering API
extern int MaxNbItersEigen;     // Maximal number of iterations for eigensolver
extern T_real TolEigen;         // Tolerance (i.e. convergence criterion) for eigensolver
extern float ModularityScore;   // Modularity score returned by nvgraphAnalyzeClustering
extern float EdgeCutScore;      // Edge cut score returned by nvgraphAnalyzeClustering
extern float RatioCutScore;     // Ratio cut score returned by nvgraphAnalyzeClustering
// - related to noise filtering
extern int FilterNoiseApproach; // Approach to filtering noise for spectral clustering
extern int NbBinsHist;          // Nb of bins for the histogram of noise filtering
extern T_real TholdNoise;       // Threshold for determining noise based on nnz per row in similarity matrix
// - related to auto-tuning of the number of clusters
extern int FlagAutoTuneNbClusters; // Flag indicating the auto-tuning of the nb of clusters
// - related to both noise filtering and auto-tuning
extern int FlagInteractive;     // Flag indicating interactive mode for auto-tuning of some parameters


// Related to representative-based spectral clustering
extern extr_t MethodToExtractReps;   
extern chain_t Chain;


// Related to block size configuration for CUDA kernels
extern int BsXN;                // BLOCK_SIZE_X related to NB_POINTS or NB_REPS (BsXN has to be in [1, 1024] & be a power of 2)
extern int BsXP;                // BLOCK_SIZE_X related to NB_POINTS devided by NbPackages (BsXP has to be in [1, 1024] & be a power of 2)
extern int BsXD;                // BLOCK_SIZE_X related to NB_DIMS (BsXD has to be in [1, 1024] & be a power of 2)
extern int BsXC;                // BLOCK_SIZE_X related to NB_CLUSTERS (BsXC has to be in [1, 1024] & be a power of 2)
extern int BsYN;                // BLOCK_SIZE_Y related to NB_POINTS or NB_REPS (BsYN has to be in [1, 1024] & BsXN*BsYN has to be at most 1024 in case of GS1)
extern int BsXK1;               // BLOCK_SIZE_X related to kernel_first_pass_CSR1_2D_grid_2D_blocks
extern int BsXK2;               // BLOCK_SIZE_X related to kernel_second_pass_CSR1_1D_grid_2D_blocks
extern int BsXK3;               // BLOCK_SIZE_X related to kernel_full_pass_CSR2_1D_grid_2D_blocks
extern int BsXK4;               // BLOCK_SIZE_X related to kernel_ellpack_to_csr_CSR2_1D_grid_2D_blocks
extern int BsXK5;               // BLOCK_SIZE_X related to kernel_supplementary_pass_CSR2_1D_grid_2D_blocks
extern int BsXK6;               // BLOCK_SIZE_X related to kernel_construct_similarity_chunk
extern int BsYK1;               // BLOCK_SIZE_Y related to kernel_first_pass_CSR1_2D_grid_2D_blocks
extern int BsYK2;               // BLOCK_SIZE_Y related to kernel_second_pass_CSR1_1D_grid_2D_blocks
extern int BsYK3;               // BLOCK_SIZE_Y related to kernel_full_pass_CSR2_1D_grid_2D_blocks
extern int BsYK4;               // BLOCK_SIZE_Y related to kernel_ellpack_to_csr_CSR2_1D_grid_2D_blocks
extern int BsYK5;               // BLOCK_SIZE_Y related to kernel_supplementary_pass_CSR2_1D_grid_2D_blocks
extern int BsYK6;               // BLOCK_SIZE_Y related to kernel_construct_similarity_chunk


// Related to the elapsed time measured by omp
extern double Tomp_application;
// -- on CPU
extern double Tomp_cpu_readData;
extern double Tomp_cpu_featureScaling;
extern double Tomp_cpu_randomSampling, Tomp_cpu_d2Sampling, Tomp_cpu_attach;
extern double Tomp_cpu_seeding, Tomp_cpu_computeAssign, Tomp_cpu_updateCentroids, Tomp_cpu_kmeans;
extern double Tomp_cpu_constructSimMatrix;
extern double Tomp_cpu_membershipAttach;
extern double Tomp_cpu_saveResults;
extern double Tomp_cpu_saveSimMatrix;
// -- on GPU
extern double Tomp_gpu_randomSampling, Tomp_gpu_attach;
extern double Tomp_gpu_cuInit;
extern double Tomp_gpu_computeUnscaledCentroids;
extern double Tomp_gpu_transposeReps;
extern double Tomp_gpu_featureScaling;
extern double Tomp_gpu_seeding, Tomp_gpu_computeAssign, Tomp_gpu_updateCentroids, Tomp_gpu_kmeans, Tomp_gpu_kmeanspp;
extern double Tomp_gpu_spectralClustering;
extern double Tomp_gpu_constructSimLapMatrix, Tomp_gpu_constructSimMatrixInCSR;
extern double Tomp_gpu_filterNoise;
extern double Tomp_gpu_cuSolverDNsyevdx, Tomp_gpu_nvGRAPHSpectralClusteringAPI, Tomp_gpu_cuGraphSpectralClusteringAPI;
extern double Tomp_gpu_autoTuneNbClusters;
extern double Tomp_gpu_normalizeEigenvectorMatrix;
extern double Tomp_gpu_finalKmeansForSC;
extern double Tomp_gpu_membershipAttach;
// -- on CPU-GPU
extern double Tomp_cpu_gpu_transfers, Tomp_gpu_cpu_transfers;

#endif