#include "../include/config.h"
#include "../include/vars.h"


// Related to global application
algo_t  ClustAlgo;
int NbThreadsCPU;        // Number of OpenMP threads
int FlagFeatureScaling;  // Flag indicating feature scaling


// Related to k-means clustering
// - related to both k-means on CPU and k-means on GPU
unsigned int SeedBase;   // Seed base for random number generator
int TholdUsePackages;    // Threshold for using multiple packages in UpdateCentroids
int NbPackages;          // Number of packages used in UpdateCentroids
int MaxNbItersKM;        // Maximal number of k-means iterations
// - only related to k-means on CPU
int SeedingKMCPU;        // Seeding method for k-means on CPU
T_real TolKMCPU;         // Tolerance (i.e. convergence criterion) for k-means on CPU
int NbItersKMCPU;        // Number of k-means iterations on CPU
// - only related to k-means on GPU
int SeedingKMGPU;        // Seeding method for k-means on GPU
T_real TolKMGPU;         // Tolerance (i.e. convergence criterion) for k-means on GPU
int NbItersKMGPU;        // Number of k-means iterations on GPU
int NbStreamsStep1;      // Number of streams for Step 1 of UpdateCentroids
int NbStreamsStep2;      // Number of streams for Step 2 of UpdateCentroids


// Related to spectral clustering
scimp_t SCImpGPU;
// - related to similarity matrix construction
int CSRAlgo;             // Algorithm for constructing similarity matrix in CSR format
T_real Sigma;            // Parameter controling the width of neighborhood in the Gaussian similarity function
T_real TholdSim;         // Threshold for similarity
T_real TholdDistSq;      // Threshold for distance (epsilon for eps-neighborhood graph)
int HypoMaxNnzRow;       // Hypothetical maximal nnz in one row of similarity matrix
int Pad1;                // Padding 1 in shared memory
int Pad2;                // Padding 2 in shared memory
int Pad3;                // Padding 3 in shared memory
T_real MemUsePercent;    // Percentage of GPU free memory usage
T_real MaxNzPercent;     // Maximal percentage of nonzeros in the similarity matrix
int SimConstrAlgoCPU;    // Algorithm for constructing similarity matrix on CPU
// - related to nvGRAPH & cuGraph libraries
int NVGraphAlgo;         // Algorithm number of nvGRAPH Spectral Clustering API
int CUGraphAlgo;         // Algorithm number of cuGraph Spectral Clustering API
int MaxNbItersEigen;     // Maximal number of iterations for eigensolver
T_real TolEigen;         // Tolerance (i.e. convergence criterion) for eigensolver
float ModularityScore;   // Modularity score returned by nvgraphAnalyzeClustering
float EdgeCutScore;      // Edge cut score returned by nvgraphAnalyzeClustering
float RatioCutScore;     // Ratio cut score returned by nvgraphAnalyzeClustering
// - related to noise filtering
int FilterNoiseApproach; // Approach to filtering noise for spectral clustering
int NbBinsHist;          // Nb of bins for the histogram of noise filtering
T_real TholdNoise;       // Threshold for determining noise based on nnz per row in similarity matrix
// - related to auto-tuning of the number of clusters
int FlagAutoTuneNbClusters; // Flag indicating the auto-tuning of the nb of clusters
// - related to both noise filtering and auto-tuning
int FlagInteractive;     // Flag indicating interactive mode for auto-tuning of some parameters


// Related to representative-based spectral clustering
extr_t MethodToExtractReps;   
chain_t Chain;


// Related to block size configuration for CUDA kernels
int BsXN;                // BLOCK_SIZE_X related to NB_POINTS or NB_REPS (BsXN has to be in [1, 1024] & be a power of 2)
int BsXP;                // BLOCK_SIZE_X related to NB_POINTS devided by NbPackages (BsXP has to be in [1, 1024] & be a power of 2)
int BsXD;                // BLOCK_SIZE_X related to NB_DIMS (BsXD has to be in [1, 1024] & be a power of 2)
int BsXC;                // BLOCK_SIZE_X related to NB_CLUSTERS (BsXC has to be in [1, 1024] & be a power of 2)
int BsYN;                // BLOCK_SIZE_Y related to NB_POINTS or NB_REPS (BsYN has to be in [1, 1024] & BsXN*BsYN has to be at most 1024 in case of GS1)
int BsXK1;               // BLOCK_SIZE_X related to kernel_first_pass_CSR1_2D_grid_2D_blocks
int BsXK2;               // BLOCK_SIZE_X related to kernel_second_pass_CSR1_1D_grid_2D_blocks
int BsXK3;               // BLOCK_SIZE_X related to kernel_full_pass_CSR2_1D_grid_2D_blocks
int BsXK4;               // BLOCK_SIZE_X related to kernel_ellpack_to_csr_CSR2_1D_grid_2D_blocks
int BsXK5;               // BLOCK_SIZE_X related to kernel_supplementary_pass_CSR2_1D_grid_2D_blocks
int BsXK6;               // BLOCK_SIZE_X related to kernel_construct_similarity_chunk
int BsYK1;               // BLOCK_SIZE_Y related to kernel_first_pass_CSR1_2D_grid_2D_blocks
int BsYK2;               // BLOCK_SIZE_Y related to kernel_second_pass_CSR1_1D_grid_2D_blocks
int BsYK3;               // BLOCK_SIZE_Y related to kernel_full_pass_CSR2_1D_grid_2D_blocks
int BsYK4;               // BLOCK_SIZE_Y related to kernel_ellpack_to_csr_CSR2_1D_grid_2D_blocks
int BsYK5;               // BLOCK_SIZE_Y related to kernel_supplementary_pass_CSR2_1D_grid_2D_blocks
int BsYK6;               // BLOCK_SIZE_Y related to kernel_construct_similarity_chunk


// Related to the elapsed time measured by omp
double Tomp_application;
// -- on CPU
double Tomp_cpu_readData;
double Tomp_cpu_featureScaling;
double Tomp_cpu_randomSampling, Tomp_cpu_d2Sampling, Tomp_cpu_attach;
double Tomp_cpu_seeding, Tomp_cpu_computeAssign, Tomp_cpu_updateCentroids, Tomp_cpu_kmeans;
double Tomp_cpu_constructSimMatrix;
double Tomp_cpu_membershipAttach;
double Tomp_cpu_saveResults;
double Tomp_cpu_saveSimMatrix;
// -- on GPU
double Tomp_gpu_randomSampling, Tomp_gpu_attach;
double Tomp_gpu_cuInit;
double Tomp_gpu_computeUnscaledCentroids;
double Tomp_gpu_transposeReps;
double Tomp_gpu_featureScaling;
double Tomp_gpu_seeding, Tomp_gpu_computeAssign, Tomp_gpu_updateCentroids, Tomp_gpu_kmeans, Tomp_gpu_kmeanspp;
double Tomp_gpu_spectralClustering;
double Tomp_gpu_constructSimLapMatrix, Tomp_gpu_constructSimMatrixInCSR;
double Tomp_gpu_filterNoise;
double Tomp_gpu_cuSolverDNsyevdx, Tomp_gpu_nvGRAPHSpectralClusteringAPI, Tomp_gpu_cuGraphSpectralClusteringAPI;
double Tomp_gpu_autoTuneNbClusters;
double Tomp_gpu_normalizeEigenvectorMatrix;
double Tomp_gpu_finalKmeansForSC;
double Tomp_gpu_membershipAttach;
// -- on CPU-GPU
double Tomp_cpu_gpu_transfers, Tomp_gpu_cpu_transfers;