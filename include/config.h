#ifndef _CONFIG_H
#define _CONFIG_H

// A few points to note: 
// Constants related to datasets are defined in "datasets.h".
// Constants defined as DEFAULT_xxx are default values for corresponding regulable variables in "main.cc".
// For a general test, users should usually pay attention to the values of the following variables:
// #define dataset_name
// ClustAlgo, regulable by the "-algo" argument, default value: DEFAULT_CLUSTERING_ALGO
// NbThreadsCPU, regulable by the "-cpu-nt" argument, default value: DEFAULT_NB_THREADS_CPU
// NbThreadsCPU, regulable by the "-cpu-nt" argument, default value: DEFAULT_NB_THREADS_CPU


/*-----------------------------------------------------------------------------------------------*/
/* Benchmark dataset (regulable by users)                                                       */
/*-----------------------------------------------------------------------------------------------*/
#define S1    // Choose a benchmark dataset from "datasets.h"
              // - 2D shape datasets: Jain, Aggregation, S1, Spiral_Clean, Complex9
              // - Noisy datasets: Compound, Cure_t2, Cluto_t7, Cluto_t8
              // - MNIST-based datasets: MNIST, MNIST120K, MNIST240K
              // - Spirals-based datasets: Spiral_Clean, Spirals_A1E2, Spirals_A1E3, Spirals_A1E4
              // - Smile2-based datasets: Smile2, Smile2_A1E5
              // - Aggregation-based datasets: Aggregation, Aggregation_A1E5
              // - Complex9-based datasets: Complex9, Complex9_A1E5
              // - 4D synthetic datasets: Clouds4D_1E6, Clouds4D_5E6, Clouds4D_5E7
              // - other real-life datasets: US_Census_1990
#include "datasets.h"   // Include the file "datasets.h" where users can find/define the characteristics of the benchmark dataset


/*-----------------------------------------------------------------------------------------------*/
/* Default values related to global application (regulable by users)                             */
/*-----------------------------------------------------------------------------------------------*/
#define DEFAULT_CLUSTERING_ALGO       KM_CPU  // Default clustering algorithm [options: KM_CPU, KM_GPU, SC_CPU, SC_GPU, SC_REPS]
                                              // KM_CPU:  parallel k-means(++) clustering on CPU
                                              // KM_GPU:  parallel k-means(++) clustering on GPU
                                              // SC_CPU:  parallel spectral clustering on CPU (imcomplete due to the need of calling an API of scikit-learn)
                                              // SC_GPU:  parallel spectral clustering on GPU
                                              // SC_REPS: parallel representative-based spectral clustering on CPU / on GPU / on CPU+GPU
#define DEFAULT_NB_THREADS_CPU        40   // Default number of threads on CPU
#define DEFAULT_FLAG_FEATURE_SCALING  1    // Default flag [options: 0, 1] indicating min-max feature scaling on input data
                                           // 0: no feature scaling
                                           // 1: perform feature scaling


/*-----------------------------------------------------------------------------------------------*/
/* Default values related to k-means(++) clustering (regulable by users)                             */
/*-----------------------------------------------------------------------------------------------*/
// - related to both k-means(++) on CPU and k-means(++) on GPU
#define DEFAULT_SEED_BASE             1        // Default seed base for choosing initial cluster centroids
#define DEFAULT_THOLD_USE_PACKAGES    10000    // Default threshold for using packages in the Update step of k-means(++)
#define DEFAULT_NB_PACKAGES           100      // Default number of packages used in the Update step of k-means(++)
#define DEFAULT_MAX_NB_ITERS_KMEANS   200      // Default value for the maximum number of k-means(++) iterations
// - only related to k-means(++) on CPU
#define DEFAULT_SEEDING_KMEANS_CPU    1        // Default method [options: 1, 2] for choosing initial cluster centroids
                                               // 1: uniformly at random
                                               // 2: D² sampling (k-means++ initialization)
#define DEFAULT_TOL_KMEANS_CPU        1.0E-4f  // Default tolerance to stop k-means(++) iterations on CPU
// - only related to k-means(++) on GPU
#define DEFAULT_SEEDING_KMEANS_GPU    1        // Default method [options: 1, 2] for choosing initial cluster centroids
                                               // 1: sampling uniformly at random 
                                               // 2: D² sampling (k-means++ initialization)
#define DEFAULT_TOL_KMEANS_GPU        1.0E-4f  // Default tolerance to stop k-means(++) iterations on GPU
#define DEFAULT_NB_STREAMS_UPDATE_S1  16       // Default number of streams for the Update Step 1
#define DEFAULT_NB_STREAMS_UPDATE_S2  32       // Default number of streams for the Update Step 2


/*-----------------------------------------------------------------------------------------------*/
/* Default values related to spectral clustering (regulable by users)                            */
/*-----------------------------------------------------------------------------------------------*/
#define DEFAULT_SC_IMPLEMENTATION_ON_GPU  SP_NVG  // Default GPU implementation [options: DN_CUS, SP_NVG, SP_NVG_KM, SP_CUG] of spectral clustering
                                                  // DN_CUS:    spectral clustering in dense storage format involving cuSolverDN library
                                                  // SP_NVG:    spectral clustering in sparse storage format involving nvGRAPH library
                                                  // SP_NVG_KM: spectral clustering in sparse storage format involving nvGRAPH library + our k-means(++) implementation
                                                  // SP_CUG:    spectral clustering in sparse storage format involving cuGraph library
// - related to similarity matrix construction
#define DEFAULT_CSR_ALGO   1    // Default algorithm [options: 1, 2, 3] for CSR-format similarity matrix construction
                                // 1: Algo CSR-1 for similarity matrix construction (2 complete passes)
                                // 2: Algo CSR-2 for similarity matrix construction (1 complete pass + 1 supplementary pass)
                                // 3: Algo CSR-3 for similarity matrix construction (chunkwise dense-to-CSR construction)
#define GAUSS_SIM_WITH_THOLD    // Define the way of computing similarity [options: UNI_SIM_WITH_SQDIST_THOLD, GAUSS_SIM_WITH_SQDIST_THOLD, GAUSS_SIM_WITH_THOLD, COS_SIM_WITH_THOLD]
                                // UNI_SIM_WITH_SQDIST_THOLD: uniform similarity with threshold for squared distance
                                // GAUSS_SIM_WITH_SQDIST_THOLD: Gaussian similarity with threshold for squared distance
                                // GAUSS_SIM_WITH_THOLD: Gaussian similarity with threshold for similarity
                                // COS_SIM_WITH_THOLD: cosine similarity with threshold for similarity
#define DEFAULT_SIGMA                 0.01f    // Default value for sigma of Gaussian similarity metric
#define DEFAULT_THOLD_SIM             0.01f    // Default threshold for similarity
#define DEFAULT_THOLD_DIST_SQ         0.01f    // Default threshold for squared distance
#define DEFAULT_HYPO_MAX_NNZ_ROW      100      // For Algo CSR-2: default hypothetical maximum number of nonzeros in one row of similarity matrix
#define DEFAULT_PAD1                  0        // For Algo CSR-2: default value for pad1 in the kernels
#define DEFAULT_PAD2                  0        // For Algo CSR-2: default value for pad2 in the kernels
#define DEFAULT_PAD3                  0        // For Algo CSR-2: default value for pad3 in the kernels
#define DEFAULT_MEM_USE_PERCENT       80.0f    // For Algo CSR-3: default percentage of free memory usage
#define DEFAULT_MAX_NZ_PERCENT        5.0f     // For Algo CSR-3: default maximal percentage of nonzeros in the similarity matrix
#define DEFAULT_SIM_CONSTR_ALGO_CPU   3        // Default algorithm [options: 1, 2, 3] for similarity matrix construction on CPU
                                               // 1: construct dense similarity matrix on CPU
                                               // 2: construct CSR-format similarity matrix on CPU (mono-thread version)
                                               // 3: construct CSR-format similarity matrix on CPU (multi-thread version)
// - related to nvGRAPH & cuGraph libraries
#define DEFAULT_NVGRAPH_SC_ALGO       3        // Default algorithm [options: 1, 2, 3] of nvGRAPH's spectral graph partitioning
                                               // 1: NVGRAPH_MODULARITY_MAXIMIZATION
                                               // 2: NVGRAPH_BALANCED_CUT_LANCZOS
                                               // 3: NVGRAPH_BALANCED_CUT_LOBPCG
#define DEFAULT_CUGRAPH_SC_ALGO       1        // Default algorithm [options: 1, 2] of cuGraph's spectral graph partitioning
                                               // 1: cuGraph Modularity Maximization
                                               // 2: cuGraph Balanced Cut using Lanczos eigensolver
#define DEFAULT_MAX_NB_ITERS_EIGEN    4000     // Default value (suggested by nvGRAPH) for the maximum number of eigensolver iterations
#define DEFAULT_TOL_EIGEN             1.0E-4f  // Default value for eigensolver tolerance (nvGRAPH doc: "Usually values between 0.01 and 0.0001 are acceptable for spectral clustering")
// - related to noise filtering
#define DEFAULT_FILTER_NOISE_APPROACH 0        // Default approach [options: 0, 1, 2] to filtering noise for spectral clustering
                                               // 0: no noise filtering
                                               // 1: noise filtering based on nnz per row of the similarity matrix
                                               // 2: noise filtering based on vertex degree of the similarity graph
#define DEFAULT_NB_BINS_HIST          20       // Default number of bins in the histogram
#define DEFAULT_THOLD_NOISE           0.0f     // Default threshold for filtering noise
// - related to auto-tuning of the number of clusters
#define DEFAULT_FLAG_AUTO_TUNE_NB_CLUSTERS  0  // Default flag [options: 0, 1] indicating the auto-tuning of the number of clusters
                                               // 0: no auto-tuning of the number of clusters
                                               // 1: enable the auto-tuning of the number of clusters based on eigengaps
// - related to both noise filtering and auto-tuning
#define DEFAULT_FLAG_INTERACTIVE      1    // Default flag [options: 0, 1] indicating interactive mode for noise filtering & auto-tuning
                                           // 0: non-interactive mode
                                           // 1: interactive mode


/*-----------------------------------------------------------------------------------------------*/
/* Default values related to representative-based spectral clustering (regulable by users)       */
/*-----------------------------------------------------------------------------------------------*/
#define DEFAULT_METHOD_TO_EXTRACT_REPS  ER_RS  // Default method [options: ER_RS, ER_KM, ER_KMPP] for extracting representatives
                                               // ER_RS:   extract representatives using random sampling
                                               // ER_KM:   extract representatives using k-means algorithm
                                               // ER_KMPP: extract representatives using k-means++ algorithm
#define DEFAULT_CHAIN_OF_SC_USING_REPS  CHAIN_ON_GPU  // Default processing chain [options: CHAIN_ON_GPU, CHAIN_ON_CPU, CHAIN_ON_CPU_GPU] for representative-based spectral clustering
                                                      // CHAIN_ON_GPU: representative-based spectral clustering on GPU
                                                      // CHAIN_ON_CPU: representative-based spectral clustering on CPU (imcomplete due to the need of calling an API of scikit-learn)
                                                      // CHAIN_ON_CPU_GPU: representative-based spectral clustering on CPU+GPU
// #define  RS


/*-----------------------------------------------------------------------------------------------*/
/* Default values related to block size configuration for CUDA kernels (regulable by users)      */
/*-----------------------------------------------------------------------------------------------*/
#define DEFAULT_BLOCK_SIZE_X_N        128      // BLOCK_SIZE_X related to NB_POINTS or NB_REPS (BsXN has to be in [1, 1024] & be a power of 2)
#define DEFAULT_BLOCK_SIZE_X_P        512      // BLOCK_SIZE_X related to NB_POINTS devided by nbPackages (BsXP has to be in [1, 1024] & be a power of 2)
#define DEFAULT_BLOCK_SIZE_X_D        32       // BLOCK_SIZE_X related to NB_DIMS (BsXD has to be in [1, 1024] & be a power of 2)
#define DEFAULT_BLOCK_SIZE_X_C        32       // BLOCK_SIZE_X related to NB_CLUSTERS (BsXC has to be in [1, 1024] & be a power of 2)
#define DEFAULT_BLOCK_SIZE_Y_N        2        // BLOCK_SIZE_Y related to NB_POINTS or NB_REPS (BsYN has to be in [1, 1024] & BsXN*BsYN has to be at most 1024)
#define BSYD  2                                // BLOCK_SIZE_Y related to NB_DIMS (BSYD has to be in [1, 1024] & BsXP*BSYD has to be in [NB_CLUSTERS, 1024] in case of k-means & BsXN*BSYD has to be at most 1024 in case of feature scaling)


/*-----------------------------------------------------------------------------------------------*/
/* Constants related to hardware (regulable by users)                                            */
/*-----------------------------------------------------------------------------------------------*/
// Limits of hardware/software
#define TOTAL_SHMEM_BLOCK  49152   // Total amount of GPU shared memory per block: 49152 bytes = 48 KB
#define SAFETY_THOLD       128     // Safety threshold for TOTAL_SHMEM_BLOCK: 128 bytes


/*-----------------------------------------------------------------------------------------------*/
/* Constants related to software (regulable by users)                                            */
/*-----------------------------------------------------------------------------------------------*/
#define STACK_SIZE_LIMIT   8388608 // Limited stack size per CPU thread: 8388608 bytes = 8 MB (only used in "kmeans\kmeans_cpu.cc")
#define SAFETY_DIVISOR     5       // Safety divisor for STACK_SIZE_LIMIT (only used in "kmeans\kmeans_cpu.cc")
#define NB_DIMS_LIMITED_BY_NB_CLUSTERS  (STACK_SIZE_LIMIT/sizeof(T_real)/NB_CLUSTERS)/SAFETY_DIVISOR  // Due to the limited stack size, we have to limit the nb of processed dimensions at a time for some arrays in "kmeans/kmeans_cpu.cc".
#define NB_DIMS_LIMITED_BY_NB_REPS      (STACK_SIZE_LIMIT/sizeof(T_real)/NB_REPS    )/SAFETY_DIVISOR  // Due to the limited stack size, we have to limit the nb of processed dimensions at a time for some arrays in "kmeans/kmeans_cpu.cc".
#define NB_DIMS_BATCH_NC                (NB_DIMS_LIMITED_BY_NB_CLUSTERS > NB_DIMS ? NB_DIMS : NB_DIMS_LIMITED_BY_NB_CLUSTERS)
#define NB_DIMS_BATCH_NR                (NB_DIMS_LIMITED_BY_NB_REPS     > NB_DIMS ? NB_DIMS : NB_DIMS_LIMITED_BY_NB_REPS)
#define MAX_LINE_LENGTH    10000   // Maximum line length (only used in "dataIO.cc")


/*-----------------------------------------------------------------------------------------------*/
/* Enumerated type (stable part)                                                                 */
/*-----------------------------------------------------------------------------------------------*/
typedef enum _algo_t {
    KM_CPU = 1,  // 1: parallel k-means(++) clustering on CPU
    KM_GPU,      // 2: parallel k-means(++) clustering on GPU
    SC_CPU,      // 3: parallel spectral clustering on CPU (imcomplete due to the need of calling an API of scikit-learn)
    SC_GPU,      // 4: parallel spectral clustering on GPU
    SC_REPS,     // 5: parallel representative-based spectral clustering on CPU / on GPU / on CPU+GPU
    NB_OF_CLUSTERING_ALGOS
} algo_t;

typedef enum _scimp_t {
    DN_CUS = 1,  // 1: spectral clustering in dense storage format with cuSolverDN library
    SP_NVG,      // 2: spectral clustering in sparse storage format with nvGRAPH library
    SP_NVG_KM,   // 3: spectral clustering in sparse storage format with nvGRAPH library + our k-means(++) implementation
    SP_CUG,      // 4: spectral clustering in sparse storage format with cuGraph library
    NB_OF_SC_IMPLEMENTATIONS_ON_GPU
} scimp_t;

typedef enum _chain_t {
    CHAIN_ON_GPU = 1,  // 1: parallel representative-based spectral clustering on GPU
    CHAIN_ON_CPU,      // 2: parallel representative-based spectral clustering on CPU (imcomplete due to the need of calling an API of scikit-learn)
    CHAIN_ON_CPU_GPU,  // 3: parallel representative-based spectral clustering on CPU+GPU
    NB_OF_CHAINS_OF_SC_USING_REPS
} chain_t;

typedef enum _extr_t {
    ER_RS = 1,   // 1: extract representatives using random sampling
    ER_KM,       // 2: extract representatives using k-means algorithm
    ER_KMPP,     // 3: extract representatives using k-means++ algorithm
    NB_OF_METHODS_TO_EXTRACT_REPS
} extr_t;


/*-----------------------------------------------------------------------------------------------*/
/* Floating point datatype and operations (stable part)                                          */
/*-----------------------------------------------------------------------------------------------*/
#ifdef DP  // double precision
typedef double T_real;
#define T_REAL_TEXT                   "doubles"
#define T_REAL_PRINT                  "%lf"
#define T_REAL                        CUDA_R_64F
#define EXP                           exp
#define SQRT                          sqrt
#define RSQRT                         rsqrt
#define CUBLAS_GEAM                   cublasDgeam
#define CURAND_UNIFORM                curand_uniform_double
#define CUSPARSE_GTHR                 cusparseDgthr
#define CUSOLVERDN_SYEVDX_BUFFERSIZE  cusolverDnDsyevdx_bufferSize
#define CUSOLVERDN_SYEVDX             cusolverDnDsyevdx
#else      // single precision
typedef float T_real;
#define T_REAL_TEXT                   "floats"
#define T_REAL_PRINT                  "%f"
#define T_REAL                        CUDA_R_32F
#define EXP                           __expf
#define SQRT                          sqrtf   // no __sqrtf
#define RSQRT                         rsqrtf  // no __rsqrtf
#define CUBLAS_GEAM                   cublasSgeam
#define CURAND_UNIFORM                curand_uniform
#define CUSPARSE_GTHR                 cusparseSgthr
#define CUSOLVERDN_SYEVDX_BUFFERSIZE  cusolverDnSsyevdx_bufferSize
#define CUSOLVERDN_SYEVDX             cusolverDnSsyevdx
#endif


/*-----------------------------------------------------------------------------------------------*/
/* Integer datatype (stable part)                                                                */
/*-----------------------------------------------------------------------------------------------*/
#ifdef LIN  // long integer
typedef size_t index_t;
#define T_INT_PRINT  "%zu"
#else       // standard integer
typedef int index_t;
#define T_INT_PRINT  "%d"
#endif


#endif
