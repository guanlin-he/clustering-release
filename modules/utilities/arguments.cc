#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <string.h>  // Library functions (e.g. strcat, strcmp, strcpy, strtok)

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/arguments.h"


/*------------------------------------------------------------------------------------------------*/
/* Command Line parsing.                                                                          */
/*------------------------------------------------------------------------------------------------*/
void usage (int ExitCode, FILE *std)
{
    fprintf(std,"Application arguments: \n");
    fprintf(std,"   -h                    <print this help>\n");
    
    fprintf(std,"\n");
    
    fprintf(std,"/* Related to global application: */\n");
    fprintf(std,"   -algo                 <clustering algorithm ID [1, ..., %d]> (default: %d)\n", NB_OF_CLUSTERING_ALGOS - 1, DEFAULT_CLUSTERING_ALGO);
    fprintf(std,"   -cpu-nt               <number of OpenMP threads> (default: %d)\n", DEFAULT_NB_THREADS_CPU);
    fprintf(std,"   -fs                   <feature scaling on input data [0, 1]> (default: %d)\n", DEFAULT_FLAG_FEATURE_SCALING);

    fprintf(std,"\n");

    fprintf(std,"/* Related to k-means(++) clustering: */\n");
    fprintf(std,"/** related to both k-means(++) on CPU and k-means(++) on GPU: **/\n");
    fprintf(std,"   -seedbase             <seed base for random number generator> (default: %d)\n", DEFAULT_SEED_BASE);
    fprintf(std,"   -thold-use-pkgs       <threshold for using packages in UpdateCentroids> (default: %d)\n", DEFAULT_THOLD_USE_PACKAGES);
    fprintf(std,"   -np                   <number of packages used in UpdateCentroids> (default: %d)\n", DEFAULT_NB_PACKAGES);
    fprintf(std,"   -max-iters-km         <max number of iterations for k-means(++)> (default: %d)\n", DEFAULT_MAX_NB_ITERS_KMEANS);
    fprintf(std,"/** only related to k-means(++) on CPU: **/\n");
    fprintf(std,"   -seeding-km-cpu       <seeding method for k-means(++) [1, 2]> (default: %d)\n", DEFAULT_SEEDING_KMEANS_CPU);
    fprintf(std,"   -tol-km-cpu           <tolerance, i.e. convergence criterion, for k-means(++) on CPU> (default: %f)\n", DEFAULT_TOL_KMEANS_CPU);
    fprintf(std,"/** only related to k-means(++) on GPU: **/\n");
    fprintf(std,"   -seeding-km-gpu       <seeding method for k-means(++) [1, 2]> (default: %d)\n", DEFAULT_SEEDING_KMEANS_GPU);
    fprintf(std,"   -tol-km-gpu           <tolerance, i.e. convergence criterion, for k-means(++) on GPU> (default: %f)\n", DEFAULT_TOL_KMEANS_GPU);
    // fprintf(std,"   -ns1                  <number of streams for Step 1 of UpdateCentroids> (default: %d)\n", DEFAULT_NB_STREAMS_UPDATE_S1);
    // fprintf(std,"   -ns2                  <number of streams for Step 2 of UpdateCentroids> (default: %d)\n", DEFAULT_NB_STREAMS_UPDATE_S2);

    fprintf(std,"\n");
    
    fprintf(std,"/* Related to spectral clustering: */\n");
    fprintf(std,"   -sc-imp               <GPU implementation of spectral clustering [1, ..., %d]> (default: %d)\n", NB_OF_SC_IMPLEMENTATIONS_ON_GPU - 1, DEFAULT_SC_IMPLEMENTATION_ON_GPU);
    fprintf(std,"/** related to similarity matrix construction: **/\n");
    fprintf(std,"   -csr-algo             <algorithm for constructing similarity matrix in CSR format [1, 2, 3]> (default: %d)\n", DEFAULT_CSR_ALGO);
    fprintf(std,"   -sigma                <Sigma in Gaussian similarity function> (default: %f)\n", DEFAULT_SIGMA);
    fprintf(std,"   -thold-sim            <threshold for similarity> (default: %f)\n", DEFAULT_THOLD_SIM);
    fprintf(std,"   -thold-dist-sq        <threshold for squared distance> (default: %f)\n", DEFAULT_THOLD_DIST_SQ);
    fprintf(std,"   -hypo-max-nnz-row     <hypothetical maximal nnz in one row> (default: %d)\n", DEFAULT_HYPO_MAX_NNZ_ROW);
    // fprintf(std,"   -pad1                 <padding 1 in shared memory> (default: %d)\n", DEFAULT_PAD1);
    // fprintf(std,"   -pad2                 <padding 2 in shared memory> (default: %d)\n", DEFAULT_PAD2);
    // fprintf(std,"   -pad3                 <padding 3 in shared memory> (default: %d)\n", DEFAULT_PAD3);
    fprintf(std,"   -mem-use-percent      <percentage of free memory usage> (default: %f)\n", DEFAULT_MEM_USE_PERCENT);
    fprintf(std,"   -max-nz-percent       <maximal percentage of nonzeros in the similarity matrix> (default: %f)\n", DEFAULT_MAX_NZ_PERCENT);
    fprintf(std,"   -sim-constr-algo-cpu  <algorithm for constructing similarity matrix on CPU [1, 2, 3]> (default: %d)\n", DEFAULT_SIM_CONSTR_ALGO_CPU);
    fprintf(std,"/** related to nvGRAPH & cuGraph libraries: **/\n");
    fprintf(std,"   -nvg-algo             <algorithm of nvGRAPH Spectral Clustering API [1, 2, 3]> (default: %d)\n", DEFAULT_NVGRAPH_SC_ALGO);
    fprintf(std,"   -cug-algo             <algorithm of cuGraph Spectral Clustering API [1, 2]> (default: %d)\n", DEFAULT_CUGRAPH_SC_ALGO);
    fprintf(std,"   -max-iters-eig        <max number of iterations for eigensolver> (default: %d)\n", DEFAULT_MAX_NB_ITERS_EIGEN);
    fprintf(std,"   -tol-eig              <tolerance, i.e. convergence criterion, for eigensolver> (default: %f)\n", DEFAULT_TOL_EIGEN);
    fprintf(std,"/** related to noise filtering: **/\n");
    fprintf(std,"   -filter-noise         <approach to filtering noise based on sparse similarity matrix [0, 1, 2]> (default: %d)\n", DEFAULT_FILTER_NOISE_APPROACH);
    fprintf(std,"   -nb-bins-hist         <number of bins for the histogram of noise filtering [1, ...]> (default: %d)\n", DEFAULT_NB_BINS_HIST);
    fprintf(std,"   -thold-noise          <threshold for noise ratio (0.0, 1.0)> (default: %f)\n", DEFAULT_THOLD_NOISE);
    fprintf(std,"/** related to auto-tuning of the number of clusters: **/\n");
    fprintf(std,"   -auto-tune-nc         <flag for the auto-tuning of the number of clusters (0 or 1)> (default: %d)\n", DEFAULT_FLAG_AUTO_TUNE_NB_CLUSTERS);
    fprintf(std,"/** related to both noise filtering and auto-tuning: **/\n");
    fprintf(std,"   -interactive          <flag indicating interactive mode for noise filtering and auto-tuning (0 or 1)> (default: %d)\n", DEFAULT_FLAG_INTERACTIVE);

    fprintf(std,"\n");
    
    fprintf(std,"/* Related to representative-based spectral clustering: */\n");
    fprintf(std,"   -er                   <method to extract reps [1, ..., %d]> (default: %d)\n", NB_OF_METHODS_TO_EXTRACT_REPS - 1, DEFAULT_METHOD_TO_EXTRACT_REPS);
    fprintf(std,"   -chain                <processing chain of spectral clustering using reps [1, ..., %d]> (default: %d)\n", NB_OF_CHAINS_OF_SC_USING_REPS - 1, DEFAULT_CHAIN_OF_SC_USING_REPS);
    
    // fprintf(std,"\n");
    
    // fprintf(std,"/* Related to block size configuration for CUDA kernels: */\n");
    // fprintf(std,"   -bsxn                 <BLOCK_SIZE_X related to NB_POINTS or NB_REPS [1, ..., power of 2, ..., 1024]> (default: %d)\n", DEFAULT_BLOCK_SIZE_X_N);
    // fprintf(std,"   -bsxp                 <BLOCK_SIZE_X related to NB_POINTS devided by NbPackages [1, ..., power of 2, ..., 1024]> (default: %d)\n", DEFAULT_BLOCK_SIZE_X_P);
    // fprintf(std,"   -bsxd                 <BLOCK_SIZE_X related to NB_DIMS [1, ..., power of 2, ..., 1024]> (default: %d)\n", DEFAULT_BLOCK_SIZE_X_D);
    // fprintf(std,"   -bsxc                 <BLOCK_SIZE_X related to NB_CLUSTERS [1, ..., power of 2, ..., 1024]> (default: %d)\n", DEFAULT_BLOCK_SIZE_X_C);
    // fprintf(std,"   -bsyn                 <BLOCK_SIZE_Y related to NB_POINTS or NB_REPS [1 - 1024]> (default: %d)\n", DEFAULT_BLOCK_SIZE_Y_N);
    // fprintf(std,"   -bsxk1                <BLOCK_SIZE_X related to kernel_first_pass_CSR1_2D_grid_2D_blocks> (default: %d)\n", DEFAULT_BLOCK_SIZE_X_N);
    // fprintf(std,"   -bsxk2                <BLOCK_SIZE_X related to kernel_second_pass_CSR1_1D_grid_2D_blocks> (default: %d)\n", DEFAULT_BLOCK_SIZE_X_N);
    // fprintf(std,"   -bsxk3                <BLOCK_SIZE_X related to kernel_full_pass_CSR2_1D_grid_2D_blocks> (default: %d)\n", DEFAULT_BLOCK_SIZE_X_N);
    // fprintf(std,"   -bsxk4                <BLOCK_SIZE_X related to kernel_ellpack_to_csr_CSR2_1D_grid_2D_blocks> (default: %d)\n", DEFAULT_BLOCK_SIZE_X_N);
    // fprintf(std,"   -bsxk5                <BLOCK_SIZE_X related to kernel_supplementary_pass_CSR2_1D_grid_2D_blocks> (default: %d)\n", DEFAULT_BLOCK_SIZE_X_N);
    // fprintf(std,"   -bsxk6                <BLOCK_SIZE_X related to kernel_construct_similarity_chunk> (default: %d)\n", DEFAULT_BLOCK_SIZE_X_N);
    // fprintf(std,"   -bsyk1                <BLOCK_SIZE_Y related to kernel_first_pass_CSR1_2D_grid_2D_blocks> (default: %d)\n", DEFAULT_BLOCK_SIZE_Y_N);
    // fprintf(std,"   -bsyk2                <BLOCK_SIZE_Y related to kernel_second_pass_CSR1_1D_grid_2D_blocks> (default: %d)\n", DEFAULT_BLOCK_SIZE_Y_N);
    // fprintf(std,"   -bsyk3                <BLOCK_SIZE_Y related to kernel_full_pass_CSR2_1D_grid_2D_blocks> (default: %d)\n", DEFAULT_BLOCK_SIZE_Y_N);
    // fprintf(std,"   -bsyk4                <BLOCK_SIZE_Y related to kernel_ellpack_to_csr_CSR2_1D_grid_2D_blocks> (default: %d)\n", DEFAULT_BLOCK_SIZE_Y_N);
    // fprintf(std,"   -bsyk5                <BLOCK_SIZE_Y related to kernel_supplementary_pass_CSR2_1D_grid_2D_blocks> (default: %d)\n", DEFAULT_BLOCK_SIZE_Y_N);
    // fprintf(std,"   -bsyk6                <BLOCK_SIZE_Y related to kernel_construct_similarity_chunk> (default: %d)\n", DEFAULT_BLOCK_SIZE_Y_N);
    
    exit(ExitCode);
}


void command_line_parsing (int argc, char *argv[])
{   
    // Init from the command line
    argc--; argv++;
    while (argc > 0) {
        if (strcmp(argv[0],"-algo") == 0) {
            argc--; argv++;
            if (argc > 0) {
                ClustAlgo = (algo_t) atoi(argv[0]);  // atoi: Convert string to integer
                argc--; argv++;
                if (ClustAlgo < 1 || ClustAlgo > NB_OF_CLUSTERING_ALGOS - 1) {
                    fprintf(stderr,"Error: <-algo> has to in [1 - %d]!\n", NB_OF_CLUSTERING_ALGOS - 1);
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }

        } else if (strcmp(argv[0],"-cpu-nt") == 0) {
            argc--; argv++;
            if (argc > 0) {
                NbThreadsCPU = atoi(argv[0]);
                argc--; argv++;
                if (NbThreadsCPU <= 0) {
                    fprintf(stderr,"Error: <-cpu-nt> has to be >= 1!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-sc-imp") == 0) {
            argc--; argv++;
            if (argc > 0) {
                SCImpGPU = (scimp_t) atoi(argv[0]);
                argc--; argv++;
                if (SCImpGPU < 1 || SCImpGPU > NB_OF_SC_IMPLEMENTATIONS_ON_GPU - 1) {
                    fprintf(stderr,"Error: <-sc-imp> has to in [1 - %d]!\n", NB_OF_SC_IMPLEMENTATIONS_ON_GPU - 1);
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-chain") == 0) {
            argc--; argv++;
            if (argc > 0) {
                Chain = (chain_t) atoi(argv[0]);
                argc--; argv++;
                if (Chain < 1 || Chain > NB_OF_CHAINS_OF_SC_USING_REPS - 1) {
                    fprintf(stderr,"Error: <-chain> has to in [1 - %d]!\n", NB_OF_CHAINS_OF_SC_USING_REPS - 1);
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-er") == 0) {
            argc--; argv++;
            if (argc > 0) {
                MethodToExtractReps = (extr_t) atoi(argv[0]);
                argc--; argv++;
                if (MethodToExtractReps < 1 || MethodToExtractReps > NB_OF_METHODS_TO_EXTRACT_REPS - 1) {
                    fprintf(stderr,"Error: <-er> has to in [1 - %d]!\n", NB_OF_METHODS_TO_EXTRACT_REPS - 1);
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-fs") == 0) {
            argc--; argv++;
            if (argc > 0) {
                FlagFeatureScaling = atoi(argv[0]);
                argc--; argv++;
                if (FlagFeatureScaling != 0 && FlagFeatureScaling != 1) {
                    fprintf(stderr,"Error: <-fs> has to be either 0 or 1!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-sigma") == 0) {
            argc--; argv++;
            if (argc > 0) {
                Sigma = atof(argv[0]);              // atof: Convert string to double
                argc--; argv++;
                if (Sigma <= 0.0f) {
                    fprintf(stderr,"Error: <-sigma> has to be > 0.0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-thold-sim") == 0) {
            argc--; argv++;
            if (argc > 0) {
                TholdSim = atof(argv[0]);
                argc--; argv++;
                if (TholdSim < 0.0f) {
                    fprintf(stderr,"Error: <-thold-sim> has to be >= 0.0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-thold-dist-sq") == 0) {
            argc--; argv++;
            if (argc > 0) {
                TholdDistSq = atof(argv[0]);
                argc--; argv++;
                if (TholdDistSq <= 0.0f) {
                    fprintf(stderr,"Error: <-thold-dist-sq> has to be > 0.0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-sim-constr-algo-cpu") == 0) {
            argc--; argv++;
            if (argc > 0) {
                SimConstrAlgoCPU = atoi(argv[0]);
                argc--; argv++;
                if (SimConstrAlgoCPU < 1 || SimConstrAlgoCPU > 3) {
                    fprintf(stderr,"Error: <-sim-constr-algo-cpu> has to be in [1, 3]!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-csr-algo") == 0) {
            argc--; argv++;
            if (argc > 0) {
                CSRAlgo = atoi(argv[0]);
                argc--; argv++;
                if (CSRAlgo < 1 || CSRAlgo > 3) {
                    fprintf(stderr,"Error: <-csr-algo> has to be in [1, 3]!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-hypo-max-nnz-row") == 0) {
            argc--; argv++;
            if (argc > 0) {
                HypoMaxNnzRow = atoi(argv[0]);
                argc--; argv++;
                if (HypoMaxNnzRow <= 0) {
                    fprintf(stderr,"Error: <-hypo-max-nnz-row> has to be > 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }

        } else if (strcmp(argv[0],"-mem-use-percent") == 0) {
            argc--; argv++;
            if (argc > 0) {
                MemUsePercent = atof(argv[0]);
                argc--; argv++;
                if (MemUsePercent < 0.0f || MemUsePercent > 100.0f) {
                    fprintf(stderr,"Error: <-mem-use-percent> has to be in [0.0f, 100.0f] !\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-max-nz-percent") == 0) {
            argc--; argv++;
            if (argc > 0) {
                MaxNzPercent = atof(argv[0]);
                argc--; argv++;
                if (MaxNzPercent < 0.0f || MaxNzPercent > 100.0f) {
                    fprintf(stderr,"Error: <-max-nz-percent> has to be in [0.0f, 100.0f] !\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-pad1") == 0) {
            argc--; argv++;
            if (argc > 0) {
                Pad1 = atoi(argv[0]);
                argc--; argv++;
                if (Pad1 < 0) {
                    fprintf(stderr,"Error: <-pad1> has to be >= 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-pad2") == 0) {
            argc--; argv++;
            if (argc > 0) {
                Pad2 = atoi(argv[0]);
                argc--; argv++;
                if (Pad2 < 0) {
                    fprintf(stderr,"Error: <-pad2> has to be >= 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-pad3") == 0) {
            argc--; argv++;
            if (argc > 0) {
                Pad3 = atoi(argv[0]);
                argc--; argv++;
                if (Pad3 < 0) {
                    fprintf(stderr,"Error: <-pad3> has to be >= 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
     
        } else if (strcmp(argv[0],"-interactive") == 0) {
            argc--; argv++;
            if (argc > 0) {
                FlagInteractive = atoi(argv[0]);
                argc--; argv++;
                if (FlagInteractive != 0 && FlagInteractive != 1) {
                    fprintf(stderr,"Error: <-interactive> has to be either 0 or 1!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-filter-noise") == 0) {
            argc--; argv++;
            if (argc > 0) {
                FilterNoiseApproach = atoi(argv[0]);
                argc--; argv++;
                if (FilterNoiseApproach < 0 || FilterNoiseApproach > 2) {
                    fprintf(stderr,"Error: <-filter-noise> has to be in [0, 2]!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-nb-bins-hist") == 0) {
            argc--; argv++;
            if (argc > 0) {
                NbBinsHist = atoi(argv[0]);
                argc--; argv++;
                if (NbBinsHist <= 0) {
                    fprintf(stderr,"Error: <-nb-bins-hist> has to be a positive integer!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-thold-noise") == 0) {
            argc--; argv++;
            if (argc > 0) {
                TholdNoise = atof(argv[0]);
                argc--; argv++;
                if (TholdNoise <= 0.0f || TholdNoise >= 1.0f) {
                    fprintf(stderr,"Error: <-thold-noise> has to be in (0.0, 1.0)!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-nvg-algo") == 0) {
            argc--; argv++;
            if (argc > 0) {
                NVGraphAlgo = atoi(argv[0]);
                argc--; argv++;
                if (NVGraphAlgo < 1 || NVGraphAlgo > 3) {
                    fprintf(stderr,"Error: <-nvg-algo> has to be in [1, 3]!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }

        } else if (strcmp(argv[0],"-cug-algo") == 0) {
            argc--; argv++;
            if (argc > 0) {
                CUGraphAlgo = atoi(argv[0]);
                argc--; argv++;
                if (CUGraphAlgo < 1 || CUGraphAlgo > 2) {
                    fprintf(stderr,"Error: <-cug-algo> has to be in [1, 2]!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }

        } else if (strcmp(argv[0],"-max-iters-eig") == 0) {
            argc--; argv++;
            if (argc > 0) {
                MaxNbItersEigen = atoi(argv[0]);
                argc--; argv++;
                if (MaxNbItersEigen <= 0) {
                    fprintf(stderr,"Error: <-max-iters-eig> has to be > 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-tol-eig") == 0) {
            argc--; argv++;
            if (argc > 0) {
                TolEigen = atof(argv[0]);
                argc--; argv++;
                if (TolEigen <= 0.0f) {
                    fprintf(stderr,"Error: <-tol-eig> has to be > 0.0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-auto-tune-nc") == 0) {
            argc--; argv++;
            if (argc > 0) {
                FlagAutoTuneNbClusters = atoi(argv[0]);
                argc--; argv++;
                if (FlagAutoTuneNbClusters < 0 && FlagAutoTuneNbClusters > 1) {
                    fprintf(stderr,"Error: <-auto-tune-nc> has to be an integer in [0, 1]!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-seedbase") == 0) {
            argc--; argv++;
            if (argc > 0) {
                SeedBase = atoi(argv[0]);
                argc--; argv++;
                if (SeedBase <= 0) {
                    fprintf(stderr,"Error: <-seedbase> has to be > 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }

        } else if (strcmp(argv[0],"-seeding-km-cpu") == 0) {
            argc--; argv++;
            if (argc > 0) {
                SeedingKMCPU = atoi(argv[0]);
                argc--; argv++;
                if (SeedingKMCPU < 1 || SeedingKMCPU > 2) {
                    fprintf(stderr,"Error: <-seeding-km-cpu> has to be in [1, 2]!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-seeding-km-gpu") == 0) {
            argc--; argv++;
            if (argc > 0) {
                SeedingKMGPU = atoi(argv[0]);
                argc--; argv++;
                if (SeedingKMGPU < 1 || SeedingKMGPU > 2) {
                    fprintf(stderr,"Error: <-seeding-km-gpu> has to be in [1, 2]!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-thold-use-pkgs") == 0) {
            argc--; argv++;
            if (argc > 0) {
                TholdUsePackages = atoi(argv[0]);
                argc--; argv++;
                if (TholdUsePackages <= 0 ) {
                    fprintf(stderr,"Error: <-thold-use-pkgs> has to be > 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-np") == 0) {
            argc--; argv++;
            if (argc > 0) {
                NbPackages = atoi(argv[0]);
                argc--; argv++;
                if (NbPackages < 1 || NbPackages > NB_POINTS) {
                    fprintf(stderr,"Error: <-np> has to be in [1, NB_POINTS]!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-ns1") == 0) {
            argc--; argv++;
            if (argc > 0) {
                NbStreamsStep1 = atoi(argv[0]);
                argc--; argv++;
                if (NbStreamsStep1 <= 0 ) {
                    fprintf(stderr,"Error: <-ns1> has to be > 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-ns2") == 0) {
            argc--; argv++;
            if (argc > 0) {
                NbStreamsStep2 = atoi(argv[0]);
                argc--; argv++;
                if (NbStreamsStep2 <= 0 ) {
                    fprintf(stderr,"Error: <-ns2> has to be > 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-max-iters-km") == 0) {
            argc--; argv++;
            if (argc > 0) {
                MaxNbItersKM = atoi(argv[0]);
                argc--; argv++;
                if (MaxNbItersKM <= 0 ) {
                    fprintf(stderr,"Error: <-max-iters-km> has to be > 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-tol-km-cpu") == 0) {
            argc--; argv++;
            if (argc > 0) {
                TolKMCPU = atof(argv[0]);
                argc--; argv++;
                if (TolKMCPU < 0.0f) {
                    fprintf(stderr,"Error: <-tol-km-cpu> has to be >= 0.0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-tol-km-gpu") == 0) {
            argc--; argv++;
            if (argc > 0) {
                TolKMGPU = atof(argv[0]);
                argc--; argv++;
                if (TolKMGPU < 0.0f) {
                    fprintf(stderr,"Error: <-tol-km-gpu> has to be >= 0.0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsxn") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsXN = atoi(argv[0]);
                argc--; argv++;
                if (BsXN < 1 || BsXN > 1024 || ((BsXN & (BsXN - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsxn> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsxp") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsXP = atoi(argv[0]);
                argc--; argv++;
                if (BsXP < 1 || BsXP > 1024 || ((BsXP & (BsXP - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsxp> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsxd") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsXD = atoi(argv[0]);
                argc--; argv++;
                if (BsXD < 1 || BsXD > 1024 || ((BsXD & (BsXD - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsxd> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsxc") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsXC = atoi(argv[0]);
                argc--; argv++;
                if (BsXC < 1 || BsXC > 1024 || ((BsXC & (BsXC - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsxc> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsyn") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsYN = atoi(argv[0]);
                argc--; argv++;
                if (BsYN < 1 || BsYN > 1024) {
                    fprintf(stderr,"Error: <-bsyn> has to be in [1, 1024]!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsxk1") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsXK1 = atoi(argv[0]);
                argc--; argv++;
                if (BsXK1 < 1 || BsXK1 > 1024 || ((BsXK1 & (BsXK1 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsxk1> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsxk2") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsXK2 = atoi(argv[0]);
                argc--; argv++;
                if (BsXK2 < 1 || BsXK2 > 1024 || ((BsXK2 & (BsXK2 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsxk2> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsxk3") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsXK3 = atoi(argv[0]);
                argc--; argv++;
                if (BsXK3 < 1 || BsXK3 > 1024 || ((BsXK3 & (BsXK3 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsxk3> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsxk4") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsXK4 = atoi(argv[0]);
                argc--; argv++;
                if (BsXK4 < 1 || BsXK4 > 1024 || ((BsXK4 & (BsXK4 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsxk4> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsxk5") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsXK5 = atoi(argv[0]);
                argc--; argv++;
                if (BsXK5 < 1 || BsXK5 > 1024 || ((BsXK5 & (BsXK5 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsxk5> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsxk6") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsXK6 = atoi(argv[0]);
                argc--; argv++;
                if (BsXK6 < 1 || BsXK6 > 1024 || ((BsXK6 & (BsXK6 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsxk6> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsyk1") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsYK1 = atoi(argv[0]);
                argc--; argv++;
                if (BsYK1 < 1 || BsYK1 > 1024 || ((BsYK1 & (BsYK1 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsyk1> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsyk2") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsYK2 = atoi(argv[0]);
                argc--; argv++;
                if (BsYK2 < 1 || BsYK2 > 1024 || ((BsYK2 & (BsYK2 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsyk2> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsyk3") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsYK3 = atoi(argv[0]);
                argc--; argv++;
                if (BsYK3 < 1 || BsYK3 > 1024 || ((BsYK3 & (BsYK3 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsyk3> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsyk4") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsYK4 = atoi(argv[0]);
                argc--; argv++;
                if (BsYK4 < 1 || BsYK4 > 1024 || ((BsYK4 & (BsYK4 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsyk4> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsyk5") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsYK5 = atoi(argv[0]);
                argc--; argv++;
                if (BsYK5 < 1 || BsYK5 > 1024 || ((BsYK5 & (BsYK5 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsyk5> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-bsyk6") == 0) {
            argc--; argv++;
            if (argc > 0) {
                BsYK6 = atoi(argv[0]);
                argc--; argv++;
                if (BsYK6 < 1 || BsYK6 > 1024 || ((BsYK6 & (BsYK6 - 1)) != 0)) {
                    fprintf(stderr,"Error: <-bsyk6> has to be in [1, 1024] & be a power of 2!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }
            
        } else if (strcmp(argv[0],"-h") == 0) {
            usage(EXIT_SUCCESS, stdout);
        } else {
            usage(EXIT_FAILURE, stderr);
        }
    }
}
