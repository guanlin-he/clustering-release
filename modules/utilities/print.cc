#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, fprintf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/print.h"


void print_configuration()
{   
    // k-means(++) clustering on CPU
    if (ClustAlgo == KM_CPU) {
        
        fprintf(stdout, "/*** Parallel k-means clustering on CPU ***/\n");
        
        fprintf(stdout, "- Dataset:\n");
        fprintf(stdout, "    Name:                 %s\n", DATASET_NAME);
        fprintf(stdout, "    Nb of data instances: %d\n", NB_POINTS);
        fprintf(stdout, "    Nb of dimensions:     %d\n", NB_DIMS);
        fprintf(stdout, "    Nb of clusters:       %d\n", NB_CLUSTERS);
        
        fprintf(stdout, "- Run configurations:\n");
        fprintf(stdout, "    Precision:            %s\n", T_REAL_TEXT);
        fprintf(stdout, "    Nb of OpenMP threads: %d\n", NbThreadsCPU);
        
        fprintf(stdout, "- Algorithm configurations:\n");
        char *seeding;
        if (INPUT_INITIAL_CENTROIDS == "") {
            if (SeedingKMCPU == 1) {
                seeding = (char*)"uniformly at random";
            }
            if (SeedingKMCPU == 2) {
                seeding = (char*)"D² sampling from n instances";
            }
        } else {
            seeding = (char*)"load given centroids";
        }
        fprintf(stdout, "    Seeding method                     %s\n", seeding);
        if (NB_POINTS/NB_CLUSTERS > TholdUsePackages) {
            fprintf(stdout, "    Nb of packages in UpdateCentroids: %d\n", NbPackages);
        }
        fprintf(stdout, "    Max nb of k-means(++) iterations:  %d\n", MaxNbItersKM);
        fprintf(stdout, "    Tolerance:                         %f\n", TolKMCPU);
    }


    // k-means(++) clustering on GPU
    if (ClustAlgo == KM_GPU) {
        
        fprintf(stdout, "/*** Parallel k-means clustering on GPU ***/\n");
        
        fprintf(stdout, "- Dataset:\n");
        fprintf(stdout, "    Name:                 %s\n", DATASET_NAME);
        fprintf(stdout, "    Nb of data instances: %d\n", NB_POINTS);
        fprintf(stdout, "    Nb of dimensions:     %d\n", NB_DIMS);
        fprintf(stdout, "    Nb of clusters:       %d\n", NB_CLUSTERS);
        
        fprintf(stdout, "- Run configurations:\n");
        fprintf(stdout, "    Precision:            %s\n", T_REAL_TEXT);
        
        fprintf(stdout,"- Algorithm configurations:\n");
        char *seeding;
        if (INPUT_INITIAL_CENTROIDS == "") {
            switch (SeedingKMGPU) {
                case 1: seeding = (char*)"uniformly at random"; break;
                case 2: seeding = (char*)"D² sampling from n instances"; break;
                default : seeding = NULL;
            }
        } else {
            seeding = (char*)"load given centroids";
        }
        fprintf(stdout, "    Seeding method                     %s\n", seeding);
        if (NB_POINTS/NB_CLUSTERS > TholdUsePackages) {
            fprintf(stdout, "    Nb of packages in UpdateCentroids: %d\n", NbPackages);
        }
        fprintf(stdout, "    Max nb of k-means(++) iterations:  %d\n", MaxNbItersKM);
        fprintf(stdout, "    Tolerance:                         %f\n", TolKMCPU);
    }
    
    
    // Spectral clustering on CPU
    if (ClustAlgo == SC_CPU) {
        
        fprintf(stdout, "/*** Parallel spectral clustering on CPU ***/\n");
        
        fprintf(stdout, "- Dataset:\n");
        fprintf(stdout, "    Name:                 %s\n", DATASET_NAME);
        fprintf(stdout, "    Nb of data instances: %d\n", NB_POINTS);
        fprintf(stdout, "    Nb of dimensions:     %d\n", NB_DIMS);
        fprintf(stdout, "    Nb of clusters:       %d\n", NB_CLUSTERS);
        
        fprintf(stdout, "- Run configurations:\n");
        fprintf(stdout, "    Precision:            %s\n", T_REAL_TEXT);
        
        fprintf(stdout,"- Algorithm configurations:\n");
        #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
            fprintf(stdout, "    Threshold for squared distance:  %f\n", TholdDistSq);
        #endif
        
        #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
            fprintf(stdout, "    Sigma for Gaussian similarity:   %f\n", Sigma);
            fprintf(stdout, "    Threshold for squared distance:  %f\n", TholdDistSq);
        #endif
        
        #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
            fprintf(stdout, "    Sigma for Gaussian similarity:   %f\n", Sigma);
            fprintf(stdout, "    Threshold for affinity:          %f\n", TholdSim);
        #endif
        
        #ifdef COS_SIM_WITH_THOLD // Cosine similarity with threshold
            fprintf(stdout, "    Threshold for affinity:          %f\n", TholdSim);
        #endif
    }
    
    
    // Spectral clustering on GPU
    if (ClustAlgo == SC_GPU) {
        
        fprintf(stdout, "/*** Parallel spectral clustering on GPU ***/\n");
        
        fprintf(stdout, "- Dataset:\n");
        fprintf(stdout, "    Name:                 %s\n", DATASET_NAME);
        fprintf(stdout, "    Nb of data instances: %d\n", NB_POINTS);
        fprintf(stdout, "    Nb of dimensions:     %d\n", NB_DIMS);
        fprintf(stdout, "    %s%d\n", (FlagAutoTuneNbClusters == 1 ? "Max nb of clusters:   " : "Nb of clusters:       "), (FlagAutoTuneNbClusters == 1 ? MAX_NB_CLUSTERS : NB_CLUSTERS));
        
        fprintf(stdout, "- Run configurations:\n");
        fprintf(stdout, "    Precision:            %s\n", T_REAL_TEXT);
        
        fprintf(stdout,"- Algorithm configurations:\n");
        char *csrAlgo, *algo, *seeding;
        #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
            fprintf(stdout, "    Threshold for squared distance:  %f\n", TholdDistSq);
        #endif
        
        #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
            fprintf(stdout, "    Sigma for Gaussian similarity:   %f\n", Sigma);
            fprintf(stdout, "    Threshold for squared distance:  %f\n", TholdDistSq);
        #endif
        
        #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
            fprintf(stdout, "    Sigma for Gaussian similarity:   %f\n", Sigma);
            fprintf(stdout, "    Threshold for affinity:          %f\n", TholdSim);
        #endif
        
        #ifdef COS_SIM_WITH_THOLD // Cosine similarity with threshold
            fprintf(stdout, "    Threshold for affinity:          %f\n", TholdSim);
        #endif
        
        if (SCImpGPU == SP_NVG || SCImpGPU == SP_NVG_KM || SCImpGPU == SP_CUG) {
            switch (CSRAlgo) {
                case 1 : csrAlgo = (char*)"Algo CSR-1"; break;
                case 2 : csrAlgo = (char*)"Algo CSR-2"; break;
                case 3 : csrAlgo = (char*)"Algo CSR-3"; break;
                default : csrAlgo = NULL;
            }
            fprintf(stdout, "    Similarity matrix construction:  %s\n", csrAlgo);
            if (CSRAlgo == 2) {
                fprintf(stdout, "    Hypothetical max nnz in one row: %d\n", HypoMaxNnzRow);
            }
            if (CSRAlgo == 3) {
                fprintf(stdout, "    Free GPU memory usage percent:   %f\n", MemUsePercent);
                fprintf(stdout, "    Hypothetical max nz percent:     %f\n", MaxNzPercent);
            }
        }
        
        fprintf(stdout, "    Algo used to compute eigenpairs: ");
        switch (SCImpGPU) {
            case DN_CUS:   
            {
                fprintf(stdout, "cuSOLVER_DN syevdx\n"); 
                if (INPUT_INITIAL_CENTROIDS == "") {
                    switch (SeedingKMGPU) {
                        case 1 : seeding = (char*)"uniformly at random"; break;
                        case 2 : seeding = (char*)"D² sampling from n instances"; break;
                        default : seeding = NULL;
                    }
                } else {
                    seeding = (char*)"load given centroids";
                }
                fprintf(stdout, "    Seeding method for final k-means(++):%s\n", seeding);
                if (NB_POINTS/NB_CLUSTERS > TholdUsePackages) {
                    fprintf(stdout, "    Nb of packages:                      %d\n", NbPackages);
                }
                fprintf(stdout, "    Tolerance for final k-means(++):     %f\n", TolKMGPU);
            }
            break;
                        
            case SP_NVG:   
            {   
                switch (NVGraphAlgo) {
                    case 1 : algo = (char*)"NVGRAPH_MODULARITY_MAXIMIZATION"; break;
                    case 2 : algo = (char*)"NVGRAPH_BALANCED_CUT_LANCZOS"; break;
                    case 3 : algo = (char*)"NVGRAPH_BALANCED_CUT_LOBPCG"; break;
                    default : algo = NULL;
                }
                fprintf(stdout, "%s\n", algo);
                fprintf(stdout, "    Tolerance for eigensolver:       %f\n", TolEigen);
            }
            break;
                        
            case SP_NVG_KM:   
            {
                switch (NVGraphAlgo) {
                    case 1 : algo = (char*)"NVGRAPH_MODULARITY_MAXIMIZATION"; break;
                    case 2 : algo = (char*)"NVGRAPH_BALANCED_CUT_LANCZOS"; break;
                    case 3 : algo = (char*)"NVGRAPH_BALANCED_CUT_LOBPCG"; break;
                    default : algo = NULL;
                }
                fprintf(stdout, "%s\n", algo);
                fprintf(stdout, "    Tolerance for eigensolver:       %f\n", TolEigen);
                if (INPUT_INITIAL_CENTROIDS == "") {
                    switch (SeedingKMGPU) {
                        case 1 : seeding = (char*)"uniformly at random"; break;
                        case 2 : seeding = (char*)"D² sampling from n instances"; break;
                        default : seeding = NULL;
                    }
                } else {
                    seeding = (char*)"load given centroids";
                }
                fprintf(stdout, "    Seeding method for final k-means(++):%s\n", seeding);
                if (NB_POINTS/NB_CLUSTERS > TholdUsePackages) {
                    fprintf(stdout, "    Nb of packages:                      %d\n", NbPackages);
                }
                fprintf(stdout, "    Tolerance for final k-means(++):     %f\n", TolKMGPU);
            }
            break;
            
            case SP_CUG:
            {   
                switch (CUGraphAlgo) {
                    case 1 : algo = (char*)"cuGraph Modularity Maximization"; break;
                    case 2 : algo = (char*)"cuGraph Balanced Cut (using Lanczos eigensolver)"; break;
                    default : algo = NULL;
                }
                fprintf(stdout, "%s\n", algo);
                fprintf(stdout, "    Tolerance for eigensolver:       %f\n", TolEigen);
            } 
            break;
            
            default :
            {
                fprintf(stderr, "    Unknown GPU implementation of spectral clustering!\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    
    
    // Spectral clustering using reps on GPU
    if (ClustAlgo == SC_REPS && Chain == CHAIN_ON_GPU) {
        
        fprintf(stdout, "/*** Parallel representative-based spectral clustering on GPU ***/\n");
        
        fprintf(stdout, "- Dataset:\n");
        fprintf(stdout, "    Name:                 %s\n", DATASET_NAME);
        fprintf(stdout, "    Nb of data instances: %d\n", NB_POINTS);
        fprintf(stdout, "    Nb of dimensions:     %d\n", NB_DIMS);
        fprintf(stdout, "    %s%d\n", (FlagAutoTuneNbClusters == 1 ? "Max nb of clusters:   " : "Nb of clusters:       "), (FlagAutoTuneNbClusters == 1 ? MAX_NB_CLUSTERS : NB_CLUSTERS));
        
        fprintf(stdout, "- Run configurations:\n");
        fprintf(stdout, "    Precision:            %s\n", T_REAL_TEXT);
        
        fprintf(stdout, "- Algorithm configurations:\n");
        fprintf(stdout, "    Nb of representatives:           %d\n", NB_REPS);
        char *csrAlgo, *algo, *seeding;
        if (MethodToExtractReps == ER_KM || MethodToExtractReps == ER_KMPP) {
            if (INPUT_INITIAL_CENTROIDS == "") {
                if (MethodToExtractReps == ER_KM) {
                    seeding = (char*)"uniformly at random";
                }
                if (MethodToExtractReps == ER_KMPP) {
                    seeding = (char*)"D² sampling from n instances";
                }
            } else {
                seeding = (char*)"load given centroids";
            }
            fprintf(stdout, "    Seeding method:                    %s\n", seeding);
            if (NB_POINTS/NB_REPS > TholdUsePackages) {
                fprintf(stdout, "    Nb of packages in UpdateCentroids: %d\n", NbPackages);
            }
            fprintf(stdout, "    Max nb of k-means(++) iterations:  %d\n", MaxNbItersKM);
            fprintf(stdout, "    Tolerance:                         %f\n", TolKMGPU);
        }
        
        #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
            fprintf(stdout, "    Threshold for squared distance:  %f\n", TholdDistSq);
        #endif
        
        #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
            fprintf(stdout, "    Sigma for Gaussian similarity:   %f\n", Sigma);
            fprintf(stdout, "    Threshold for squared distance:  %f\n", TholdDistSq);
        #endif
        
        #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
            fprintf(stdout, "    Sigma for Gaussian similarity:   %f\n", Sigma);
            fprintf(stdout, "    Threshold for affinity:          %f\n", TholdSim);
        #endif
        
        #ifdef COS_SIM_WITH_THOLD // Cosine similarity with threshold
            fprintf(stdout, "    Threshold for affinity:          %f\n", TholdSim);
        #endif
        
        if (SCImpGPU == SP_NVG || SCImpGPU == SP_NVG_KM || SCImpGPU == SP_CUG) {
            switch (CSRAlgo) {
                case 1 : csrAlgo = (char*)"Algo CSR-1"; break;
                case 2 : csrAlgo = (char*)"Algo CSR-2"; break;
                case 3 : csrAlgo = (char*)"Algo CSR-3"; break;
                default : csrAlgo = NULL;
            }
            fprintf(stdout, "    Similarity matrix construction:  %s\n", csrAlgo);
            if (CSRAlgo == 2) {
                fprintf(stdout, "    Hypothetical max nnz in one row: %d\n", HypoMaxNnzRow);
            }
            if (CSRAlgo == 3) {
                fprintf(stdout, "    Free GPU memory usage percent:   %f\n", MemUsePercent);
                fprintf(stdout, "    Hypothetical max nz percent:     %f\n", MaxNzPercent);
            }
        }
        
        fprintf(stdout, "    Algo used to compute eigenpairs: ");
        switch (SCImpGPU) {
            case DN_CUS:   
            {
                fprintf(stdout, "cuSOLVER_DN syevdx\n"); 
                switch (SeedingKMGPU) {
                    case 1 : seeding = (char*)"uniformly at random"; break;
                    case 2 : seeding = (char*)"D² sampling from n instances"; break;
                    default : seeding = NULL;
                }
                fprintf(stdout, "    Seeding method for final k-means(++):%s\n", seeding);
                if (NB_REPS/NB_CLUSTERS > TholdUsePackages) {
                    fprintf(stdout, "    Nb of packages:                      %d\n", NbPackages);
                }
                fprintf(stdout, "    Tolerance for final k-means(++):     %f\n", TolKMGPU);
            }
            break;
                        
            case SP_NVG:   
            {   
                switch (NVGraphAlgo) {
                    case 1 : algo = (char*)"NVGRAPH_MODULARITY_MAXIMIZATION"; break;
                    case 2 : algo = (char*)"NVGRAPH_BALANCED_CUT_LANCZOS"; break;
                    case 3 : algo = (char*)"NVGRAPH_BALANCED_CUT_LOBPCG"; break;
                    default : algo = NULL;
                }
                fprintf(stdout, "%s\n", algo);
                fprintf(stdout, "    Tolerance for eigensolver:       %f\n", TolEigen);
            } 
            break;
                        
            case SP_NVG_KM:   
            {
                switch (NVGraphAlgo) {
                    case 1 : algo = (char*)"NVGRAPH_MODULARITY_MAXIMIZATION"; break;
                    case 2 : algo = (char*)"NVGRAPH_BALANCED_CUT_LANCZOS"; break;
                    case 3 : algo = (char*)"NVGRAPH_BALANCED_CUT_LOBPCG"; break;
                    default : algo = NULL;
                }
                fprintf(stdout, "%s\n", algo);
                fprintf(stdout, "    Tolerance for eigensolver:       %f\n", TolEigen);
                switch (SeedingKMGPU) {
                    case 1 : seeding = (char*)"uniformly at random"; break;
                    case 2 : seeding = (char*)"D² sampling from n instances"; break;
                    default : seeding = NULL;
                }
                fprintf(stdout, "    Seeding method for final k-means(++):%s\n", seeding);
                if (NB_REPS/NB_CLUSTERS > TholdUsePackages) {
                    fprintf(stdout, "    Nb of packages:                      %d\n", NbPackages);
                }
                fprintf(stdout, "    Tolerance for final k-means(++):     %f\n", TolKMGPU);
            }
            break;
            
            case SP_CUG:
            {   
                switch (CUGraphAlgo) {
                    case 1 : algo = (char*)"cuGraph Modularity Maximization"; break;
                    case 2 : algo = (char*)"cuGraph Balanced Cut (using Lanczos eigensolver)"; break;
                    default : algo = NULL;
                }
                fprintf(stdout, "%s\n", algo);
                fprintf(stdout, "    Tolerance for eigensolver:       %f\n", TolEigen);
            } 
            break;
                        
            default :
            {
                fprintf(stderr,"    Unknown GPU implementation of spectral clustering!\n");
                exit(EXIT_FAILURE);
            }
        }
    }


    // Spectral clustering using reps on CPU
    if (ClustAlgo == SC_REPS && Chain == CHAIN_ON_CPU) {
        
        fprintf(stdout, "/*** Parallel representative-based spectral clustering on CPU ***/\n");
        
        fprintf(stdout, "- Dataset:\n");
        fprintf(stdout, "    Name:                 %s\n", DATASET_NAME);
        fprintf(stdout, "    Nb of data instances: %d\n", NB_POINTS);
        fprintf(stdout, "    Nb of dimensions:     %d\n", NB_DIMS);
        fprintf(stdout, "    Nb of clusters:       %d\n", NB_CLUSTERS);
        
        fprintf(stdout, "- Run configurations:\n");
        fprintf(stdout, "    Precision:            %s\n", T_REAL_TEXT);
        fprintf(stdout, "    Nb of OpenMP threads: %d\n", NbThreadsCPU);
        
        fprintf(stdout, "- Algorithm configurations:\n");
        fprintf(stdout, "    Nb of representatives:           %d\n", NB_REPS);
        
        char *seeding;
        if (MethodToExtractReps == ER_KM || MethodToExtractReps == ER_KMPP) {
            if (INPUT_INITIAL_CENTROIDS == "") {
                if (MethodToExtractReps == ER_KM) {
                    seeding = (char*)"uniformly at random";
                }
                if (MethodToExtractReps == ER_KMPP) {
                    seeding = (char*)"D² sampling from n instances";
                }
            } else {
                seeding = (char*)"load given centroids";
            }
            fprintf(stdout, "    Seeding method:                    %s\n", seeding);
            if (NB_POINTS/NB_REPS > TholdUsePackages) {
                fprintf(stdout, "    Nb of packages in UpdateCentroids: %d\n", NbPackages);
            }
            fprintf(stdout, "    Max nb of k-means(++) iterations:  %d\n", MaxNbItersKM);
            fprintf(stdout, "    Tolerance:                         %f\n", TolKMCPU);
        }
    }


    // Spectral clustering using reps on CPU+GPU
    if (ClustAlgo == SC_REPS && Chain == CHAIN_ON_CPU_GPU) {
        
        fprintf(stdout, "/*** Parallel representative-based spectral clustering on CPU+GPU ***/\n");
        
        fprintf(stdout, "- Dataset:\n");
        fprintf(stdout, "    Name:                 %s\n", DATASET_NAME);
        fprintf(stdout, "    Nb of data instances: %d\n", NB_POINTS);
        fprintf(stdout, "    Nb of dimensions:     %d\n", NB_DIMS);
        fprintf(stdout, "    %s%d\n", (FlagAutoTuneNbClusters == 1 ? "Max nb of clusters:   " : "Nb of clusters:       "), (FlagAutoTuneNbClusters == 1 ? MAX_NB_CLUSTERS : NB_CLUSTERS));
        
        fprintf(stdout, "- Run configurations:\n");
        fprintf(stdout, "    Precision:            %s\n", T_REAL_TEXT);
        fprintf(stdout, "    Nb of OpenMP threads: %d\n", NbThreadsCPU);
        
        fprintf(stdout, "- Algorithm configurations:\n");
        fprintf(stdout, "    Nb of representatives:           %d\n", NB_REPS);
        char *csrAlgo, *algo, *seeding;
        if (MethodToExtractReps == ER_KM || MethodToExtractReps == ER_KMPP) {
            if (INPUT_INITIAL_CENTROIDS == "") {
                if (MethodToExtractReps == ER_KM) {
                    seeding = (char*)"uniformly at random";
                }
                if (MethodToExtractReps == ER_KMPP) {
                    seeding = (char*)"D² sampling from n instances";
                }
            } else {
                seeding = (char*)"load given centroids";
            }
            fprintf(stdout, "    Seeding method:                    %s\n", seeding);
            if (NB_POINTS/NB_REPS > TholdUsePackages) {
                fprintf(stdout, "    Nb of packages in UpdateCentroids: %d\n", NbPackages);
            }
            fprintf(stdout, "    Max nb of k-means(++) iterations:  %d\n", MaxNbItersKM);
            fprintf(stdout, "    Tolerance:                         %f\n", TolKMCPU);
        }
        
        #ifdef UNI_SIM_WITH_SQDIST_THOLD   // uniform similarity with threshold for squared distance
            fprintf(stdout, "    Threshold for squared distance:  %f\n", TholdDistSq);
        #endif
        
        #ifdef GAUSS_SIM_WITH_SQDIST_THOLD  // Gaussian similarity with threshold for squared distance
            fprintf(stdout, "    Sigma for Gaussian similarity:   %f\n", Sigma);
            fprintf(stdout, "    Threshold for squared distance:  %f\n", TholdDistSq);
        #endif
        
        #ifdef GAUSS_SIM_WITH_THOLD  // Gaussian similarity with threshold
            fprintf(stdout, "    Sigma for Gaussian similarity:   %f\n", Sigma);
            fprintf(stdout, "    Threshold for affinity:          %f\n", TholdSim);
        #endif
        
        #ifdef COS_SIM_WITH_THOLD // Cosine similarity with threshold
            fprintf(stdout, "    Threshold for affinity:          %f\n", TholdSim);
        #endif
        
        if (SCImpGPU == SP_NVG || SCImpGPU == SP_NVG_KM || SCImpGPU == SP_CUG) {
            switch (CSRAlgo) {
                case 1 : csrAlgo = (char*)"Algo CSR-1"; break;
                case 2 : csrAlgo = (char*)"Algo CSR-2"; break;
                case 3 : csrAlgo = (char*)"Algo CSR-3"; break;
                default : csrAlgo = NULL;
            }
            fprintf(stdout, "    Similarity matrix construction:  %s\n", csrAlgo);
            if (CSRAlgo == 2) {
                fprintf(stdout, "    Hypothetical max nnz in one row: %d\n", HypoMaxNnzRow);
            }
            if (CSRAlgo == 3) {
                fprintf(stdout, "    Free GPU memory usage percent:   %f\n", MemUsePercent);
                fprintf(stdout, "    Hypothetical max nz percent:     %f\n", MaxNzPercent);
            }
        }
        
        fprintf(stdout, "    Algo used to compute eigenpairs: ");
        switch (SCImpGPU) {
            case DN_CUS:   
            {
                fprintf(stdout, "cuSOLVER_DN syevdx\n"); 
                switch (SeedingKMGPU) {
                    case 1 : seeding = (char*)"uniformly at random"; break;
                    case 2 : seeding = (char*)"D² sampling from n instances"; break;
                    default : seeding = NULL;
                }
                fprintf(stdout, "    Seeding method for final k-means(++):%s\n", seeding);
                if (NB_REPS/NB_CLUSTERS > TholdUsePackages) {
                    fprintf(stdout, "    Nb of packages:                      %d\n", NbPackages);
                }
                fprintf(stdout, "    Tolerance for final k-means(++):     %f\n", TolKMGPU);
            }
            break;
                        
            case SP_NVG:   
            {   
                switch (NVGraphAlgo) {
                    case 1 : algo = (char*)"NVGRAPH_MODULARITY_MAXIMIZATION"; break;
                    case 2 : algo = (char*)"NVGRAPH_BALANCED_CUT_LANCZOS"; break;
                    case 3 : algo = (char*)"NVGRAPH_BALANCED_CUT_LOBPCG"; break;
                    default : algo = NULL;
                }
                fprintf(stdout, "%s\n", algo);
                fprintf(stdout, "    Tolerance for eigensolver:       %f\n", TolEigen);
            } 
            break;
                        
            case SP_NVG_KM:   
            {
                switch (NVGraphAlgo) {
                    case 1 : algo = (char*)"NVGRAPH_MODULARITY_MAXIMIZATION"; break;
                    case 2 : algo = (char*)"NVGRAPH_BALANCED_CUT_LANCZOS"; break;
                    case 3 : algo = (char*)"NVGRAPH_BALANCED_CUT_LOBPCG"; break;
                    default : algo = NULL;
                }
                fprintf(stdout, "%s\n", algo);
                fprintf(stdout, "    Tolerance for eigensolver:       %f\n", TolEigen);
                switch (SeedingKMGPU) {
                    case 1 : seeding = (char*)"uniformly at random"; break;
                    case 2 : seeding = (char*)"D² sampling from n instances"; break;
                    default : seeding = NULL;
                }
                fprintf(stdout, "    Seeding method for final k-means(++):%s\n", seeding);
                if (NB_REPS/NB_CLUSTERS > TholdUsePackages) {
                    fprintf(stdout, "    Nb of packages:                      %d\n", NbPackages);
                }
                fprintf(stdout, "    Tolerance for final k-means(++):     %f\n", TolKMGPU);
            }
            break;
            
            case SP_CUG:
            {   
                switch (CUGraphAlgo) {
                    case 1 : algo = (char*)"cuGraph Modularity Maximization"; break;
                    case 2 : algo = (char*)"cuGraph Balanced Cut (using Lanczos eigensolver)"; break;
                    default : algo = NULL;
                }
                fprintf(stdout, "%s\n", algo);
                fprintf(stdout, "    Tolerance for eigensolver:       %f\n", TolEigen);
            } 
            break;
                        
            default :
            {
                fprintf(stderr,"    Unknown GPU implementation of spectral clustering!\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    
    fflush(stdout);    
}



void print_results_performance()
{
    // k-means(++) clustering on CPU
    if (ClustAlgo == KM_CPU) {
        fprintf(stdout, "- Elapsed time:\n");
        fprintf(stdout, "  - On CPU:\n");
        fprintf(stdout, "    Input data:     %.2f s\n", (float)(Tomp_cpu_readData));
        fprintf(stdout, "    Output results: %.2f s\n", (float)(Tomp_cpu_saveResults));
        if (FlagFeatureScaling) {
            fprintf(stdout, "    Feature scaling:      %.2f s\n", (float)(Tomp_cpu_featureScaling));
        }
        fprintf(stdout, "    k-means(++) clustering: %.2f s\n", (float)(Tomp_cpu_kmeans));
        fprintf(stdout, "        Initialize centroids:     %.2f ms\n", (float)(Tomp_cpu_seeding*1E3f));
        fprintf(stdout, "        ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_cpu_computeAssign*1E3f)/NbItersKMCPU);
        fprintf(stdout, "        UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_cpu_updateCentroids*1E3f)/NbItersKMCPU);
        fprintf(stdout, "        Nb of iterations:         %d\n", NbItersKMCPU);
        fprintf(stdout, "  - Total appli.: %.2f s\n", (float)(Tomp_application));
    }
    
    
    // k-means(++) clustering on GPU
    if (ClustAlgo == KM_GPU) {
        fprintf(stdout, "- Elapsed time:\n");
        fprintf(stdout, "  - On CPU:\n");
        fprintf(stdout, "    Input data:     %.2f s\n", (float)(Tomp_cpu_readData));
        fprintf(stdout, "    Output results: %.2f s\n", (float)(Tomp_cpu_saveResults));
        fprintf(stdout, "  - On GPU:\n");
        fprintf(stdout, "    CUDA init:      %.2f s\n", (float)(Tomp_gpu_cuInit));
        fprintf(stdout, "    Data transfers: %f s\n", (float)(Tomp_cpu_gpu_transfers + Tomp_gpu_cpu_transfers));
        fprintf(stdout, "        CPU->GPU:   %.2f ms\n", (float)(Tomp_cpu_gpu_transfers*1E3f));
        fprintf(stdout, "        CPU<-GPU:   %.2f ms\n", (float)(Tomp_gpu_cpu_transfers*1E3f));
        fprintf(stdout, "    Feature scaling:     %.2f s\n", (float)(Tomp_gpu_featureScaling));
        fprintf(stdout, "    k-means(++) clustering:  %.2f s\n", (float)(Tomp_gpu_kmeans));
        fprintf(stdout, "        Initialize centroids:     %.2f ms\n", (float)(Tomp_gpu_seeding*1E3f));
        fprintf(stdout, "        ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_gpu_computeAssign*1E3f)/NbItersKMGPU);
        fprintf(stdout, "        UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_gpu_updateCentroids*1E3f)/NbItersKMGPU);
        fprintf(stdout, "        Nb of iterations:         %d\n", NbItersKMGPU);
        fprintf(stdout, "  - Total appli.: %.2f s\n", (float)Tomp_application);
    }
    
    
    // spectral clustering on GPU
    if (ClustAlgo == SC_GPU) {
        fprintf(stdout, "- Evaluation of clustering quality:\n");
        if (SCImpGPU == SP_NVG || SCImpGPU == SP_NVG_KM || SCImpGPU == SP_CUG) {
            fprintf(stdout, "    Modularity score: %.2f\n", ModularityScore);
            fprintf(stdout, "    Edge cut score:   %.2f\n", EdgeCutScore);
            fprintf(stdout, "    Ratio cut score:  %.2f\n", RatioCutScore);
        }
        fprintf(stdout, "- Elapsed time:\n");
        fprintf(stdout, "  - On CPU:\n");
        fprintf(stdout, "    Input data:     %.2f s\n", (float)(Tomp_cpu_readData));
        fprintf(stdout, "    Output results: %.2f s\n", (float)(Tomp_cpu_saveResults));
        fprintf(stdout, "  - On GPU:\n");
        fprintf(stdout, "    CUDA init:      %.2f s\n", (float)(Tomp_gpu_cuInit));
        fprintf(stdout, "    Data transfers: %f s\n", (float)(Tomp_cpu_gpu_transfers + Tomp_gpu_cpu_transfers));
        fprintf(stdout, "        CPU->GPU:   %.2f ms\n", (float)(Tomp_cpu_gpu_transfers*1E3f));
        fprintf(stdout, "        CPU<-GPU:   %.2f ms\n", (float)(Tomp_gpu_cpu_transfers*1E3f));
        fprintf(stdout, "    Feature scaling:     %.2f s\n", (float)(Tomp_gpu_featureScaling));
        switch(SCImpGPU) {
            case DN_CUS:  // Case 1: SC with cuSolverDN
            {
                fprintf(stdout, "    Spectral clustering: %.2f ms\n", (float)(Tomp_gpu_spectralClustering*1E3f));
                fprintf(stdout, "        Construct affinity matrix and its graph Laplacian: %.2f ms\n", (float)(Tomp_gpu_constructSimLapMatrix*1E3f));
                fprintf(stdout, "        Calculate eigenpairs by cuSolverDn:                %.2f ms\n", (float)(Tomp_gpu_cuSolverDNsyevdx*1E3f));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:          %.2f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        Normalize eigenvector matrix:                      %.2f ms\n", (float)(Tomp_gpu_normalizeEigenvectorMatrix*1E3f));
                fprintf(stdout, "        Final k-means(++) clustering:                      %.2f ms\n", (float)(Tomp_gpu_finalKmeansForSC*1E3f));
                fprintf(stdout, "            Initialize centroids:     %.2f ms\n", (float)(Tomp_gpu_seeding*1E3f));
                fprintf(stdout, "            ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_gpu_computeAssign*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_gpu_updateCentroids*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            Nb of iterations:         %d\n", NbItersKMGPU);
            } break;
            
            case SP_NVG :  // Case 2: SC with nvGRAPH
            {
                fprintf(stdout, "    Spectral clustering: %f s\n", (float)(Tomp_gpu_spectralClustering));
                fprintf(stdout, "        Construct affinity matrix in CSR: %f s\n", (float)(Tomp_gpu_constructSimMatrixInCSR));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:              %f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        nvGRAPH spectral clustering API:  %f s\n", (float)(Tomp_gpu_nvGRAPHSpectralClusteringAPI));
            } break;
            
            case SP_NVG_KM :  // Case 3: SC with nvGRAPH + our k-means(++)
            {
                fprintf(stdout, "    Spectral clustering: %.2f s\n", (float)(Tomp_gpu_spectralClustering));
                fprintf(stdout, "        Construct affinity matrix in CSR:       %.2f s\n", (float)(Tomp_gpu_constructSimMatrixInCSR));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:                    %.2f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        nvGRAPH eigen-cal & k-means++ 1st iter: %.2f s\n", (float)(Tomp_gpu_nvGRAPHSpectralClusteringAPI));
                if (FlagAutoTuneNbClusters == 1) {
                    fprintf(stdout, "        Auto-tune the nb of clusters based on eigengaps: %.2f s\n", (float)(Tomp_gpu_autoTuneNbClusters));
                }
                fprintf(stdout, "        Normalize eigenvector matrix:           %.2f s\n", (float)(Tomp_gpu_normalizeEigenvectorMatrix));
                fprintf(stdout, "        Final k-means(++) clustering:           %.2f s\n", (float)(Tomp_gpu_finalKmeansForSC));
                fprintf(stdout, "            Initialize centroids:     %.2f ms\n", (float)(Tomp_gpu_seeding*1E3f));
                fprintf(stdout, "            ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_gpu_computeAssign*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_gpu_updateCentroids*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            Nb of iterations:         %d\n", NbItersKMGPU);
            } break;
            
            case SP_CUG:  // Case 4: SC with cuGraph
            {
                fprintf(stdout, "    Spectral clustering: %.2f s\n", (float)(Tomp_gpu_spectralClustering));
                fprintf(stdout, "        Construct affinity matrix in CSR: %.2f s\n", (float)(Tomp_gpu_constructSimMatrixInCSR));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:              %.2f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        cuGraph spectral clustering API:  %.2f s\n", (float)(Tomp_gpu_cuGraphSpectralClusteringAPI));
            } break;
            
            default :
                fprintf(stderr,"Unknown GPU implementation of spectral clustering!\n");
                exit(EXIT_FAILURE);
        }
        
        fprintf(stdout, "  - Total appli.: %.2f s\n", (float)Tomp_application);
    }
    
    
    // spectral clustering using reps on GPU
    if (ClustAlgo == SC_REPS && Chain == CHAIN_ON_GPU) {
        fprintf(stdout, "- Evaluation of clustering quality:\n");
        if (SCImpGPU == SP_NVG) {
            fprintf(stdout, "    Modularity score: %.2f\n", ModularityScore);
            fprintf(stdout, "    Edge cut score:   %.2f\n", EdgeCutScore);
            fprintf(stdout, "    Ratio cut score:  %.2f\n", RatioCutScore);
        }
        fprintf(stdout, "- Elapsed time:\n");
        fprintf(stdout, "  - On CPU:\n");
        fprintf(stdout, "    Input data:     %.2f s\n", (float)(Tomp_cpu_readData));
        fprintf(stdout, "    Output results: %.2f s\n", (float)(Tomp_cpu_saveResults));
        
        fprintf(stdout, "  - On GPU:\n");
        fprintf(stdout, "    CUDA init:      %.2f s\n", (float)(Tomp_gpu_cuInit));
        
        if (FlagFeatureScaling) {
            fprintf(stdout, "    Feature scaling:       %.2f s\n", (float)(Tomp_gpu_featureScaling));
        }
        if (MethodToExtractReps == ER_RS) {
            fprintf(stdout, "    Extract representatives via random sampling: %f s\n", (float)(Tomp_gpu_randomSampling + Tomp_gpu_attach));
            fprintf(stdout, "        Initialize centroids:     %.2f ms\n", (float)(Tomp_gpu_randomSampling*1E3f));
            fprintf(stdout, "        ComputeAssign (attach):   %.2f ms\n", (float)(Tomp_gpu_attach*1E3f));
        }
        if (MethodToExtractReps == ER_KM) {
            fprintf(stdout, "    Extract representatives via k-means:     %.2f s\n", (float)(Tomp_gpu_kmeans));
            fprintf(stdout, "        Initialize centroids:     %.2f ms\n", (float)(Tomp_gpu_seeding*1E3f));
            fprintf(stdout, "        ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_gpu_computeAssign*1E3f)/NbItersKMGPU);
            fprintf(stdout, "        UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_gpu_updateCentroids*1E3f)/NbItersKMGPU);
            fprintf(stdout, "        Nb of iterations:         %d\n", NbItersKMGPU);
        }
        if (MethodToExtractReps == ER_KMPP) {
            fprintf(stdout, "    Extract representatives via k-means++: %.2f s\n", (float)(Tomp_gpu_kmeanspp));
            fprintf(stdout, "        Initialize centroids:     %.2f ms\n", (float)(Tomp_gpu_seeding*1E3f));
            fprintf(stdout, "        ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_gpu_computeAssign*1E3f)/NbItersKMGPU);
            fprintf(stdout, "        UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_gpu_updateCentroids*1E3f)/NbItersKMGPU);
            fprintf(stdout, "        Nb of iterations:         %d\n", NbItersKMGPU);
        }
        
        fprintf(stdout, "    Data transfers: %.2f s\n", (float)(Tomp_cpu_gpu_transfers + Tomp_gpu_cpu_transfers));
        fprintf(stdout, "        CPU->GPU:   %.2f ms\n", (float)(Tomp_cpu_gpu_transfers*1E3f));
        fprintf(stdout, "        CPU<-GPU:   %.2f ms\n", (float)(Tomp_gpu_cpu_transfers*1E3f));
        fprintf(stdout, "    Transpose reps:      %.2f ms\n", (float)(Tomp_gpu_transposeReps*1E3f));
        switch(SCImpGPU) {
            case DN_CUS:  // Case 1: SC with cuSolverDN
            {
                fprintf(stdout, "    Spectral clustering: %.2f ms\n", (float)(Tomp_gpu_spectralClustering*1E3f));
                fprintf(stdout, "        Construct affinity matrix and its graph Laplacian: %.2f ms\n", (float)(Tomp_gpu_constructSimLapMatrix*1E3f));
                fprintf(stdout, "        Calculate eigenpairs by cuSolverDn:                %.2f ms\n", (float)(Tomp_gpu_cuSolverDNsyevdx*1E3f));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:                               %.2f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        Normalize eigenvector matrix:                      %.2f ms\n", (float)(Tomp_gpu_normalizeEigenvectorMatrix*1E3f));
                fprintf(stdout, "        Final k-means(++) clustering:                      %.2f ms\n", (float)(Tomp_gpu_finalKmeansForSC*1E3f));
                fprintf(stdout, "            Initialize centroids:     %.2f ms\n", (float)(Tomp_gpu_seeding*1E3f));
                fprintf(stdout, "            ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_gpu_computeAssign*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_gpu_updateCentroids*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            Nb of iterations:         %d\n", NbItersKMGPU);
            } break;
            
            case SP_NVG :  // Case 2: SC with nvGRAPH
            {
                fprintf(stdout, "    Spectral clustering: %.2f s\n", (float)(Tomp_gpu_spectralClustering));
                fprintf(stdout, "        Construct affinity matrix in CSR: %f s\n", (float)(Tomp_gpu_constructSimMatrixInCSR));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:              %f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        nvGRAPH spectral clustering API:  %.2f s\n", (float)(Tomp_gpu_nvGRAPHSpectralClusteringAPI));
            } break;
            
            case SP_NVG_KM :  // Case 3: SC with nvGRAPH + our k-means(++)
            {
                fprintf(stdout, "    Spectral clustering: %.2f s\n", (float)(Tomp_gpu_spectralClustering));
                fprintf(stdout, "        Construct affinity matrix in CSR:       %.2f s\n", (float)(Tomp_gpu_constructSimMatrixInCSR));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:                    %.2f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        nvGRAPH eigen-cal & k-means++ 1st iter: %.2f s\n", (float)(Tomp_gpu_nvGRAPHSpectralClusteringAPI));
                if (FlagAutoTuneNbClusters == 1) {
                    fprintf(stdout, "        Auto-tune the nb of clusters based on eigengaps: %.2f s\n", (float)(Tomp_gpu_autoTuneNbClusters));
                }
                fprintf(stdout, "        Normalize eigenvector matrix:           %.2f s\n", (float)(Tomp_gpu_normalizeEigenvectorMatrix));
                fprintf(stdout, "        Final k-means(++) clustering:           %.2f s\n", (float)(Tomp_gpu_finalKmeansForSC));
                fprintf(stdout, "            Initialize centroids:     %.2f ms\n", (float)(Tomp_gpu_seeding*1E3f));
                fprintf(stdout, "            ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_gpu_computeAssign*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_gpu_updateCentroids*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            Nb of iterations:         %d\n", NbItersKMGPU);
            } break;
            
            case SP_CUG:   // Case 4: SC with cuGraph
            {
                fprintf(stdout, "    Spectral clustering: %.2f s\n", (float)(Tomp_gpu_spectralClustering));
                fprintf(stdout, "        Construct affinity matrix in CSR: %.2f s\n", (float)(Tomp_gpu_constructSimMatrixInCSR));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:              %.2f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        cuGraph spectral clustering API:  %.2f s\n", (float)(Tomp_gpu_cuGraphSpectralClusteringAPI));
            } break;
            
            default :
                fprintf(stderr,"Unknown GPU implementation of spectral clustering!\n");
                exit(EXIT_FAILURE);
        }
        
        fprintf(stdout, "    Membership recovering: %f s\n", (float)(Tomp_gpu_membershipAttach));
        
        fprintf(stdout, "  - Total appli.: %.2f s\n", (float)Tomp_application);
    }
 
    
    // spectral clustering using reps on CPU+GPU
    if (ClustAlgo == SC_REPS && Chain == CHAIN_ON_CPU_GPU) {
        fprintf(stdout, "- Evaluation of clustering quality:\n");
        if (SCImpGPU == SP_NVG || SCImpGPU == SP_NVG_KM || SCImpGPU == SP_CUG) {
            fprintf(stdout, "    Modularity score: %.2f\n", ModularityScore);
            fprintf(stdout, "    Edge cut score:   %.2f\n", EdgeCutScore);
            fprintf(stdout, "    Ratio cut score:  %.2f\n", RatioCutScore);
        }
        fprintf(stdout, "- Elapsed time:\n");
        fprintf(stdout, "  - On CPU:\n");
        fprintf(stdout, "    Input data:     %.2f s\n", (float)(Tomp_cpu_readData));
        fprintf(stdout, "    Output results: %.2f s\n", (float)(Tomp_cpu_saveResults));
        if (FlagFeatureScaling) {
            fprintf(stdout, "    Feature scaling:       %.2f s\n", (float)(Tomp_cpu_featureScaling));
        }
        if (MethodToExtractReps == ER_RS) {
            fprintf(stdout, "    Extract representatives via random sampling: %.2f s\n", (float)(Tomp_cpu_randomSampling + Tomp_cpu_attach));
            fprintf(stdout, "        Initialize centroids:     %.2f ms\n", (float)(Tomp_cpu_randomSampling*1E3f));
            fprintf(stdout, "        ComputeAssign (attach):   %.2f ms\n", (float)(Tomp_cpu_attach*1E3f));
        }
        if (MethodToExtractReps == ER_KM) {
            fprintf(stdout, "    Extract representatives via k-means: %.2f s\n", (float)(Tomp_cpu_kmeans));
            fprintf(stdout, "        Initialize centroids:     %.2f ms\n", (float)(Tomp_cpu_seeding*1E3f));
            fprintf(stdout, "        ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_cpu_computeAssign*1E3f)/NbItersKMCPU);
            fprintf(stdout, "        UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_cpu_updateCentroids*1E3f)/NbItersKMCPU);
            fprintf(stdout, "        Nb of iterations:         %d\n", NbItersKMCPU);
        }
        if (MethodToExtractReps == ER_KMPP) {
            fprintf(stdout, "    Extract representatives via k-means(++): %.2f s\n", (float)(Tomp_cpu_kmeans));
            fprintf(stdout, "        Initialize centroids:     %.2f ms\n", (float)(Tomp_cpu_seeding*1E3f));
            fprintf(stdout, "        ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_cpu_computeAssign*1E3f)/NbItersKMCPU);
            fprintf(stdout, "        UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_cpu_updateCentroids*1E3f)/NbItersKMCPU);
            fprintf(stdout, "        Nb of iterations:         %d\n", NbItersKMCPU);
        }
        fprintf(stdout, "    Membership recovering: %.2f s\n", (float)(Tomp_cpu_membershipAttach));
        
        fprintf(stdout, "  - On GPU:\n");
        fprintf(stdout, "    CUDA init:      %.2f s\n", (float)(Tomp_gpu_cuInit));
        fprintf(stdout, "    Data transfers: %.2f s\n", (float)(Tomp_cpu_gpu_transfers + Tomp_gpu_cpu_transfers));
        fprintf(stdout, "        CPU->GPU:   %.2f ms\n", (float)(Tomp_cpu_gpu_transfers*1E3f));
        fprintf(stdout, "        CPU<-GPU:   %.2f ms\n", (float)(Tomp_gpu_cpu_transfers*1E3f));
        fprintf(stdout, "    Transpose reps:      %.2f ms\n", (float)(Tomp_gpu_transposeReps*1E3f));
        switch(SCImpGPU) {
            case DN_CUS:  // Case 1: SC with cuSolverDN
            {
                fprintf(stdout, "    Spectral clustering: %.2f ms\n", (float)(Tomp_gpu_spectralClustering*1E3f));
                fprintf(stdout, "        Construct affinity matrix and its graph Laplacian: %.2f ms\n", (float)(Tomp_gpu_constructSimLapMatrix*1E3f));
                fprintf(stdout, "        Calculate eigenpairs by cuSolverDn:                %.2f ms\n", (float)(Tomp_gpu_cuSolverDNsyevdx*1E3f));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:                               %.2f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        Normalize eigenvector matrix:                      %.2f ms\n", (float)(Tomp_gpu_normalizeEigenvectorMatrix*1E3f));
                fprintf(stdout, "        Final k-means(++) clustering:                      %.2f ms\n", (float)(Tomp_gpu_finalKmeansForSC*1E3f));
                fprintf(stdout, "            Initialize centroids:     %.2f ms\n", (float)(Tomp_gpu_seeding*1E3f));
                fprintf(stdout, "            ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_gpu_computeAssign*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_gpu_updateCentroids*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            Nb of iterations:         %d\n", NbItersKMGPU);
            } break;
            
            case SP_NVG :  // Case 2: SC with nvGRAPH
            {
                fprintf(stdout, "    Spectral clustering: %.2f s\n", (float)(Tomp_gpu_spectralClustering));
                fprintf(stdout, "        Construct affinity matrix in CSR: %f s\n", (float)(Tomp_gpu_constructSimMatrixInCSR));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:              %f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        nvGRAPH spectral clustering API:  %.2f s\n", (float)(Tomp_gpu_nvGRAPHSpectralClusteringAPI));
            } break;
            
            case SP_NVG_KM :  // Case 3: SC with nvGRAPH + our k-means(++)
            {
                fprintf(stdout, "    Spectral clustering: %.2f s\n", (float)(Tomp_gpu_spectralClustering));
                fprintf(stdout, "        Construct affinity matrix in CSR:       %.2f s\n", (float)(Tomp_gpu_constructSimMatrixInCSR));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:                    %.2f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        nvGRAPH eigen-cal & k-means++ 1st iter: %.2f s\n", (float)(Tomp_gpu_nvGRAPHSpectralClusteringAPI));
                if (FlagAutoTuneNbClusters == 1) {
                    fprintf(stdout, "        Auto-tune the nb of clusters based on eigengaps: %.2f s\n", (float)(Tomp_gpu_autoTuneNbClusters));
                }
                fprintf(stdout, "        Normalize eigenvector matrix:           %.2f s\n", (float)(Tomp_gpu_normalizeEigenvectorMatrix));
                fprintf(stdout, "        Final k-means(++) clustering:           %.2f s\n", (float)(Tomp_gpu_finalKmeansForSC));
                fprintf(stdout, "            Initialize centroids:     %.2f ms\n", (float)(Tomp_gpu_seeding*1E3f));
                fprintf(stdout, "            ComputeAssign per iter:   %.2f ms\n", (float)(Tomp_gpu_computeAssign*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            UpdateCentroids per iter: %.2f ms\n", (float)(Tomp_gpu_updateCentroids*1E3f)/NbItersKMGPU);
                fprintf(stdout, "            Nb of iterations:         %d\n", NbItersKMGPU);
            } break;
            
            case SP_CUG:   // Case 4: SC with cuGraph
            {
                fprintf(stdout, "    Spectral clustering: %.2f s\n", (float)(Tomp_gpu_spectralClustering));
                fprintf(stdout, "        Construct affinity matrix in CSR: %.2f s\n", (float)(Tomp_gpu_constructSimMatrixInCSR));
                if (FilterNoiseApproach != 0) {
                    fprintf(stdout, "        Filter noise in CSR:              %.2f s\n", (float)(Tomp_gpu_filterNoise));
                }
                fprintf(stdout, "        cuGraph spectral clustering API:  %.2f s\n", (float)(Tomp_gpu_cuGraphSpectralClusteringAPI));
            } break;
            
            default :
                fprintf(stderr,"Unknown GPU implementation of spectral clustering!\n");
                exit(EXIT_FAILURE);
        }
        
        fprintf(stdout, "  - Total appli.: %.2f s\n", (float)Tomp_application);
    }
 
 
    fflush(stdout);
}
