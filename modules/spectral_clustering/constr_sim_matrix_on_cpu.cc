#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <float.h>   // Library Macros (e.g. FLT_MAX, FLT_MIN)
#include <climits>   // INT_MAX 
#include <math.h>    // expf
#include <omp.h>     // omp_get_wtime

#include "../../include/config.h"
#include "../../include/vars.h"
#include "../../include/utilities/dataIO.h"
#include "../../include/spectral_clustering/constr_sim_matrix_on_cpu.h"


void constr_dense_sim_matrix (int nbPoints, int nbDims, T_real *data,
                              T_real sigma, T_real tholdSim, T_real tholdSqDist)
{
    T_real *sim;
    sim = (T_real *)calloc(((size_t)nbPoints)*((size_t)nbPoints), sizeof(T_real));  // allocate and zero out the similarity array
    
    #pragma omp parallel
    {
        // Uniform similarity with threshold for squared distance
        #ifdef UNI_SIM_WITH_SQDIST_THOLD
            #pragma omp for 
            for (int i = 0; i < nbPoints; i++) {
                for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                    T_real sqDist = 0.0f;
                    T_real diffTab[NB_DIMS];
                    T_real sqDistTab[NB_DIMS] = {0.0f};
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                        sqDistTab[d] += diffTab[d]*diffTab[d];
                    }
                    for (int d = 0; d < NB_DIMS; d++) {
                        sqDist += sqDistTab[d];
                    }
                    if (sqDist < tholdSqDist && i != j) {
                        size_t idxSim = ((size_t)i)*((size_t)nbPoints) + ((size_t)j);  // Avoid integer overflow
                        sim[idxSim] = 1.0f;   //sim[j*nbPoints + i] = 1.0f;
                    }
                }
            }
        #endif
        
        // Gaussian similarity with threshold for squared distance
        #ifdef GAUSS_SIM_WITH_SQDIST_THOLD
            #pragma omp for 
            for (int i = 0; i < nbPoints; i++) {
                for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                    T_real sqDist = 0.0f;
                    T_real diffTab[NB_DIMS];
                    T_real sqDistTab[NB_DIMS] = {0.0f};
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                        sqDistTab[d] += diffTab[d]*diffTab[d];
                    }
                    for (int d = 0; d < NB_DIMS; d++) {
                        sqDist += sqDistTab[d];
                    }
                    if (sqDist < tholdSqDist && i != j) {
                        size_t idxSim = ((size_t)i)*((size_t)nbPoints) + ((size_t)j);  // Avoid integer overflow
                        sim[idxSim] = expf((-1.0f)*sqDist/(2.0f*sigma*sigma));  // possible underflow of the similarity value
                    }
                }
            }
        #endif
        
        // Gaussian similarity with threshold
        #ifdef GAUSS_SIM_WITH_THOLD
            #pragma omp for 
            for (int i = 0; i < nbPoints; i++) {
                for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                    T_real sqDist = 0.0f;
                    T_real diffTab[NB_DIMS];
                    T_real sqDistTab[NB_DIMS] = {0.0f};
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                        sqDistTab[d] += diffTab[d]*diffTab[d];
                    }
                    for (int d = 0; d < NB_DIMS; d++) {
                        sqDist += sqDistTab[d];
                    }
                    T_real s = expf((-1.0f)*sqDist/(2.0f*sigma*sigma));   // possible underflow of the similarity value
                    if (s > tholdSim && i != j) {
                        size_t idxSim = ((size_t)i)*((size_t)nbPoints) + ((size_t)j);  // Avoid integer overflow
                        sim[idxSim] = s;   //sim[j*nbPoints + i] = s;
                    }
                }
            }
        #endif

        // Cosine similarity with threshold
        #ifdef COS_SIM_WITH_THOLD
            #pragma omp for 
            for (int i = 0; i < nbPoints; i++) {
                for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                    T_real dot = 0.0f, sq1 = 0.0f, sq2 = 0.0f;
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        T_real elm1 = data[idxOffsetDatai + (index_t)d];
                        T_real elm2 = data[idxOffsetDataj + (index_t)d];
                        dot += elm1*elm2;
                        sq1 += elm1*elm1;
                        sq2 += elm2*elm2;
                    }
                    T_real sqSim = (dot*dot)/(sq1*sq2);
                    if (sqSim > tholdSim*tholdSim && i != j) {
                        size_t idxSim = ((size_t)i)*((size_t)nbPoints) + ((size_t)j);  // Avoid integer overflow
                        sim[idxSim] = SQRT(sqSim);   //sim[j*nbPoints + i] = s;
                    }
                }
            }
        #endif
    }
    
    
    double begin, finish;
    begin = omp_get_wtime();
    save_file_real(sim, nbPoints, nbPoints, "output/SimMat-dense.txt", "\t");
    finish = omp_get_wtime();
    Tomp_cpu_saveSimMatrix += (finish - begin);
    
    // Deallocate memory
    free(sim);
}


void constr_csr_sim_matrix_using_monothread (int nbPoints, int nbDims, T_real *data,
                                             T_real sigma, T_real tholdSim, T_real tholdSqDist)
{
    int *csrRow, *csrCol;
    T_real *csrVal;
    int nnz = 0;
    csrRow = (int *) malloc((nbPoints + 1)*sizeof(int));
    
    // Uniform or Gaussian similarity with threshold for squared distance
    #if defined(UNI_SIM_WITH_SQDIST_THOLD) || defined(GAUSS_SIM_WITH_SQDIST_THOLD)
        for (int i = 0; i < nbPoints; i++) {
            csrRow[i] = nnz;
            for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                T_real sqDist = 0.0f;
                T_real diffTab[NB_DIMS];
                T_real sqDistTab[NB_DIMS] = {0.0f};
                index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                for (int d = 0; d < NB_DIMS; d++) {
                    diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                    sqDistTab[d] += diffTab[d]*diffTab[d];
                }
                for (int d = 0; d < NB_DIMS; d++) {
                    sqDist += sqDistTab[d];
                }
                if (sqDist < tholdSqDist && i != j) {
                    nnz++;
                }
            }
        }
    #endif
    
    // Gaussian similarity with threshold
    #ifdef GAUSS_SIM_WITH_THOLD
        for (int i = 0; i < nbPoints; i++) {
            csrRow[i] = nnz;
            for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                T_real sqDist = 0.0f;
                T_real diffTab[NB_DIMS];
                T_real sqDistTab[NB_DIMS] = {0.0f};
                index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                for (int d = 0; d < NB_DIMS; d++) {
                    diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                    sqDistTab[d] += diffTab[d]*diffTab[d];
                }
                for (int d = 0; d < NB_DIMS; d++) {
                    sqDist += sqDistTab[d];
                }
                T_real s = expf((-1.0f)*sqDist/(2.0f*sigma*sigma));   // possible underflow of the similarity value
                if (s > tholdSim && i != j) {
                    nnz++;
                }
            }
        }
    #endif

    // Cosine similarity with threshold
    #ifdef COS_SIM_WITH_THOLD
        for (int i = 0; i < nbPoints; i++) {
            csrRow[i] = nnz;
            for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                T_real dot = 0.0f, sq1 = 0.0f, sq2 = 0.0f;
                index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                for (int d = 0; d < NB_DIMS; d++) {
                    T_real elm1 = data[idxOffsetDatai + (index_t)d];
                    T_real elm2 = data[idxOffsetDataj + (index_t)d];
                    dot += elm1*elm2;
                    sq1 += elm1*elm1;
                    sq2 += elm2*elm2;
                }
                T_real sqSim = (dot*dot)/(sq1*sq2);
                if (sqSim > tholdSim*tholdSim && i != j) {
                    nnz++;
                }
            }
        }
    #endif
    
    csrRow[nbPoints] = nnz;
    if (nnz > 0) {
        printf("    Average nnz per row: %d\n", nnz/nbPoints);
        printf("    Total nnz:           %d\n", nnz);
        printf("    Sparsity:            %.3lf%%\n", 100 - ((((double)nnz/nbPoints)*100)/nbPoints));
    } else {
        printf("Total number of nonzeros/edges exceeds the maximum limit of integer (%d), leading to integer overflow !\n", INT_MAX);
        exit(EXIT_FAILURE);
    }
    csrVal = (T_real *) malloc(nnz*sizeof(T_real));
    csrCol = (int *) malloc(nnz*sizeof(int));
    nnz = 0;
    
    // Uniform similarity with threshold for squared distance
    #ifdef UNI_SIM_WITH_SQDIST_THOLD
        #pragma omp for 
        for (int i = 0; i < nbPoints; i++) {
            for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                T_real sqDist = 0.0f;
                T_real diffTab[NB_DIMS];
                T_real sqDistTab[NB_DIMS] = {0.0f};
                index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                for (int d = 0; d < NB_DIMS; d++) {
                    diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                    sqDistTab[d] += diffTab[d]*diffTab[d];
                }
                for (int d = 0; d < NB_DIMS; d++) {
                    sqDist += sqDistTab[d];
                }
                if (sqDist < tholdSqDist && i != j) {
                    csrVal[nnz] = 1.0f;
                    csrCol[nnz] = j;
                    nnz++;
                }
            }
        }
    #endif
    
    // Gaussian similarity with threshold for squared distance
    #ifdef GAUSS_SIM_WITH_SQDIST_THOLD
        #pragma omp for 
        for (int i = 0; i < nbPoints; i++) {
            for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                T_real sqDist = 0.0f;
                T_real diffTab[NB_DIMS];
                T_real sqDistTab[NB_DIMS] = {0.0f};
                index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                for (int d = 0; d < NB_DIMS; d++) {
                    diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                    sqDistTab[d] += diffTab[d]*diffTab[d];
                }
                for (int d = 0; d < NB_DIMS; d++) {
                    sqDist += sqDistTab[d];
                }
                if (sqDist < tholdSqDist && i != j) {
                    csrVal[nnz] = expf((-1.0f)*sqDist/(2.0f*sigma*sigma));  // possible underflow of the similarity value
                    csrCol[nnz] = j;
                    nnz++;
                }
            }
        }
    #endif
    
    // Gaussian similarity with threshold
    #ifdef GAUSS_SIM_WITH_THOLD
        #pragma omp for 
        for (int i = 0; i < nbPoints; i++) {
            for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                T_real sqDist = 0.0f;
                T_real diffTab[NB_DIMS];
                T_real sqDistTab[NB_DIMS] = {0.0f};
                index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                for (int d = 0; d < NB_DIMS; d++) {
                    diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                    sqDistTab[d] += diffTab[d]*diffTab[d];
                }
                for (int d = 0; d < NB_DIMS; d++) {
                    sqDist += sqDistTab[d];
                }
                T_real s = expf((-1.0f)*sqDist/(2.0f*sigma*sigma));   // possible underflow of the similarity value
                if (s > tholdSim && i != j) {
                    csrVal[nnz] = s;
                    csrCol[nnz] = j;
                    nnz++;
                }
            }
        }
    #endif

    // Cosine similarity with threshold
    #ifdef COS_SIM_WITH_THOLD
        #pragma omp for 
        for (int i = 0; i < nbPoints; i++) {
            for (int j = 0; j < nbPoints; j++) { // for (int j = i; j < nbPoints; j++)
                T_real dot = 0.0f, sq1 = 0.0f, sq2 = 0.0f;
                index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                for (int d = 0; d < NB_DIMS; d++) {
                    T_real elm1 = data[idxOffsetDatai + (index_t)d];
                    T_real elm2 = data[idxOffsetDataj + (index_t)d];
                    dot += elm1*elm2;
                    sq1 += elm1*elm1;
                    sq2 += elm2*elm2;
                }
                T_real sqSim = (dot*dot)/(sq1*sq2);
                if (sqSim > tholdSim*tholdSim && i != j) {
                    csrVal[nnz] = SQRT(sqSim);
                    csrCol[nnz] = j;
                    nnz++;
                }
            }
        }
    #endif
    
    // Save the CSR representation of similarity matrix
    double begin, finish;
    begin = omp_get_wtime();
    save_file_real(csrVal, nnz,          1, "output/SimMat-csrVal.txt", "");
    save_file_int (csrRow, nbPoints + 1, 1, "output/SimMat-csrRow.txt", "");
    save_file_int (csrCol, nnz,          1, "output/SimMat-csrCol.txt", "");
    finish = omp_get_wtime();
    Tomp_cpu_saveSimMatrix += (finish - begin);
    
    // Deallocate memory
    free(csrVal);
    free(csrRow);
    free(csrCol);
}


void constr_csr_sim_matrix_using_multithreads (int nbPoints, int nbDims, T_real *data,
                                               int NbThreadsCPU,
                                               T_real sigma, T_real tholdSim, T_real tholdSqDist)
{
    int nnz = 0;
    int *nnzPerChunk, *csrRow, *csrCol;
    T_real *csrVal;
    nnzPerChunk    = (int *)malloc(NbThreadsCPU*sizeof(int));
    csrRow         = (int *)malloc((nbPoints + 1)*sizeof(int));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nnzRow;
        int lastRowIdx;
        
        // Uniform or Gaussian similarity with threshold for squared distance
        #if defined(UNI_SIM_WITH_SQDIST_THOLD) || defined(GAUSS_SIM_WITH_SQDIST_THOLD)
            #pragma omp for reduction(+: nnz)
            for (int i = 0; i < nbPoints; i++) {
                csrRow[i] = nnz;
                nnzRow = 0;
                for (int j = 0; j < nbPoints; j++) {
                    T_real sqDist = 0.0f;
                    T_real diffTab[NB_DIMS];
                    T_real sqDistTab[NB_DIMS] = {0.0f};
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                        sqDistTab[d] += diffTab[d]*diffTab[d];
                    }
                    for (int d = 0; d < NB_DIMS; d++) {
                        sqDist += sqDistTab[d];
                    }
                    if (sqDist < tholdSqDist && i != j) {
                        nnzRow++;
                    }
                }
                nnz += nnzRow;
                lastRowIdx = i;
            }
        #endif
        
        // Gaussian similarity with threshold
        #ifdef GAUSS_SIM_WITH_THOLD
            #pragma omp for reduction(+: nnz)
            for (int i = 0; i < nbPoints; i++) {
                csrRow[i] = nnz;
                nnzRow = 0;
                for (int j = 0; j < nbPoints; j++) {
                    T_real sqDist = 0.0f;
                    T_real diffTab[NB_DIMS];
                    T_real sqDistTab[NB_DIMS] = {0.0f};
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                        sqDistTab[d] += diffTab[d]*diffTab[d];
                    }
                    for (int d = 0; d < NB_DIMS; d++) {
                        sqDist += sqDistTab[d];
                    }
                    T_real s = expf((-1.0f)*sqDist/(2.0f*sigma*sigma));   // possible underflow of the similarity value
                    if (s > tholdSim && i != j) {
                        nnzRow++;
                    }
                }
                nnz += nnzRow;
                lastRowIdx = i;
            }
        #endif
        
        // Cosine similarity with threshold
        #ifdef COS_SIM_WITH_THOLD
            #pragma omp for reduction(+: nnz)
            for (int i = 0; i < nbPoints; i++) {
                csrRow[i] = nnz;
                nnzRow = 0;
                for (int j = 0; j < nbPoints; j++) {
                    T_real dot = 0.0f, sq1 = 0.0f, sq2 = 0.0f;
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        T_real elm1 = data[idxOffsetDatai + (index_t)d];
                        T_real elm2 = data[idxOffsetDataj + (index_t)d];
                        dot += elm1*elm2;
                        sq1 += elm1*elm1;
                        sq2 += elm2*elm2;
                    }
                    T_real sqSim = (dot*dot)/(sq1*sq2);
                    if (sqSim > tholdSim*tholdSim && i != j) {
                        nnzRow++;
                    }
                }
                nnz += nnzRow;
                lastRowIdx = i;
            }
        #endif
        
        nnzPerChunk[tid] = csrRow[lastRowIdx] + nnzRow;
        
        #pragma omp single
        {
            csrRow[nbPoints] = nnz;
            if (nnz > 0) {
                printf("    Average nnz per row: %d\n", nnz/nbPoints);
                printf("    Total nnz:           %d\n", nnz);
                printf("    Sparsity:            %.3lf%%\n", 100 - ((((double)nnz/nbPoints)*100)/nbPoints));
            } else {
                printf("Total number of nonzeros/edges exceeds the maximum limit of integer (%d), leading to integer overflow !\n", INT_MAX);
                exit(EXIT_FAILURE);
            }
            csrVal = (T_real *) malloc(nnz*sizeof(T_real));
            csrCol = (int *) malloc(nnz*sizeof(int));
        } // There is an implicit barrier at the end of the single construct unless a nowait clause is specified.
        
        int nnzChunkOffset = 0;
        for (int i = 0; i < tid; i++) {
            nnzChunkOffset += nnzPerChunk[i]; // Get the nnzChunkOffset for each omp partitions
        }
        
        // Uniform similarity with threshold for squared distance
        #ifdef UNI_SIM_WITH_SQDIST_THOLD
            #pragma omp for 
            for (int i = 0; i < nbPoints; i++) {
                int nnzOffset = nnzChunkOffset + csrRow[i];
                csrRow[i] = nnzOffset;
                for (int j = 0; j < nbPoints; j++) {
                    T_real sqDist = 0.0f;
                    T_real diffTab[NB_DIMS];
                    T_real sqDistTab[NB_DIMS] = {0.0f};
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                        sqDistTab[d] += diffTab[d]*diffTab[d];
                    }
                    for (int d = 0; d < NB_DIMS; d++) {
                        sqDist += sqDistTab[d];
                    }
                    if (sqDist < tholdSqDist && i != j) {
                        csrVal[nnzOffset] = 1.0f;
                        csrCol[nnzOffset] = j;
                        nnzOffset++;
                    }
                }
            }
        #endif
        
        // Gaussian similarity with threshold for squared distance
        #ifdef GAUSS_SIM_WITH_SQDIST_THOLD
            #pragma omp for 
            for (int i = 0; i < nbPoints; i++) {
                int nnzOffset = nnzChunkOffset + csrRow[i];
                csrRow[i] = nnzOffset;
                for (int j = 0; j < nbPoints; j++) {
                    T_real sqDist = 0.0f;
                    T_real diffTab[NB_DIMS];
                    T_real sqDistTab[NB_DIMS] = {0.0f};
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                        sqDistTab[d] += diffTab[d]*diffTab[d];
                    }
                    for (int d = 0; d < NB_DIMS; d++) {
                        sqDist += sqDistTab[d];
                    }
                    if (sqDist < tholdSqDist && i != j) {
                        csrVal[nnzOffset] = expf((-1.0f)*sqDist/(2.0f*sigma*sigma));  // possible underflow of the similarity value
                        csrCol[nnzOffset] = j;
                        nnzOffset++;
                    }
                }
            }
        #endif
        
        // Gaussian similarity with threshold
        #ifdef GAUSS_SIM_WITH_THOLD
            #pragma omp for
            for (int i = 0; i < nbPoints; i++) {
                int nnzOffset = nnzChunkOffset + csrRow[i];
                csrRow[i] = nnzOffset;
                for (int j = 0; j < nbPoints; j++) { 
                    T_real sqDist = 0.0f;
                    T_real diffTab[NB_DIMS];
                    T_real sqDistTab[NB_DIMS] = {0.0f};
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        diffTab[d] = data[idxOffsetDatai + (index_t)d] - data[idxOffsetDataj + (index_t)d];
                        sqDistTab[d] += diffTab[d]*diffTab[d];
                    }
                    for (int d = 0; d < NB_DIMS; d++) {
                        sqDist += sqDistTab[d];
                    }
                    T_real s = expf((-1.0f)*sqDist/(2.0f*sigma*sigma));   // possible underflow of the similarity value
                    if (s > tholdSim && i != j) {
                        csrVal[nnzOffset] = s;
                        csrCol[nnzOffset] = j;
                        nnzOffset++;
                    }
                }
            }
        #endif
        
        // Cosine similarity with threshold
        #ifdef COS_SIM_WITH_THOLD
            #pragma omp for 
            for (int i = 0; i < nbPoints; i++) {
                int nnzOffset = nnzChunkOffset + csrRow[i];
                csrRow[i] = nnzOffset;
                for (int j = 0; j < nbPoints; j++) {
                    T_real dot = 0.0f, sq1 = 0.0f, sq2 = 0.0f;
                    index_t idxOffsetDatai = ((index_t)i)*((index_t)NB_DIMS);
                    index_t idxOffsetDataj = ((index_t)j)*((index_t)NB_DIMS);
                    for (int d = 0; d < NB_DIMS; d++) {
                        T_real elm1 = data[idxOffsetDatai + (index_t)d];
                        T_real elm2 = data[idxOffsetDataj + (index_t)d];
                        dot += elm1*elm2;
                        sq1 += elm1*elm1;
                        sq2 += elm2*elm2;
                    }
                    T_real sqSim = (dot*dot)/(sq1*sq2);
                    if (sqSim > tholdSim*tholdSim && i != j) {
                        csrVal[nnzOffset] = SQRT(sqSim);
                        csrCol[nnzOffset] = j;
                        nnzOffset++;
                    }
                }
            }
        #endif
    }
    
    // Save the CSR representation of similarity matrix
    double begin, finish;
    begin = omp_get_wtime();
    save_file_real(csrVal, nnz,          1, "output/SimMat-csrVal.txt", "");
    save_file_int (csrRow, nbPoints + 1, 1, "output/SimMat-csrRow.txt", "");
    save_file_int (csrCol, nnz,          1, "output/SimMat-csrCol.txt", "");
    finish = omp_get_wtime();
    Tomp_cpu_saveSimMatrix += (finish - begin);
    
    // Deallocate memory
    free(csrVal);
    free(csrRow);
    free(csrCol);
    free(nnzPerChunk);
}


void constr_similarity_matrix_on_cpu (int nbPoints, int nbDims, T_real *data,
                                      int constrAlgoCPU, int NbThreadsCPU,
                                      T_real sigma, T_real tholdSim, T_real tholdSqDist)
{
    switch(constrAlgoCPU) {
        case 1 :  // Dense
            constr_dense_sim_matrix(nbPoints, nbDims, data,
                                    sigma, tholdSim, tholdSqDist);
            break;
        
        case 2 :  // CSR, monothread version
            constr_csr_sim_matrix_using_monothread(nbPoints, nbDims, data,
                                                   sigma, tholdSim, tholdSqDist);
        break;

        case 3 :  // CSR, multi-thread version
            constr_csr_sim_matrix_using_multithreads(nbPoints, nbDims, data,
                                                     NbThreadsCPU,
                                                     sigma, tholdSim, tholdSqDist);
        break;
        
        default :
            fprintf(stderr, "Unknown algo for constructing similarity matrix on CPU!");
            exit(EXIT_FAILURE);
    }
}
