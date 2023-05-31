#include <stdio.h>  // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h> // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <float.h>  // Library Macros (e.g. FLT_MAX, FLT_MIN)
#include <omp.h>

#include "../../include/config.h"
#include "../../include/utilities/feature_scaling_cpu.h"


void feature_scaling (int nbPoints, int nbDims,
                      T_real *data, 
                      T_real *dimMax, T_real *dimMin)
{
    // Declaration
    T_real dimMaxVal[NB_DIMS];   // Maximal value in each dimension
    T_real dimMinVal[NB_DIMS];   // Minimal value in each dimension
    
    #pragma omp parallel
    {
        // Declaration for each thread
        T_real value;
        
        // Initialize dimMaxVal & dimMinVal
        #pragma omp for
        for (int j = 0; j < NB_DIMS; j++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
            dimMaxVal[j] = -FLT_MAX;
            dimMinVal[j] = FLT_MAX;
        }
        
        // Find the maximal & minimal values of each dimension
        #pragma omp for reduction(max: dimMaxVal) reduction(min: dimMinVal)
        for (int i = 0; i < nbPoints; i++) {
            index_t idxOffset = ((index_t)i)*((index_t)nbDims);
            for (int j = 0; j < NB_DIMS; j++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                value = data[idxOffset + (index_t)j];
                if (value > dimMaxVal[j])  dimMaxVal[j] = value;
                if (value < dimMinVal[j])  dimMinVal[j] = value;
            }
        }
        
        // Perform min-max scaling
        #pragma omp for
        for (int i = 0; i < nbPoints; i++) {
            index_t idxOffset = ((index_t)i)*((index_t)nbDims);
            for (int j = 0; j < NB_DIMS; j++) { // Using the constant "NB_DIMS" instead of the variable "nbDims" may improve the performance significantly.
                T_real width = dimMaxVal[j] - dimMinVal[j];
                if (width != 0.0f) {
                    data[idxOffset + (index_t)j] = (data[idxOffset + (index_t)j] - dimMinVal[j]) / width;
                }
            }
        }
        
        #pragma omp for
        for (int j = 0; j < NB_DIMS; j++) {
            dimMax[j] = dimMaxVal[j];
            dimMin[j] = dimMinVal[j];
        }
    }
}


void inverse_feature_scaling (T_real *dimMax, T_real *dimMin,
                              int nbPoints, int nbDims,
                              T_real *data)
{   
    #pragma omp parallel
    {   
        #pragma omp for
        for (int i = 0; i < nbPoints; i++) {
            for (int j = 0; j < nbDims; j++) {
                data[i*nbDims + j] = data[i*nbDims + j] * (dimMax[j] - dimMin[j]) + dimMin[j];
            }
        }
    }
}