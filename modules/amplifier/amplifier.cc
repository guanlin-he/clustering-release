// To compile: g++ -fopenmp amplifier.cc -o amplifier

#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <string.h>  // Library functions (e.g. strcat, strcmp, strcpy, strtok)
#include <float.h>   // Library Macros (e.g. FLT_MAX, FLT_MIN)
#include <math.h>    // M_PI
#include <omp.h>

#define NB_POINTS         788
#define NB_DIMS           2
#define INPUT_DATA        "path-to-the-file/DATA_Aggregation.txt"
#define INPUT_REF_LABELS  "path-to-the-file/REF_Aggregation.txt"
#define OUTPUT_DATA       "path-to-the-file/DATA_Aggregation_A1E3_F2E-2.txt"
#define OUTPUT_REF_LABELS "path-to-the-file/REF_Aggregation_A1E3_F2E-2.txt"
#define MAX_LINE_LENGTH   4066
#define AMP_FACTOR        1000  // Amplifier factor
#define FLUC_FACTOR       0.02  // Fluctuation factor
#define SEED_BASE         1     // Seed base for the rand_r function
#define NB_THREADS        40

#ifdef DP                // Floating point datatype and op
typedef double T_real;
#define T_REAL_PRINT  "%lf"
#else
typedef float T_real;
#define T_REAL_PRINT  "%f"
#endif


// Read a file of integers
void read_file_int (int *array, size_t nbPoints, size_t nbDims, 
                    const char filename[], const char delim[], 
                    bool isTransposed)
{
    // Declaration
    FILE *fp = NULL;  // File pointer
    char line[MAX_LINE_LENGTH];
    char *token = NULL;
    size_t i = 0, j = 0, total = 0;

    // Open the file
    fp = fopen(filename, "r");  // Open the file in "read-only" mode
    if (fp == NULL) {           // Check if the file has been successfully opened
        fprintf(stderr, "    Fail to open the file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read a text file line by line
    // Note that there are nbDims values per line and each value is separated by a delimiter (e.g. ",", " ", "\t").
    while (fgets(line, MAX_LINE_LENGTH, fp) != NULL && i < nbPoints) {
        if (line[0] != '%'){  // Ignore the lines that begin with '%' (comments)
            // The C library function char *strtok(char *str, const char *delim) 
            // breaks string str into a series of tokens using the delimiter delim.
            // Get the first token
            token = strtok(line, delim); 
            
            // Walk through other tokens
            j = 0;
            if (isTransposed) {
                while (token != NULL && j < nbDims) {
                    array[j*nbPoints + i] = atoi(token);
                    token = strtok(NULL, delim);
                    j++;
                }
            } else {
                while (token != NULL && j < nbDims) {
                    array[i*nbDims + j] = atoi(token);
                    token = strtok(NULL, delim);
                    j++;
                }
            }
            i++;
            total += j;
        }
    }
    
    if (total != nbPoints*nbDims) {
        fprintf(stderr, "    Fail to read the text file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Close the file
    fclose(fp);
}


// Read a file of real numbers
void read_file_real (T_real *array, size_t nbPoints, size_t nbDims, 
                     const char filename[], const char delim[], 
                     bool isTransposed)
{
    // Declaration
    FILE *fp = NULL;  // File pointer
    char line[MAX_LINE_LENGTH];
    char *token = NULL;
    size_t i = 0, j = 0, total = 0;

    // Open the file
    fp = fopen(filename, "r");  // Open the file in "read-only" mode
    if (fp == NULL) {           // Check if the file has been successfully opened
        fprintf(stderr, "    Fail to open the file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read a text file line by line
    // Note that there are nbDims values per line and each value is separated by a delimiter (e.g. ",", " ", "\t").
    while (fgets(line, MAX_LINE_LENGTH, fp) != NULL && i < nbPoints) {
        if (line[0] != '%'){  // Ignore the lines that begin with '%' (comments)
            // The C library function char *strtok(char *str, const char *delim) 
            // breaks string str into a series of tokens using the delimiter delim.
            // Get the first token
            token = strtok(line, delim); 
            
            // Walk through other tokens
            j = 0;
            if (isTransposed) {
                while (token != NULL && j < nbDims) {
                    array[j*nbPoints + i] = atof(token);
                    token = strtok(NULL, delim);
                    j++;
                }
            } else {
                while (token != NULL && j < nbDims) {
                    array[i*nbDims + j] = atof(token);
                    token = strtok(NULL, delim);
                    j++;
                }
            }
            i++;
            total += j;
        }
    }
    
    if (total != nbPoints*nbDims) {
        fprintf(stderr, "    Fail to read the text file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Close the file
    fclose(fp);
}


// Save a file of integers
void save_file_int (int *array, size_t nbPoints, size_t nbDims, 
                    const char filename[], const char delim[])
{
    // Declaration
    FILE *fp = NULL;  // File pointer
    
    // Open the file
    fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("    Fail to open the file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
        
    // Write a text file line by line
    char line[MAX_LINE_LENGTH];
    char strInt[50];
    for (size_t i = 0; i < nbPoints; i++) {
        strcpy(line, "");
        for (size_t j = 0; j < nbDims; j++) {
            // The C library function char *strcpy(char *dest, const char *src) 
            // copies the string pointed to, by src to dest.
            strcpy(strInt, "");
            
            // The C library function int sprintf(char *str, const char *format, ...)
            // sends formatted output to a string pointed to, by str.
            sprintf(strInt, "%d%s", array[i*nbDims + j], delim);
            
            // The C library function char *strcat(char *dest, const char *src) 
            // appends the string pointed to by src to the end of the string pointed to by dest.
            strcat(line, strInt);
        }
        fprintf(fp, "%s\n", line);

    }

    // Close the file
    fclose(fp);
}


// Save a file of real numbers
void save_file_real (T_real *array, size_t nbPoints, size_t nbDims, 
                     const char filename[], const char delim[])
{
    // Declaration
    FILE *fp = NULL;  // File pointer

    // Open the file
    fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("    Fail to open the file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    // Write a text file line by line
    char line[MAX_LINE_LENGTH];
    char strReal[50];
    for (size_t i = 0; i < nbPoints; i++) {
        strcpy(line, "");              // Reset line to nothing
        for (size_t j = 0; j < nbDims; j++) {
            // The C library function char *strcpy(char *dest, const char *src) 
            // copies the string pointed to, by src to dest.
            strcpy(strReal, "");
            
            // The C library function int sprintf(char *str, const char *format, ...)
            // sends formatted output to a string pointed to, by str.
            sprintf(strReal, T_REAL_PRINT"%s", array[i*nbDims + j], delim);
            
            // The C library function char *strcat(char *dest, const char *src) 
            // appends the string pointed to by src to the end of the string pointed to by dest.
            strcat(line, strReal);
        }
        fprintf(fp, "%s\n", line);

    }
    
    // Close the file
    fclose(fp);
}


void find_range(T_real *array, T_real *width)
{
    T_real dimMax[NB_DIMS];
    T_real dimMin[NB_DIMS];
        
    #pragma omp parallel
    {
        T_real value;
        
        #pragma omp for
        for (int j = 0; j < NB_DIMS; j++) {
            dimMax[j] = -FLT_MAX;
            dimMin[j] = FLT_MAX;
        }
        
        #pragma omp for reduction(max: dimMax) reduction(min: dimMin)
        for (int i = 0; i < NB_POINTS; i++) {
            for (int j = 0; j < NB_DIMS; j++) {
                value = array[i*NB_DIMS +j];
                if (value > dimMax[j])  dimMax[j] = value;
                if (value < dimMin[j])  dimMin[j] = value;
            }
        }
        
        #pragma omp for
        for (int j = 0; j < NB_DIMS; j++) {
            width[j] = dimMax[j] - dimMin[j];
        }
        
        // Print max, min and width of dimensions
        // #pragma omp single 
        // {
            // for (int j = 0; j < NB_DIMS; j++) {
                // printf("    Max[%d]   = %f", j, dimMax[j]);
                // printf("    Min[%d]   = %f", j, dimMin[j]);
                // printf("    Range[%d] = %f\n", j, width[j]);
            // }
        // }
    }
}


void amplifier_manhattan(T_real *inputData, int *inputLabels, T_real *width, unsigned int seedBase, T_real *outputData, int *outputLabels)
{
    #pragma omp parallel
    {
        unsigned int seed = seedBase * omp_get_thread_num();
        T_real randReal = 0.0f;
        
        //#pragma omp for
        for (int a = 0; a < AMP_FACTOR; a++) {
            #pragma omp for
            for (int i = 0; i < NB_POINTS; i++) {
                for (int j = 0; j < NB_DIMS; j++) {
                    randReal = rand_r(&seed)/(T_real)RAND_MAX * 2 - 1;  // Generate a random real number in [-1, 1]
                    outputData[(a*NB_POINTS + i)*NB_DIMS + j] = inputData[i*NB_DIMS + j] + width[j]*FLUC_FACTOR*randReal;
                }
                outputLabels[a*NB_POINTS + i] = inputLabels[i];
            }
        }
    }
}


void amplifier_circle_2d(T_real *inputData, int *inputLabels, T_real *width, unsigned int seedBase, T_real *outputData, int *outputLabels)
{
    #pragma omp parallel
    {
        unsigned int seed = seedBase * omp_get_thread_num();
        T_real randReal = 0.0f;
        T_real randAngle = 0.0f;
        
        //#pragma omp for
        for (int a = 0; a < AMP_FACTOR; a++) {
            #pragma omp for
            for (int i = 0; i < NB_POINTS; i++) {
                randReal = rand_r(&seed)/(T_real)RAND_MAX;  // Generate a random real number in [0, 1]
                randAngle = 2*M_PI*randReal;
                randReal = rand_r(&seed)/(T_real)RAND_MAX;  // Generate a random real number in [0, 1]
                outputData[(a*NB_POINTS + i)*NB_DIMS]     = inputData[i*NB_DIMS]     + width[0]*FLUC_FACTOR*sqrt(randReal)*cos(randAngle);
                outputData[(a*NB_POINTS + i)*NB_DIMS + 1] = inputData[i*NB_DIMS + 1] + width[1]*FLUC_FACTOR*sqrt(randReal)*sin(randAngle);
                outputLabels[a*NB_POINTS + i] = inputLabels[i];
            }
        }
    }
}


int main()
{
    double begin, finish;
    double beginApp, finishApp;
    
    beginApp = omp_get_wtime();
    
    T_real *InputData, *OutputData, *Range;
    int *InputLabels, *OutputLabels;
    
    // Allocate memory
    InputData    = (T_real *) malloc((sizeof(T_real)*NB_POINTS)*NB_DIMS);
    InputLabels  = (int *) malloc(sizeof(int)*NB_POINTS);
    OutputData   = (T_real *) malloc((sizeof(T_real)*NB_POINTS*AMP_FACTOR)*NB_DIMS);
    OutputLabels = (int *) malloc(sizeof(int)*NB_POINTS*AMP_FACTOR);
    Range        = (T_real *) malloc(sizeof(T_real)*NB_DIMS);
    
    // Set the number of OpenMP threads on CPU
    omp_set_num_threads(NB_THREADS);
    
    // Read input data and cluster labels
    begin = omp_get_wtime();
    read_file_real(InputData, NB_POINTS, NB_DIMS, INPUT_DATA, "\t", 0);
    read_file_int(InputLabels, NB_POINTS, 1, INPUT_REF_LABELS, "", 0);
    finish = omp_get_wtime();
    printf("read_file_*:       %f s\n", finish - begin);
    
    // Find the range of data in each dimension
    begin = omp_get_wtime();
    find_range(InputData, Range);
    finish = omp_get_wtime();
    printf("find_range:        %f s\n", finish - begin);
    
    // Amplify data
    begin = omp_get_wtime();
    if (NB_DIMS == 2) {
        amplifier_circle_2d(InputData, InputLabels, Range, SEED_BASE, OutputData, OutputLabels);
    }
    if (NB_DIMS > 2) {
        amplifier_manhattan(InputData, InputLabels, Range, SEED_BASE, OutputData, OutputLabels);
    }
    finish = omp_get_wtime();
    printf("amplifier:         %f s\n", finish - begin);
    
    // Save amplified data and corresponding cluster labels
    begin = omp_get_wtime();
    save_file_real(OutputData, NB_POINTS*AMP_FACTOR, NB_DIMS, OUTPUT_DATA, "\t");
    save_file_int(OutputLabels, NB_POINTS*AMP_FACTOR, 1, OUTPUT_REF_LABELS, "");
    finish = omp_get_wtime();
    printf("save_file_*:       %f s\n", finish - begin);
    
    finishApp = omp_get_wtime();
    
    printf("Total application: %f s\n", finishApp - beginApp);

    return(EXIT_SUCCESS);
}
