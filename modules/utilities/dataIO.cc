#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <string.h>  // Library functions (e.g. strcat, strcmp, strcpy, strtok)

#include "../../include/config.h"
#include "../../include/utilities/dataIO.h"


// Some of the following code is derived from MC-K-means/util/dataIo.cpp 
// at https://gitlab.cs.univie.ac.at/martinp16cs/MC-K-means/-/tree/master/

// Read a file of integers (derived from MC-K-means)
void read_file_int (int *array, const size_t N, const size_t D, 
                    const char filename[], const char delim[], 
                    bool isBinary, bool isTransposed)
{
    // Declaration
    FILE *fp;  // File pointer
    size_t counts = 0;
    size_t i = 0, j = 0, total = 0;
    char line[MAX_LINE_LENGTH];
    char *token = NULL;

    // Open file
    fp = fopen(filename, "r");  // Open the file in "read-only" mode
    if (fp == NULL) {           // Check if the file has been successfully opened
        fprintf(stderr, "    Fail to open the file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read file
    if (isBinary) {
        // Read a binary file (everything at once)
        counts = fread(array, sizeof(int) * N * D, 1, fp);
        if (counts == 0) {
            fprintf(stderr, "    Fail to read the binary file: %s\n", filename);
            exit(EXIT_FAILURE);
        }
    } else {
        // read a text file line by line
        // format: there are D values each line. Each value is separated by a delimiter (e.g. ",", " ", "\t").
        // notice MAX_LINE_LENGTH = 2049
        i = 0;
        while ( fgets(line, MAX_LINE_LENGTH, fp) != NULL && i < N ) {
            if (line[0] != '%'){ // ignore '%' comment char
                // The C library function char *strtok(char *str, const char *delim) 
                // breaks string str into a series of tokens using the delimiter delim.
                // Get the first token
                token = strtok(line, delim); 
                
                // Walk through other tokens
                j = 0;
                if (isTransposed) {
                    while (token != NULL && j < D) {
                        array[j*N + i] = atoi(token);
                        token = strtok(NULL, delim);
                        j++;
                    }
                } else {
                    while (token != NULL && j < D) {
                        array[i*D + j] = atoi(token);
                        token = strtok(NULL, delim);
                        j++;
                    }
                }
                i++;
                total += j;
            }
        }
        
        if (total != N*D) {
            fprintf(stderr, "    Fail to read the text file: %s\n", filename);
            exit(EXIT_FAILURE);
        }
    }

    // Close file
    fclose(fp);
}


// Read a file of real numbers (derived from MC-K-means)
void read_file_real (T_real *array, const size_t N, const size_t D, 
                     const char filename[], const char delim[], 
                     bool isBinary, bool isTransposed)
{
    // Declaration
    FILE *fp;  // File pointer
    size_t counts = 0;
    size_t i = 0, j = 0, total = 0;
    char line[MAX_LINE_LENGTH];
    char *token = NULL;

    // Open file
    fp = fopen(filename, "r");  // Open the file in "read-only" mode
    if (fp == NULL) {           // Check if the file has been successfully opened
        fprintf(stderr, "    Fail to open the file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read file
    if (isBinary) {
        // Read a binary file (everything at once)
        counts = fread(array, sizeof(T_real) * N * D, 1, fp);
        if (counts == 0) {
            fprintf(stderr, "    Fail to read the binary file: %s\n", filename);
            exit(EXIT_FAILURE);
        }
    } else {
        // read a text file line by line
        // format: there are D values each line. Each value is separated by a delimiter (e.g. ",", " ", "\t").
        // notice MAX_LINE_LENGTH = 2049
        i = 0;
        while ( fgets(line, MAX_LINE_LENGTH, fp) != NULL && i < N ) {
            if (line[0] != '%'){ // ignore '%' comment char
                // The C library function char *strtok(char *str, const char *delim) 
                // breaks string str into a series of tokens using the delimiter delim.
                // Get the first token
                token = strtok(line, delim); 
                
                // Walk through other tokens
                j = 0;
                if (isTransposed) {
                    while (token != NULL && j < D) {
                        array[j*N + i] = atof(token);
                        token = strtok(NULL, delim);
                        j++;
                    }
                } else {
                    while (token != NULL && j < D) {
                        array[i*D + j] = atof(token);
                        token = strtok(NULL, delim);
                        j++;
                    }
                }
                i++;
                total += j;
            }
        }
        
        if (total != N*D) {
            fprintf(stderr, "    Fail to read the text file: %s\n", filename);
            exit(EXIT_FAILURE);
        }
    }

    // Close file
    fclose(fp);
}


// Save a file of integers (derived from MC-K-means)
void save_file_int (int *array, const size_t N, const size_t D, 
                    const char filename[], const char delim[], bool isBinary)
{
    // Declaration
    FILE *fp;  // File pointer
    
    // Open file
    fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("    Fail to open the file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    // Write into file
    if (isBinary) {
        // Write a binary file
        size_t counts = 0;
        counts = fwrite(array, sizeof(int) * N * D, 1, fp);
        if (counts == 0) {
            fprintf(stderr, "    Fail to write the binary file: %s\n", filename);
            exit(EXIT_FAILURE);
        }
    } else {
        // Write a text file element by element
        /*
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < D; j++) {
                fprintf(fp, "%d%s", array[i*D + j], delim);
            }
            fprintf(fp, "\n");
        }*/
        
        // Write a text file line by line
        char line[MAX_LINE_LENGTH];
        char strInt[50];
        for (size_t i = 0; i < N; i++) {
            strcpy(line, "");
            for (size_t j = 0; j < D; j++) {
                // The C library function char *strcpy(char *dest, const char *src) 
                // copies the string pointed to, by src to dest.
                strcpy(strInt, "");
                
                // The C library function int sprintf(char *str, const char *format, ...)
                // sends formatted output to a string pointed to, by str.
                sprintf(strInt, "%d%s", array[i*D + j], delim);
                
                // The C library function char *strcat(char *dest, const char *src) 
                // appends the string pointed to by src 
                // to the end of the string pointed to by dest.
                strcat(line, strInt);
            }
            fprintf(fp, "%s\n", line);

        }
    }

    // Close file
    fclose(fp);
}


// Save a file of real numbers (derived from MC-K-means)
void save_file_real (T_real *array, const size_t N, const size_t D, 
                     const char filename[], const char delim[], bool isBinary)
{
    // Declaration
    FILE *fp;  // File pointer

    fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("    Fail to open the file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    // Write into file
    if (isBinary) {
        // Write a binary file
        size_t counts = 0;
        counts = fwrite(array, sizeof(T_real) * N * D, 1, fp);
        if (counts == 0) {
            fprintf(stderr, "    Fail to write the binary file: %s\n", filename);
            exit(EXIT_FAILURE);
        }
    } else {
        // Write a text file
        /*
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < D; j++) {
                fprintf(fp, T_REAL_PRINT"%s", array[i*D + j], delim);
            }
            fprintf(fp, "\n");
        }*/
        
        // Write a text file line by line
        char line[MAX_LINE_LENGTH];
        char strReal[50];
        for (size_t i = 0; i < N; i++) {
            strcpy(line, "");              // Reset line to nothing
            for (size_t j = 0; j < D; j++) {
                // The C library function char *strcpy(char *dest, const char *src) 
                // copies the string pointed to, by src to dest.
                strcpy(strReal, "");
                
                // The C library function int sprintf(char *str, const char *format, ...)
                // sends formatted output to a string pointed to, by str.
                sprintf(strReal, T_REAL_PRINT"%s", array[i*D + j], delim);
                
                // The C library function char *strcat(char *dest, const char *src) 
                // appends the string pointed to by src 
                // to the end of the string pointed to by dest.
                strcat(line, strReal);
            }
            fprintf(fp, "%s\n", line);

        }
    }
    
    // Close file
    fclose(fp);
}


