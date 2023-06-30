#include <stdio.h>   // Library variables (e.g. FILE), library macros (e.g. stderr, stdin, stdout), library functions (e.g. fopen, fclose, fread, fwrite, fprintf, printf, fscanf, fgets, fflush)
#include <stdlib.h>  // Library variables (e.g. size_t), library macros (e.g. EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX), library functions (e.g. atof, atoi, malloc, free, exit, rand)
#include <string.h>  // Library functions (e.g. strcat, strcmp, strcpy, strtok)

#include "../../include/config.h"
#include "../../include/utilities/dataIO.h"

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


