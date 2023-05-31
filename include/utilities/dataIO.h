#ifndef _DATAIO_H
#define _DATAIO_H

void read_file_int (int *array, const size_t N, const size_t D, 
                    const char filename[], const char delim[], 
                    bool isBinary, bool isTransposed);

void read_file_real (T_real *array, const size_t N, const size_t D, 
                     const char filename[], const char delim[], 
                     bool isBinary, bool isTransposed);

void save_file_int (int *array, const size_t N, const size_t D, 
                    const char filename[], const char delim[], bool isBinary);

void save_file_real (T_real *array, const size_t N, const size_t D, 
                     const char filename[], const char delim[], bool isBinary);

#endif