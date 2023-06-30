#ifndef _DATAIO_H
#define _DATAIO_H

void read_file_int (int *array, size_t nbPoints, size_t nbDims, 
                    const char filename[], const char delim[], 
                    bool isTransposed);

void read_file_real (T_real *array, size_t nbPoints, size_t nbDims, 
                     const char filename[], const char delim[], 
                     bool isTransposed);

void save_file_int (int *array, size_t nbPoints, size_t nbDims, 
                    const char filename[], const char delim[]);

void save_file_real (T_real *array, size_t nbPoints, size_t nbDims, 
                     const char filename[], const char delim[]);

#endif