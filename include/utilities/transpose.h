#ifndef _TRANSPOSE_H
#define _TRANSPOSE_H

void transpose_data (int nbPoints, int nbDims,  // input
                     T_real *GPU_data,          // input
                     T_real *GPU_dataT);        // output

#endif