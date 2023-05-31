#ifndef _GET_EDGE_LIST_FROM_CSR_H
#define _GET_EDGE_LIST_FROM_CSR_H

void get_edge_list_from_csr (int nbPoints, int nnz,                                  // input
                             int *GPU_csrRow, int *GPU_csrCol, T_real *GPU_csrVal);  // input

#endif