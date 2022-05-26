#include "parallel_structs.h"

__host__ void find_nw_corner_bfs_seq(int *supplies, int *demands, MatrixCell *costMatrix, flowInformation *flows,
    int matrixSupplies, int matrixDemands);