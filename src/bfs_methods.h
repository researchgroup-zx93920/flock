#include<iostream>
#include<cstring>

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/transform.h>
#include<thrust/sort.h>
#include<thrust/extrema.h>
#include<thrust/execution_policy.h>

#include "utils.h"

#define blockSize 512

#ifndef BFS
#define BFS

__host__ void find_nw_corner_bfs_seq(int * supplies, int * demands, MatrixCell * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands);

__host__ void find_vogel_bfs_parallel(int * supplies, int * demands, MatrixCell * costMatrix, 
        int * flows, int matrixSupplies, int matrixDemands); 

#endif