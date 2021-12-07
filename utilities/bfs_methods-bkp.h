#include<iostream>
#include<cstring>

#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

#include "utils.h"

#define blockSize 256

#ifndef BFS
#define BFS


__host__ void find_nw_corner_bfs_seq(int * supplies, int * demands, float * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands);

__host__ void find_vogel_bfs_seq(int * supplies, int * demands, float * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands);

__host__ void find_vogel_bfs_parallel(int * supplies, int * demands, MatrixCell * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands);

// Kernels for Vogel BFS Parallel - 

__device__ int colIndxInFlat(int r, int w, int i);
__device__ int rowIndxInFlat(int c, int w, int i);

__global__ void find_least_two_with_indexes(float * flatMatrix2D, int orientation, int vector_index, 
        int width, vogelDifference * diff);

#endif