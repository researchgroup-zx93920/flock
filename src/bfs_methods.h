#include<iostream>
#include<cstring>

#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include "utils.h"

#define blockSize 256

#ifndef BFS
#define BFS

struct vogelDifference {
        double diff;
        int indx, ileast_1, ileast_2;
        // idx it itselves index in diff array
        // ileast_1 and ileast2 are indexes of min-2 values
        // least_1,least_2,
};

__host__ void find_nw_corner_bfs_seq(int * supplies, int * demands, double * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands);

__host__ void find_vogel_bfs_seq(int * supplies, int * demands, double * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands);

__host__ void find_vogel_bfs_parallel(int * supplies, int * demands, double * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands);

// Kernels for Vogel BFS Parallel - 

__device__ int colIndxInFlat(int r, int w, int i);
__device__ int rowIndxInFlat(int c, int w, int i);

__global__ void find_least_two_with_indexes(double * flatMatrix2D, int orientation, int vector_index, 
        int width, vogelDifference * diff);

#endif