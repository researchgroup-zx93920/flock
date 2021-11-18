#include<iostream>
#include "utils.h"

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/transform.h>
#include<thrust/sort.h>
#include<thrust/extrema.h>
#include<thrust/execution_policy.h>

#define blockSize 512

#ifndef UV
#define UV

struct Variable {
    float value;
    bool assigned = false;

    __host__ __device__ Variable& operator=(const float& x)
    {
        value=x;
        assigned=true;
        return *this;
    }
};

std::ostream& operator << (std::ostream& o, const Variable& x);


__host__ void find_current_uv();

__host__ void find_reduced_costs(MatrixCell * costMatrix, flowInformation * flows, float * reduced_costs,
    int matrixSupplies, int matrixDemands);

__host__ void pivot();

#endif