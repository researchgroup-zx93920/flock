#include<iostream>
#include<map>
#include<vector>
#include<algorithm>

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

struct rowNodes {
    std::vector<int> child;
    bool covered;
};

struct colNodes {
    std::vector<int> parent;
    bool covered;
};

std::ostream& operator << (std::ostream& o, const Variable& x);

__host__ void find_reduced_costs(MatrixCell * costMatrix, flowInformation * flows, float * reduced_costs,
    int matrixSupplies, int matrixDemands);

__global__ void assign_next(flowInformation * flows, MatrixCell * device_costMatrix, Variable * u_vars, 
    Variable * v_vars, int matrixSupplies, int matrixDemands);

__global__ void computeReducedCosts(Variable * u_vars, Variable * v_vars, MatrixCell * device_costMatrix, float * reduced_costs, 
    int matrixSupplies, int matrixDemands);

__host__ void pivot();

#endif