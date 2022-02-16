#include <iostream>

#include "../structs.h"
#include "./parallel_structs.h"

__global__ void assign_next(flowInformation *flows, MatrixCell *device_costMatrix, Variable *u_vars,
                            Variable *v_vars, int matrixSupplies, int matrixDemands);