#include <iostream>

#include "../structs.h"
#include "./parallel_structs.h"

__global__ void assign_next(float * d_adjMtx_ptr, float * d_costs_ptr, 
                            Variable * u_vars, Variable * v_vars, int numSupplies, int numDemands);