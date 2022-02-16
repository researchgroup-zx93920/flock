#include "DUAL_tree.h"

__global__ void assign_next(flowInformation * flows, MatrixCell * device_costMatrix, Variable * u_vars, 
    Variable * v_vars, int matrixSupplies, int matrixDemands) {
    
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    // __shared__ >> Can solve the equations locally before taking them to global

    if (indx < matrixSupplies + matrixDemands - 1) {
        flowInformation eqn = flows[indx];
        if (u_vars[eqn.source].assigned && (!v_vars[eqn.destination].assigned)) {
            // In this case >> v_j = c_ij - u_i
            Variable var;
            var.assigned = true;
            var.value = device_costMatrix[eqn.source*matrixDemands+eqn.destination].cost - u_vars[eqn.source].value;
            v_vars[eqn.destination] = var;
        }
        else if ((!u_vars[eqn.source].assigned) && v_vars[eqn.destination].assigned) {
            // In this case >> u_j = c_ij - v_j
            Variable var;
            var.assigned = true;
            var.value = device_costMatrix[eqn.source*matrixDemands+eqn.destination].cost -  v_vars[eqn.destination].value;
            u_vars[eqn.source] = var;
        }
    }
}