#include "uv_method.h"

__global__ assign_next(flowInformation * flows, Variable * u_vars, Variable * v_vars, int matrixSupplies, int matrixDemands) {
    
    indx = blockIdx.x*blockDim.x + threadIdx.x;
    // __shared__ >> Can solve the equations locally before taking them to global

    if indx < (matrixSupplies + matrixDemands - 1) {
        flowInformation eqn = flowInformation[indx];
        if (u_vars(eqn.row) | u_vars(eqn.col)) {

        } 
    } 
}


__host__ void find_reduced_costs(MatrixCell * costMatrix, flowInformation * flows, float * reduced_costs,
    int matrixSupplies, int matrixDemands){
        
        // Start U-V vectors
        thrust::device_vector<Variable> U_vars(matrixSupplies);
        thrust::device_vector<Variable> V_vars(matrixDemands);
        thrust::device_vector<flowInformation> device_flows(flows, flows + (matrixSupplies+matrixDemands-1));
        thrust::device_vector<MatrixCell> device_costMatrix(costMatrix, costMatrix + matrixSupplies*matrixDemands);

        // Make any one as 0
        Variable default_assign = 0;
        U_vars[0] = default_assign;

        // Start solving the system of eqn's
        thrust::transform




    }