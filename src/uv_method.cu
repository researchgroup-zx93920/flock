#include "uv_method.h"

std::ostream& operator << (std::ostream& o, const Variable& x) {
    o << x.value;
    return o;
}

__global__ void computeReducedCosts(Variable * u_vars, Variable * v_vars, MatrixCell * device_costMatrix, float * reduced_costs, 
    int matrixSupplies, int matrixDemands) {

        int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

        if (row_indx < matrixDemands && col_indx < matrixSupplies) {
            // r = u_i + v_j - C_ij;
            float r;
            r = u_vars[col_indx].value + v_vars[row_indx].value - device_costMatrix[row_indx*matrixDemands+col_indx].cost;
            reduced_costs[row_indx*matrixDemands+col_indx] = r;
        }

}

__global__ void assign_next(flowInformation * flows, MatrixCell * device_costMatrix, Variable * u_vars, 
    Variable * v_vars, int matrixSupplies, int matrixDemands) {
    
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    // __shared__ >> Can solve the equations locally before taking them to global

    if (indx < matrixSupplies + matrixDemands - 1) {
        flowInformation eqn = flows[indx];
        if (u_vars[eqn.row].assigned && !v_vars[eqn.col].assigned) {
            // v_j = c_ij - u_i
            Variable var;
            var.assigned = true;
            var.value = device_costMatrix[eqn.row*matrixDemands+eqn.col].cost - u_vars[eqn.row].value;
            v_vars[eqn.col] = var;
        }
        else if (!u_vars[eqn.row].assigned && v_vars[eqn.col].assigned) {
            // u_j = c_ij - v_j
            Variable var;
            var.assigned = true;
            var.value = device_costMatrix[eqn.row*matrixDemands+eqn.col].cost -  v_vars[eqn.col].value;
            u_vars[eqn.row] = var;
        }
    } 
}


__host__ void find_reduced_costs(MatrixCell * costMatrix, flowInformation * flows, float * reduced_costs,
    int matrixSupplies, int matrixDemands){
        
        std::cout<<"Determining Dual Costs"<<std::endl;
        // Start U-V vectors
        thrust::device_vector<Variable> U_vars(matrixSupplies);
        Variable * u_vars_ptr = thrust::raw_pointer_cast(U_vars.data());
        thrust::device_vector<Variable> V_vars(matrixDemands);
        Variable * v_vars_ptr = thrust::raw_pointer_cast(V_vars.data());
        thrust::device_vector<flowInformation> device_flows(flows, flows + (matrixSupplies+matrixDemands-1));
        flowInformation * device_flows_ptr = thrust::raw_pointer_cast(device_flows.data());
        thrust::device_vector<MatrixCell> device_costMatrix(costMatrix, costMatrix + matrixSupplies*matrixDemands);
        MatrixCell * device_costMatrix_ptr = thrust::raw_pointer_cast(device_costMatrix.data());

        // Make any one as 0
        Variable default_assign; 
        default_assign.value = 0.0;
        default_assign.assigned = true;
        U_vars[0] = default_assign;

        std::cout<<"Determining Solving the UV System"<<std::endl;
        // Start solving the system of eqn's (complementary slackness conditions)
        dim3 dimGrid(ceil(1.0*matrixSupplies+matrixDemands-1/blockSize),1,1);
        dim3 dimBlock(blockSize,1,1);
        
        // Solve the system in at most m+n-1 iterations
        
        // assigned cells
        //  (i - j) -> m + n

        // m + n -1 ->>

        // C_ij = u_i + v_j;
        // C_ij = u_i + v_j;
        // C_ij = u_i + v_j;

        for (int i=0; i < matrixSupplies+matrixDemands-1;i++ ){
            assign_next <<< dimGrid, dimBlock >>> (device_flows_ptr, device_costMatrix_ptr, u_vars_ptr, v_vars_ptr, matrixSupplies, matrixDemands);
        }

        // Questions ::
        // Diagonal zero - corner case for U-V method
        dim3 dimGrid2(matrixDemands, matrixSupplies, 1);
        dim3 dimBlock2(blockSize, blockSize, 1);
        computeReducedCosts<<< dimGrid2, dimBlock2 >>>(u_vars_ptr, v_vars_ptr, device_costMatrix_ptr, reduced_costs, matrixSupplies, matrixDemands);
        cudaDeviceSynchronize();
    }