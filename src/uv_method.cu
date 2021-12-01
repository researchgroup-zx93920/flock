#include "uv_method.h"

std::ostream& operator << (std::ostream& o, const Variable& x) {
    o << x.value;
    return o;
}


__global__ void computeReducedCosts(Variable * u_vars, Variable * v_vars, MatrixCell * device_costMatrix, float * device_reducedCosts_ptr, 
    int matrixSupplies, int matrixDemands) {

        int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

        if (row_indx < matrixSupplies && col_indx < matrixDemands) {
            // r =  C_ij - (u_i + v_j);
            float r = device_costMatrix[row_indx*matrixDemands+col_indx].cost - u_vars[row_indx].value - v_vars[col_indx].value;
            device_reducedCosts_ptr[row_indx*matrixDemands+col_indx] = r;
        }
}

__global__ void assign_next(flowInformation * flows, MatrixCell * device_costMatrix, Variable * u_vars, 
    Variable * v_vars, int matrixSupplies, int matrixDemands) {
    
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    // __shared__ >> Can solve the equations locally before taking them to global

    if (indx < matrixSupplies + matrixDemands - 1) {
        flowInformation eqn = flows[indx];
        if (u_vars[eqn.row].assigned && !v_vars[eqn.col].assigned) {
            // In this case >> v_j = c_ij - u_i
            Variable var;
            var.assigned = true;
            var.value = device_costMatrix[eqn.row*matrixDemands+eqn.col].cost - u_vars[eqn.row].value;
            v_vars[eqn.col] = var;
        }
        else if (!u_vars[eqn.row].assigned && v_vars[eqn.col].assigned) {
            // In this case >> u_j = c_ij - v_j
            Variable var;
            var.assigned = true;
            var.value = device_costMatrix[eqn.row*matrixDemands+eqn.col].cost -  v_vars[eqn.col].value;
            u_vars[eqn.row] = var;
        }
    }
}

__host__ void find_reduced_costs(MatrixCell * costMatrix, flowInformation * flows, float * reduced_costs,
    int matrixSupplies, int matrixDemands){
        
        std::cout<<"TESTING CURRENT BFS: Determining Dual Costs"<<std::endl;
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

        std::cout<<"TESTING CURRENT BFS: Solving the UV System"<<std::endl;
        
        // Start solving the system of eqn's (complementary slackness conditions)
        // Solve the system in at most m+n-1 iterations        
        // assigned cells -> m + n, and m+n-1 linearly independent equations to solve
        // For each of the m + n - 1 assignements in the assignment tree ->>
        // C_ij = u_i + v_j

        dim3 dimGrid(ceil(1.0*matrixSupplies+matrixDemands-1/blockSize),1,1);
        dim3 dimBlock(blockSize,1,1);

        // Solve the system of linear equations in the following kernel >>
        for (int i=0; i < matrixSupplies+matrixDemands-1;i++ ) {
            assign_next <<< dimGrid, dimBlock >>> (device_flows_ptr, device_costMatrix_ptr, u_vars_ptr, v_vars_ptr, matrixSupplies, matrixDemands);
        }

        // Questions ::
        // 1. Diagonal zero - corner case for U-V method (Resolution: Recall discussion with Hem, degenerate case)
        std::cout<<"TESTING CURRENT BFS: Computing Reduced Costs"<<std::endl;
        dim3 dimGrid2(ceil(1.0*matrixDemands/32), ceil(1.0*matrixSupplies/32), 1);
        dim3 dimBlock2(32, 32, 1); // Refine this based on device query

        thrust::device_vector<float> device_reducedCosts(matrixSupplies*matrixDemands);
        float * device_reducedCosts_ptr = thrust::raw_pointer_cast(device_reducedCosts.data());
        computeReducedCosts<<< dimGrid2, dimBlock2 >>>(u_vars_ptr, v_vars_ptr, device_costMatrix_ptr, device_reducedCosts_ptr, matrixSupplies, matrixDemands);
        cudaDeviceSynchronize();
        
        cudaMemcpy(reduced_costs, device_reducedCosts_ptr, matrixDemands*matrixSupplies*sizeof(float), cudaMemcpyDeviceToHost);
        

        // Pivoting - sequencial >>
        int pivot_row = 0;
        int pivot_col = 0;
        float most_negative = 0.0;
        // Look for most negative reduced cost >> 
        // In parallel - just look for a negative reduced cost
        for (int i = 0; i < matrixSupplies*matrixDemands; i++) {
            if (reduced_costs[i] < 0 && reduced_costs[i] < most_negative) {
                most_negative = reduced_costs[i];
                pivot_col = i%matrixDemands;
                pivot_row = (i - pivot_col)/matrixDemands;
            }
        }

        // Checkpoint >>
        // pivot_row, pivot_col
        pivot_row = 0;
        pivot_col = 4;

        // Now create a bipratite graph
        std::map<int, rowNodes> row_map; // left side 
        std::map<int, colNodes> col_map; // right side 

        // Do some preprocess >>
        for (int i=0; i< matrixDemands+matrixSupplies-1; i++) {
            flowInformation f = flows[i];
            row_map[f.row].child.push_back(f.col);
            col_map[f.col].parent.push_back(f.row);
        }

        int side, current_idx;
        // current_idx = on the particular side which index are we standing on
        // side = 0 means currently standing on left side - look for children
        // side = 1 means currently standing on right side - look for parents
        std::vector<int> alternating_path;
        
        // Entering Arc >>
        alternating_path.push_back(pivot_row);
        alternating_path.push_back(pivot_col);
        
        std::vector<int> rowCandidates;
        std::vector<int> colCandidates;

        side = 1;
        current_idx = pivot_col;
        rowCandidates.insert(rowCandidates.end(), col_map[current_idx].parent.begin(), col_map[current_idx].parent.end());
        int counter = 0;
        // Following is a depth first search >>
        while (counter < 10){
            counter++;

            if (side==1) {
                // Being on the column side >>
                // look at the available row candidates >> 
                // Make sure that current_idx col is one of the child of new_row and not covered (assuming valid)
                int new_row = rowCandidates.back();
                rowCandidates.pop_back();
                std::cout<<"PRINT NEW ROW : "<<new_row<<std::endl;

                if (!row_map[new_row].covered) {

                    // Make sure that current_idx row is one of the parent of new_col and not covered:
                    std::vector<int> lookup = row_map[new_row].child;
                    if (std::find(lookup.begin(), lookup.end(), current_idx) != lookup.end()){
                        alternating_path.push_back(new_row);
                        // mark this col as covered
                        col_map[current_idx].covered = true;
                        // jump current idx to row further update column candidates
                        
                        // stopping criteria >>
                        std::vector<int> lookup = col_map[current_idx].parent;
                        if (std::find(lookup.begin(), lookup.end(), pivot_row) != lookup.end()){
                            std::cout<<"Found the loop!"<<std::endl;
                            break;
                            }
                        
                        current_idx = new_row;
                        colCandidates.insert(colCandidates.end(), row_map[current_idx].child.begin(), row_map[current_idx].child.end());
                        side = 0;
                        }
                    else {
                        // Go one step back -
                        // The column has been removed >> now you're on column side
                        col_map[current_idx].covered = false;
                        alternating_path.pop_back();
                        // Update the path, update the side and move on!
                        side = 1;
                        current_idx = alternating_path.back();
                    }
                }
            }

            else {
                // Being on the row side >>
                // look at the available col candidates >> 
                int new_col = colCandidates.back();
                colCandidates.pop_back();
                
                // Make sure if this new col is not covered >>
                if (!col_map[new_col].covered){
                    // Make sure that current_idx row is one of the parent of new_col and not covered:
                    std::vector<int> lookup = col_map[new_col].parent;
                    if (std::find(lookup.begin(), lookup.end(), current_idx) != lookup.end()){
                        alternating_path.push_back(new_col);
                        // mark this col as covered
                        row_map[current_idx].covered = true;
                        // jump current idx to col  further update row candidates
                        current_idx = new_col;
                        rowCandidates.insert(rowCandidates.end(), col_map[current_idx].parent.begin(), col_map[current_idx].parent.end());
                        side = 1;
                    }
                    else {
                        // Go one step back -
                        // The row has been removed >> now you're on row side
                        row_map[current_idx].covered = false;
                        alternating_path.pop_back();
                        // Update the path, update the side and move on!
                        side = 0;
                        current_idx = alternating_path.back();
                    }   
                }
            }

            std::cout<<"Iteration : "<<side<<std::endl;
            for (int i =0; i<alternating_path.size(); i++) {
                std::cout<<"Alternating path : "<< alternating_path[i]<<std::endl;
            }
        }

    }
