#include "DUAL_tree.h"

__global__ void assign_next(int * d_adjMtx_ptr, float * d_costs_ptr, 
    Variable *u_vars, Variable *v_vars, int numSupplies, int numDemands) {
    
    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies + numDemands;
    // int adjMtx_row = row_indx; // Equal to supply
    // int adjMtx_col = col_indx + numSupplies; // Adjusted vertex ID

    // Within the scope of the adj matrix
    if (row_indx < numSupplies && col_indx < numDemands) {
        // Check if these are adjacent - (checks in upper triangular matrix, because row < adj-col-index)
        int indx = TREE_LOOKUP(row_indx, col_indx + numSupplies, V); // Adjusted destination vertex ID
        if (d_adjMtx_ptr[indx] > 0) {
            // Check if any of the u or v has not been assigned and adjacent is assigned - then assign it
            if (u_vars[row_indx].assigned && (!v_vars[col_indx].assigned)) {
                // In this case >> v_j = c_ij - u_i
                Variable var;
                var.assigned = true;
                var.value = d_costs_ptr[row_indx*numDemands+col_indx] - u_vars[row_indx].value;
                v_vars[col_indx] = var;
            }
            else if ((!u_vars[row_indx].assigned) && v_vars[col_indx].assigned) {
                // In this case >> u_j = c_ij - v_j
                Variable var;
                var.assigned = true;
                var.value = d_costs_ptr[row_indx*numDemands+col_indx] -  v_vars[col_indx].value;
                u_vars[row_indx] = var;
            }
        }
    }
}