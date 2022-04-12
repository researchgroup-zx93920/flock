/*
PARALLEL KERNELS are classified as simple and special

(^_^) All simple kernels are here 

Some of the kernels in parallel simplex are specialized and slightly complicated,
they are stored in a separate module for cleanliness, follow the
usage in uv_model_parallel.cu (aka parent) file. 

FYI Directly reviewing this file wouldn't make sense. If you see a kernel in parent
You'll either find it here or there's a comment that would take you to the 
appropriate place
*/

#include <iostream>

#include "../structs.h"
#include "./parallel_structs.h"

// ##################################################
// PREPROCESS and POSTPROCESS  >>
// ##################################################

/*
Kernel to convert float cost matrix to the MatrixCell objects
*/
__global__ void createCostMatrix(MatrixCell *d_costMtx, float * d_costs_ptr, int n_supplies, int n_demands)
{

    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;

    if (s < n_supplies && d < n_demands)
    {
        int id = s * n_demands + d;
        MatrixCell _c = {.row = s, .col = d, .cost = d_costs_ptr[id]};
        d_costMtx[id] = _c;
    }
}

/*
Generate a tree on the global memory using the initial set of feasible flows
*/
__global__ void create_initial_tree(flowInformation * d_flows_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr,
    int numSupplies, int numDemands)
{
    
    int V = numSupplies+numDemands;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    if (gid < V - 1) {
    
        flowInformation _this_flow = d_flows_ptr[gid];
        int row = _this_flow.source;
        int column =  _this_flow.destination;
        int idx = TREE_LOOKUP(row, numSupplies+column, V); // Index in adjacency matrix
        float _qty = 1.0*_this_flow.qty;
        if (_qty==0){
            // Handling degeneracy - Flow purturbation
            _qty=epsilon;
        }
        d_flowMtx_ptr[gid] = _qty;
        d_adjMtx_ptr[idx] = gid+1;
    }
}

/*
Reverse operation of generating a tree from the feasible flows - unordered allocation
*/
__global__ void retrieve_final_tree(flowInformation * d_flows_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr,
        int numSupplies, int numDemands) 
{

    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies+numDemands;
    
    // Upper triangle scope of adj matrix
    if (col_indx < V && col_indx >= numSupplies && row_indx < numSupplies) {
        
        // Check if this is a flow edge - 
        int gid = TREE_LOOKUP(row_indx, col_indx, V);
        int flow_id = d_adjMtx_ptr[gid];
        if (flow_id > 0) {

            flowInformation _this_flow;
            _this_flow.qty = round(d_flowMtx_ptr[flow_id - 1]);
            _this_flow.source = row_indx;
            _this_flow.destination = col_indx - numSupplies;
            d_flows_ptr[flow_id - 1] = _this_flow;

        }
    }
}

// ##################################################
// SOLVING DUAL >>
// ##################################################

/*
APPROACH 1 :
Kernels concerned with solving the UV System using a BFS Traversal Approach
*/

__global__ void copy_row_shadow_prices(Variable * U_vars, float * u_vars_ptr, int numSupplies) 
{    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < numSupplies) {
        u_vars_ptr[gid] = U_vars[gid].value;
    }
}

__global__ void copy_col_shadow_prices(Variable * V_vars, float * v_vars_ptr, int numDemands) 
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < numDemands) {
        v_vars_ptr[gid] = V_vars[gid].value;
    }
}

__global__ void initialize_U_vars(Variable * U_vars, int numSupplies) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    Variable default_var;
    if (gid < numSupplies) {
        U_vars[gid] = default_var;
    }
}

__global__ void initialize_V_vars(Variable * V_vars, int numDemands) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    Variable default_var;
    if (gid < numDemands) {
        V_vars[gid] = default_var;
    }
}

/*
Breadth First Traversal on UV
*/
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

/*
APPROACH 2:
Kernels concerned with solving the UV System using a using a matrix solver
*/

// Custom Fill kernel for csr row pointers
__global__ void fill_csr_offset (int * d_csr_offsets, int length) {
        
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < length) {
                if (idx == 0) {
                        d_csr_offsets[idx] = 0;
                }
                else {
                        d_csr_offsets[idx] = 2*idx - 1; 
                }
        }
}

/*
Create a dense linear system in parallel by looking at current feasible tree 
*/
__global__ void initialize_dense_u_v_system(float * d_A, float * d_b, int * d_adjMtx_ptr, 
    float * d_costs_ptr, int numSupplies, int numDemands) {
        
    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies + numDemands;

    if (row_indx < numSupplies && col_indx < numDemands) {
        int indx = TREE_LOOKUP(row_indx, col_indx + numSupplies, V); // Adjusted destination vertex ID
        int flow_indx = d_adjMtx_ptr[indx];
        if (flow_indx > 0) {
            // This is a flow - flow_indx = row_number, u = row_number, v = col_number
            d_A[flow_indx * V + row_indx] = 1;
            d_A[flow_indx * V + numSupplies + col_indx] = 1;
            d_b[flow_indx] = d_costs_ptr[row_indx*numDemands + col_indx];
        }
    }
}

/*
Create a sparse linear system in parallel by looking at current feasible tree 
*/
__global__ void initialize_sparse_u_v_system(int * d_csr_columns, float * d_b, int * d_adjMtx_ptr, 
    float * d_costs_ptr, int numSupplies, int numDemands) {
        
    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies + numDemands;

    if (row_indx < numSupplies && col_indx < numDemands) {
        int indx = TREE_LOOKUP(row_indx, col_indx + numSupplies, V); // Adjusted destination vertex ID
        int flow_indx = d_adjMtx_ptr[indx];
        if (flow_indx > 0) {
            // This is a flow - flow_indx = row_number, u = row_number, v = col_number
            d_csr_columns[2*flow_indx-1] = row_indx;
            d_csr_columns[2*flow_indx] = col_indx;
            d_b[flow_indx] = d_costs_ptr[row_indx*numDemands + col_indx];
        }
    }
}

/*
Load the solution of system to the appropriate place
*/
__global__ void retrieve_uv_solution(float * d_x, float * u_vars_ptr, float * v_vars_ptr, int numSupplies, int numDemands) 
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int V = numSupplies + numDemands;
    if (gid < V) {
        if (gid < numSupplies) {
            u_vars_ptr[gid] = d_x[gid];
        } 
        else {
            v_vars_ptr[gid - numSupplies] = d_x[gid];
        }
    }
}

// ##################################################
// COMPUTING REDUCED COSTS >>
// ##################################################


/*
Kernel to compute Reduced Costs in the transportation table
*/
__global__ void computeReducedCosts(float * u_vars_ptr, float * v_vars_ptr, float * d_costs_ptr, float * d_reducedCosts_ptr, 
    int numSupplies, int numDemands)
{

        int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

        if (row_indx < numSupplies && col_indx < numDemands) {
            // r =  C_ij - (u_i + v_j);
            float r = d_costs_ptr[row_indx*numDemands+col_indx] - u_vars_ptr[row_indx] - v_vars_ptr[col_indx];
            d_reducedCosts_ptr[row_indx*numDemands+col_indx] = r;
        }
}