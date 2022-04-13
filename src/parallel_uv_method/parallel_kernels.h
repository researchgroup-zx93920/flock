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

#include "parallel_structs.h"

#ifndef KERNELS
#define KERNELS

// ##################################################
// PREPROCESS and POSTPROCESS  >>
// ##################################################

__global__ void createCostMatrix(MatrixCell *d_costMtx, float * d_costs_ptr, int n_supplies, int n_demands);

__global__ void create_initial_tree(flowInformation * d_flows_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr,
    int numSupplies, int numDemands);

__global__ void retrieve_final_tree(flowInformation * d_flows_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr,
        int numSupplies, int numDemands);

// ##################################################
// SOLVING DUAL >>
// ##################################################

/*
APPROACH 1 :
Kernels concerned with solving the UV System using a BFS Traversal Approach
*/

__global__ void copy_row_shadow_prices(Variable * U_vars, float * u_vars_ptr, int numSupplies);

__global__ void copy_col_shadow_prices(Variable * V_vars, float * v_vars_ptr, int numDemands);

__global__ void initialize_U_vars(Variable * U_vars, int numSupplies);

__global__ void initialize_V_vars(Variable * V_vars, int numDemands);

/*
Breadth First Traversal on UV
*/
__global__ void assign_next(int * d_adjMtx_ptr, float * d_costs_ptr, 
    Variable *u_vars, Variable *v_vars, int numSupplies, int numDemands);

/*
APPROACH 2:
Kernels concerned with solving the UV System using a using a matrix solver
*/

// Custom Fill kernel for csr row pointers
__global__ void fill_csr_offset (int * d_csr_offsets, int length);

/*
Create a dense linear system in parallel by looking at current feasible tree 
*/
__global__ void initialize_dense_u_v_system(float * d_A, float * d_b, int * d_adjMtx_ptr, 
    float * d_costs_ptr, int numSupplies, int numDemands);

/*
Create a sparse linear system in parallel by looking at current feasible tree 
*/
__global__ void initialize_sparse_u_v_system(int * d_csr_columns, float * d_b, int * d_adjMtx_ptr, 
    float * d_costs_ptr, int numSupplies, int numDemands);

/*
Load the solution of system to the appropriate place
*/
__global__ void retrieve_uv_solution(float * d_x, float * u_vars_ptr, float * v_vars_ptr, int numSupplies, int numDemands);

// ##################################################
// COMPUTING REDUCED COSTS >>
// ##################################################

/*
Kernel to compute Reduced Costs in the transportation table
*/
__global__ void computeReducedCosts(float * u_vars_ptr, float * v_vars_ptr, float * d_costs_ptr, float * d_reducedCosts_ptr, 
    int numSupplies, int numDemands);

#endif