/* 
NOTE - user may not understand this file directly 
Follow the implementation of uv_method_parallel class methods
and all of this will suddenly make sense
*/

#include <iostream>

#include "parallel_structs.h"
#include "parallel_kernels.h"

// ALLOCATE and DE-ALLOCATE RESOURCES
__host__ void initialize_device_DUAL(float ** u_vars_ptr, float ** v_vars_ptr, 
        Variable ** U_vars, Variable ** V_vars,
        int ** length, int ** start, int ** Ea, bool ** Fa, bool ** Xa, float ** variables, 
        float ** d_csr_values, int ** d_csr_columns, int ** d_csr_offsets,
        float ** d_A, float ** d_b, float ** d_x, int64_t &nnz, 
        int ** h_length, int ** h_start, int ** h_Ea, bool ** h_visited, float ** h_variables,
        int numSupplies, int numDemands);

__host__ void terminate_device_DUAL(float * u_vars_ptr, float * v_vars_ptr, 
        Variable * U_vars, Variable * V_vars, 
        int * length, int * start, int * Ea, bool * Fa, bool * Xa, float * variables,
        float * d_csr_values, int * d_csr_columns, int * d_csr_offsets,
        float * d_A, float * d_b, float * d_x, 
        int * h_length, int * h_start, int * h_Ea, bool * h_visited, float * h_variables);


// BREADTH FIRST SEARCH
__host__ void find_dual_using_tree(float * u_vars_ptr, float * v_vars_ptr, 
        int * d_adjMtx_ptr, float * d_costs_ptr, Variable * U_vars, Variable * V_vars, 
        int numSupplies, int numDemands);

__host__ void find_dual_using_bfs(float * u_vars_ptr, float * v_vars_ptr,
        int * length, int * start, int * Ea, bool * Fa, bool * Xa, float * variables,
        int * d_adjMtx_ptr, float * d_costs_ptr, int numSupplies, int numDemands);

__host__ void find_dual_using_seq_bfs(float * u_vars_ptr, float * v_vars_ptr, 
        int * length, int * start, int * Ea, int * d_adjMtx_ptr, float * h_costs_ptr, 
        int * h_length, int * h_start, int * h_Ea, bool * h_visited, float * h_variables,
        int numSupplies, int numDemands);

// SOLVE LINEAR SYSTEM
__host__ void find_dual_using_sparse_solver(float * u_vars_ptr, float * v_vars_ptr, 
        float * d_costs_ptr, int * d_adjMtx_ptr,
        float * d_csr_values, int * d_csr_columns, int * d_csr_offsets, float * d_x, float * d_b, 
        int64_t nnz, int numSupplies, int numDemands);

__host__ void find_dual_using_dense_solver(float * u_vars_ptr, float * v_vars_ptr, 
        float * d_costs_ptr, int * d_adjMtx_ptr,
        float * d_A, float * d_x, float * d_b, 
        int numSupplies, int numDemands);


