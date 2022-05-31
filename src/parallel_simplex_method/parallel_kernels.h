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

#include "parallel_structs.h"

#ifndef KERNELS
#define KERNELS

// ##################################################
// PREPROCESS and POSTPROCESS  >>
// ##################################################

__global__ void createCostMatrix(MatrixCell *d_costMtx, float * d_costs_ptr, int n_supplies, const int n_demands);

__global__ void create_initial_tree(flowInformation * d_flows_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr,
    const int numSupplies, const int numDemands);

__global__ void retrieve_final_tree(flowInformation * d_flows_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr,
        const int numSupplies, const int numDemands);

__host__ void create_IBF_tree_on_host_device(Graph &graph, flowInformation * feasible_flows, 
    const int numSupplies, const int numDemands);

__host__ void close_solver(Graph &graph);

__host__ void retrieve_solution_on_current_tree(flowInformation * feasible_flows, Graph &graph,
    int &active_flows, const int numSupplies, const int numDemands);

/*Determine number of non zero elements in each row of adjacency matrix */
__global__ void determine_length(int * length, int * d_adjMtx_ptr, const int V);

/* Convert adjacency matrix to adjacency list - compaction kernel */
__global__ void fill_Ea(int * start, int * Ea, int * d_adjMtx_ptr, const int V, const int numSupplies);

/* View adjacency list */
__host__ void __debug_view_adjList(int * start, int * length, int * Ea, const int V);

/* Convert d_adjMatrix to adjacency list (utillized by both host_bfs and DFS in pivoting) */
__host__ void make_adjacency_list(Graph &graph, const int numSupplies, const int numDemands);

// ##################################################
// SOLVING DUAL >>
// ##################################################

/*
APPROACH 1 :
Kernels concerned with solving the UV System using a BFS Traversal Approach
*/

__global__ void copy_row_shadow_prices(Variable * U_vars, float * u_vars_ptr, const int numSupplies);

__global__ void copy_col_shadow_prices(Variable * V_vars, float * v_vars_ptr, const int numDemands);

__global__ void initialize_U_vars(Variable * U_vars, const int numSupplies);

__global__ void initialize_V_vars(Variable * V_vars, const int numDemands);

/*
Breadth First Traversal on UV
*/

/* assign_next is deprecated because of inherence race condition */
__global__ void assign_next(int * d_adjMtx_ptr, float * d_costs_ptr, 
    Variable *u_vars, Variable *v_vars, const int numSupplies, const int numDemands);

__global__ void CUDA_BFS_KERNEL(int * start, int * length, int *Ea, bool * Fa, bool * Xa, 
        float * variables, float * d_costs_ptr, bool * done, const int numSupplies, const int numDemands, const int V);

/*
APPROACH 2:
Kernels concerned with solving the UV System using a using a matrix solver
*/

// Custom Fill kernel for csr row pointers
__global__ void fill_csr_offset (int * d_csr_offsets, const int length);

/*
Create a dense linear system in parallel by looking at current feasible tree 
*/
__global__ void initialize_dense_u_v_system(float * d_A, float * d_b, int * d_adjMtx_ptr, 
    float * d_costs_ptr, const int numSupplies, const int numDemands);

/*
Create a sparse linear system in parallel by looking at current feasible tree 
*/
__global__ void initialize_sparse_u_v_system(int * d_csr_columns, float * d_b, int * d_adjMtx_ptr, 
    float * d_costs_ptr, const int numSupplies, const int numDemands);

/*
Load the solution of system to the appropriate place
*/
__global__ void retrieve_uv_solution(float * d_x, float * u_vars_ptr, float * v_vars_ptr, const int numSupplies, const int numDemands);

// ##################################################
// COMPUTING REDUCED COSTS >>
// ##################################################

/*
Kernel to compute Reduced Costs in the transportation table
*/
__global__ void computeReducedCosts(float * u_vars_ptr, float * v_vars_ptr, float * d_costs_ptr, float * d_reducedCosts_ptr, 
    const int numSupplies, const int numDemands);

__global__ void computeReducedCosts(float * u_vars_ptr, float * v_vars_ptr, float * d_costs_ptr, MatrixCell * d_reducedCosts_ptr, 
    const int numSupplies, const int numDemands);

// ##################################################
// PARALLEL PIVOTING UTILITY KERNELS >>
// ##################################################

__global__ void _naive_floyd_warshall_kernel(const int k, const int V, int * d_adjMtx, int * path);

__global__ void fill_adjMtx(int * d_adjMtx_transform, int * d_adjMtx_actual, int * d_pathMtx, const int V);

__global__ void expand_all_cycles(int * d_adjMtx_transform, int * d_pathMtx, int * d_pivot_cycles, const int diameter, const int numSupplies, const int numDemands);

__global__ void check_pivot_feasibility(int * d_adjMtx_transform, int * d_pivot_cycles, MatrixCell * d_reducedCosts_ptr, int min_r_index, const int diameter, const int numSupplies, const int numDemands);

__global__ void check_pivot_feasibility(int * d_adjMtx_transform, int * d_pivot_cycles, 
    float * d_opportunity_costs, int min_r_index, const int diameter, const int numSupplies, const int numDemands);

__global__ void check_pivot_feasibility_dfs(int * depth, int * backtracker, 
    MatrixCell * d_reducedCosts_ptr, const int min_r_index, 
    const int numSupplies, const int numDemands, const int num_threads_launching);

__global__ void compute_opportunity_cost_and_delta(int * d_adjMtx_ptr, float * d_flowMtx_ptr, float * d_costs_ptr, 
    int * d_adjMtx_transform, int * d_pivot_cycles, float * d_opportunity_costs, 
    const int diameter, const int numSupplies, const int numDemands);

__global__ void compute_opportunity_cost(int * d_adjMtx_ptr, float * d_flowMtx_ptr, float * d_costs_ptr, 
    int * d_adjMtx_transform, int * d_pivot_cycles, float * d_opportunity_costs, 
    const int diameter, const int numSupplies, const int numDemands);

#endif