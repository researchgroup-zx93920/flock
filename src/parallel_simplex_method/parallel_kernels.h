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


// ##################################################
// COMPUTING REDUCED COSTS >>
// ##################################################

/*
Kernel to compute Reduced Costs in the transportation table
*/
__global__ void computeReducedCosts(float * u_vars_ptr, float * v_vars_ptr, float * d_costs_ptr, float * d_reducedCosts_ptr, 
    const int numSupplies, const int numDemands);

// ##################################################
// PIVOTING UTILITY KERNELS >>
// ##################################################

/* Convert d_adjMatrix to node degrees and node neighbors */
__host__ void make_adjacency_list(Graph &graph, const int numSupplies, const int numDemands);

__global__ void expand_all_cycles(int * d_pivot_cycles, int * d_adjMtx_transform, 
    int * d_pathMtx, MatrixCell * d_reducedCosts_ptr,  int * d_numNegativeCosts, 
    const int diameter, const int numSupplies, const int numDemands);

__global__ void derive_cells_on_paths(int * d_pivot_cycles, int * d_adjMtx_transform, 
    int * d_pathMtx, MatrixCell * d_reducedCosts_ptr,  int * d_numNegativeCosts, 
    const int diameter, const int numSupplies, const int numDemands);

__global__ void check_pivot_feasibility(MatrixCell * d_reducedCosts_ptr, const int min_indx,
                const int earlier_from, const int earlier_to, 
                int * d_adjMtx_transform, int * d_pivot_cycles,
                const int diameter, const int numSupplies, const int numDemands);

__global__ void initialize_parallel_pivot(vertexPin * empty_frontier, 
    int * d_vertex_start, int * d_vertex_degree, int * d_adjVertices,
    float * d_costs_ptr, const int numSupplies, const int numDemands);

__global__ void update_distance_path_and_create_next_frontier(int * pathMtx, int * d_adjMtx_transform, int * d_vertex_start,
            int * d_vertex_length, int * d_adjVertices, vertexPin * currLevelNodes, vertexPin * nextLevelNodes,
            int * numCurrLevelNodes, int * numNextLevelNodes,
            float * d_costs_ptr, float * opporutnity_cost, 
            const int numSupplies, const int numDemands, const int iteration_number);
            // unsigned long long int * sm_profile);

__global__ void collectNegativeReducedCosts(MatrixCell * d_reducedCosts_ptr, int * numNegativeCosts,
    float * opportunity_costs, const int numSupplies, const int numDemands);

#endif