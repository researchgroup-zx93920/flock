#include <iostream>
#include <chrono>

#include "parallel_structs.h"
#include "parallel_kernels.h"



// ALLOCATE and DE-ALLOCATE RESOURCES
__host__ void initialize_device_PIVOT(int ** backtracker, stackNode ** stack, bool ** visited, 
    int ** depth, float ** loop_minimum, int ** loop_min_from, int ** loop_min_to, int ** loop_min_id,
    vertex_conflicts ** v_conflicts, int numSupplies, int numDemands);

__host__ void terminate_device_PIVOT(int * backtracker, stackNode * stack, bool * visited, 
    int * depth, float * loop_minimum, int * loop_min_from, int * loop_min_to, int * loop_min_id,
    vertex_conflicts * v_conflicts);

// Sequencial Pivoting API >>>

__host__ void perform_a_sequencial_pivot(int * backtracker, stackNode * stack, bool * visited,
    int * h_adjMtx_ptr, float * h_flowMtx_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr,
    bool &result, int pivot_row, int pivot_col, 
    double &dfs_time, double &resolve_time, double &adjustment_time,
    int numSupplies, int numDemands);

// Parallel Pivoting API >>>
__host__ void perform_a_parallel_pivot(int * backtracker, stackNode * stack, bool * visited,
    int * h_adjMtx_ptr, float * h_flowMtx_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr, 
    bool &result, 
    float * d_reducedCosts_ptr, int * depth, // Parallel pivot book-keeping
    float * loop_minimum, int * loop_min_from, int * loop_min_to, int * loop_min_id, // Parallel pivot book-keeping
    vertex_conflicts * v_conflicts, // Parallel pivot book-keeping
    double &dfs_time, double &resolve_time, double &adjustment_time,
    int numSupplies, int numDemands);