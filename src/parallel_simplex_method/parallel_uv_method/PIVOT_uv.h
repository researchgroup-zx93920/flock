/* 
NOTE - user may not understand this file directly 
Follow the implementation of uv_method_parallel class methods
and all of this will make sense

Todo: 
1. Make a pivoting strategy selector -  pivot malloc will take strategy and assign it within a iterator in handler
2. Optimize function arguments when life is good (low prioirity)
3. 
*/

#include "../parallel_structs.h"
#include "../parallel_kernels.h"


namespace UV_METHOD {

// ALLOCATE and DE-ALLOCATE RESOURCES
__host__ void pivotMalloc(PivotHandler &pivot, int numSupplies, int numDemands);

__host__ void pivotFree(PivotHandler &pivot);


// Sequencial Pivoting API >>>
__host__ void perform_a_sequencial_pivot(PivotHandler &pivot, PivotTimer &timer,
    Graph &graph, MatrixCell * d_reducedCosts_ptr, bool &result, int numSupplies, int numDemands);

// Parallel Pivoting API >>>
__host__ void perform_a_parallel_pivot_dfs(PivotHandler &pivot, PivotTimer &timer, 
    Graph &graph, MatrixCell * d_reducedCosts_ptr, bool &result, int numSupplies, int numDemands, int iteration);

__host__ void perform_a_parallel_pivot_floyd_warshall(PivotHandler &pivot, PivotTimer &timer, 
    Graph &graph, MatrixCell * d_reducedCosts_ptr, bool &result, int numSupplies, int numDemands, int iteration);

}