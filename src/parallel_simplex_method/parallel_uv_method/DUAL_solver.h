/* 
NOTE - user may not understand this file directly 
Follow the implementation of uv_method_parallel class methods
and all of this will suddenly make sense

Todo: 
1. Make a dual strategy selector - dual malloc will take strategy and assign it within a iterator in handler
*/

#include "../parallel_structs.h"
#include "../parallel_kernels.h"

namespace UV_METHOD {
    
// ALLOCATE and DE-ALLOCATE RESOURCES for dual based on appropriate strategy
__host__ void dualMalloc(DualHandler &dual, int numSupplies, int numDemands);

__host__ void dualFree(DualHandler &dual);

// BREADTH FIRST SEARCH
__host__ void find_dual_using_host_bfs(DualHandler &dual,  Graph &graph, float * h_costs_ptr, int numSupplies, int numDemands);

} // End of namespace