#include "DUAL_solver.h"

namespace UV_METHOD {

__host__ void dualMalloc(DualHandler &dual, int numSupplies, int numDemands) {
    
    int V = numSupplies + numDemands;
    // Create and Initialize u and v variables 
    //  empty u and v equations using the Variable Data Type >>
    dual.h_visited = (bool *) malloc(sizeof(bool)*V);
    dual.h_variables = (float *) malloc(sizeof(float)*V);
    
    if (REDUCED_COST_MODE == "parallel") {
        
        gpuErrchk(cudaMalloc((void **) &dual.u_vars_ptr, sizeof(float)*numSupplies));
        gpuErrchk(cudaMalloc((void **) &dual.v_vars_ptr, sizeof(float)*numDemands));

    }
}

__host__ void dualFree(DualHandler &dual) {
     
     if (REDUCED_COST_MODE == "parallel") {

        gpuErrchk(cudaFree(dual.u_vars_ptr));
        gpuErrchk(cudaFree(dual.v_vars_ptr));

     }
        
        free(dual.h_visited);
        free(dual.h_variables);

}

__host__ void find_dual_using_host_bfs(DualHandler &dual,  Graph &graph, float * h_costs_ptr, 
        int numSupplies, int numDemands) {

        // Copy Adjacency list on host >> assuming Tranformation already occured at the start of pivoting 
        gpuErrchk(cudaMemcpy(graph.h_vertex_degree, &graph.d_vertex_degree[1], sizeof(int)*graph.V, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(graph.h_vertex_start, graph.d_vertex_start, sizeof(int)*graph.V, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(graph.h_adjVertices, graph.d_adjVertices, sizeof(int)*2*(graph.V-1), cudaMemcpyDeviceToHost));

        thrust::fill(thrust::host, dual.h_visited, dual.h_visited + graph.V, false);
        thrust::fill(thrust::host, dual.h_variables, dual.h_variables + graph.V, 0.0f);

        // Initialize >>
        std::queue<int> assigned_parents;
        dual.h_visited[0] = true;
        // The value of u0 is already zero as initialized
        assigned_parents.push(0);

        // Perform a BFS on Host (trickle down) >> 
        int parent, child, row, col;
        
        int * h_start = graph.h_vertex_start;
        int * h_length = graph.h_vertex_degree;
        int * h_Ea = graph.h_adjVertices;


        while (!assigned_parents.empty()) {
                parent = assigned_parents.front();
                for (int i = h_start[parent]; i < h_start[parent] + h_length[parent]; i++) {
                        child = h_Ea[i];
                        if (!dual.h_visited[child]) {
                                dual.h_visited[child] = true;
                                row = min(parent, child);
                                col = max(parent, child) - numSupplies;
                                dual.h_variables[child] = h_costs_ptr[row*numDemands + col] - dual.h_variables[parent];
                                assigned_parents.push(child);
                        }
                }
                assigned_parents.pop();
        }

        // Transfer back to GPU >> 
        if (REDUCED_COST_MODE == "parallel") {

                gpuErrchk(cudaMemcpy(dual.u_vars_ptr, &dual.h_variables[0], sizeof(int)*numSupplies, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(dual.v_vars_ptr, &dual.h_variables[numSupplies], sizeof(int)*numDemands, cudaMemcpyHostToDevice));
        }
}

} // End of namespace