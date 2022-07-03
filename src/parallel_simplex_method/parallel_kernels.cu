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

/*
Kernel to convert float cost matrix to the MatrixCell objects
*/
__global__ void createCostMatrix(MatrixCell *d_costMtx, float * d_costs_ptr, const int n_supplies, const int n_demands)
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
        const int numSupplies, const int numDemands) 
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


/* 
Transfer flows on device and prepare an adjacency and flow matrix using the flows from IBFS
In case of sequencial pivoting - one would need a copy of adjMatrix on the host to traverse the graph
*/
__host__ void create_IBF_tree_on_host_device(Graph &graph, flowInformation * feasible_flows, 
    const int numSupplies, const int numDemands) {

    int V = numSupplies+numDemands;
    int _utm_entries = (V*(V+1))/2; // Number of entries in upper triangular matrix 

    gpuErrchk(cudaMalloc((void **) &graph.d_adjMtx_ptr, sizeof(int)*_utm_entries)); 
    thrust::fill(thrust::device, graph.d_adjMtx_ptr, (graph.d_adjMtx_ptr) + _utm_entries, 0);

    gpuErrchk(cudaMalloc((void **) &graph.d_flowMtx_ptr, sizeof(float)*(V-1)));
    thrust::fill(thrust::device, graph.d_flowMtx_ptr, (graph.d_flowMtx_ptr) + (V-1), 0);

    // Make a replica of feasible flows on device
    flowInformation * d_flows_ptr;
    gpuErrchk(cudaMalloc((void **) &d_flows_ptr, sizeof(flowInformation)*(V-1)));
    gpuErrchk(cudaMemcpy(d_flows_ptr, feasible_flows, sizeof(flowInformation)*(V-1), cudaMemcpyHostToDevice));

    // Small kernel to parallely create a tree using the flows
    create_initial_tree <<< ceil(1.0*(V-1)/blockSize), blockSize >>> (d_flows_ptr, graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Now device_flows are useless; 
    // All information about graph is now contained within d_adjMatrix, d_flowMatrix on device =>
    gpuErrchk(cudaFree(d_flows_ptr));
    
    // Make a copy on host >>
    graph.h_adjMtx_ptr = (int *) malloc(sizeof(int)*(_utm_entries));
    gpuErrchk(cudaMemcpy(graph.h_adjMtx_ptr, graph.d_adjMtx_ptr, sizeof(int)*(_utm_entries), cudaMemcpyDeviceToHost));
    graph.h_flowMtx_ptr = (float *) malloc(sizeof(float)*(V-1));
    gpuErrchk(cudaMemcpy(graph.h_flowMtx_ptr, graph.d_flowMtx_ptr, sizeof(float)*(V-1), cudaMemcpyDeviceToHost));

    // Iterations would also work with a adjacency list
    gpuErrchk(cudaMalloc((void **) &graph.d_vertex_start, sizeof(int)*(V)));
    gpuErrchk(cudaMalloc((void **) &graph.d_vertex_degree, sizeof(int)*(V+1)));
    gpuErrchk(cudaMalloc((void **) &graph.d_adjVertices, sizeof(int)*2*(V-1)));
    graph.h_vertex_start = (int *) malloc(sizeof(int)*V);
    graph.h_vertex_degree = (int *) malloc(sizeof(int)*V);
    graph.h_adjVertices = (int *) malloc(sizeof(int)*2*(V-1));
    
    // Initialize host graph 
    std::vector<int> _neighborhood;
    for (int i=0; i < V; i++) {
        graph.h_Graph.push_back(_neighborhood);
    }

}

/*
Given a feasible tree on device, load a feasible solution to transportation problem on the host
*/
__host__ void retrieve_solution_on_current_tree(flowInformation * feasible_flows, Graph &graph,
    int &active_flows, const int numSupplies, const int numDemands)
{
    
    // Recreate device flows using the current adjMatrix
    flowInformation default_flow;
    default_flow.qty = 0;

    flowInformation * d_flows_ptr;
    gpuErrchk(cudaMalloc((void **) &d_flows_ptr, sizeof(flowInformation)*(numSupplies*numDemands)));
    thrust::fill(thrust::device, d_flows_ptr, d_flows_ptr + (numSupplies*numDemands), default_flow);

    dim3 __blockDim(blockSize, blockSize, 1);
    int grid_size = ceil(1.0*(numSupplies+numDemands)/blockSize); // VxV threads
    dim3 __gridDim(grid_size, grid_size, 1);
    retrieve_final_tree <<< __gridDim, __blockDim >>> (d_flows_ptr, graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy the (flows > 0) back on the host >>
    auto flow_end = thrust::remove_if(thrust::device,
        d_flows_ptr, d_flows_ptr + (numSupplies*numDemands), is_zero());
    int flow_count = flow_end - d_flows_ptr;
    // Update active flows in result 
    active_flows = flow_count;
    gpuErrchk(cudaMemcpy(feasible_flows, d_flows_ptr, (flow_count)*sizeof(flowInformation), cudaMemcpyDeviceToHost));

}

/* Clear up the memory occupied by graph on host on device */
__host__ void close_solver(Graph &graph)
{

    gpuErrchk(cudaFree(graph.d_vertex_start));
    gpuErrchk(cudaFree(graph.d_vertex_degree));
    gpuErrchk(cudaFree(graph.d_adjVertices));
    gpuErrchk(cudaFree(graph.d_adjMtx_ptr));
    gpuErrchk(cudaFree(graph.d_flowMtx_ptr));
    
    // DEBUG :: following is double free (-_-) idk where!
    // Since it's a single time allocation - no tension get rid of this!
    
    free(graph.h_vertex_start);
    free(graph.h_vertex_degree);
    free(graph.h_adjVertices);
    free(graph.h_adjMtx_ptr);
    free(graph.h_flowMtx_ptr);
    

}

__global__ void determine_length(int * length, int * d_adjMtx_ptr, const int V) {

    __shared__ int L;
    int location;
    
    if (threadIdx.x == 0) {
        L = 0;
        length[0] = 0;
    }
    __syncthreads();

    if (blockIdx.x < V) {
            int j = threadIdx.x;
            while (j < V) {
                int idx = TREE_LOOKUP(blockIdx.x, j, V);
                if (d_adjMtx_ptr[idx] > 0) {
                    location = atomicAdd(&L, 1);
                }
                j = j + blockDim.x;
            }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {

        length[blockIdx.x+1] = L;
    }
    
}

__global__ void fill_Ea(int * start, int * Ea, int * d_adjMtx_ptr, const int V, const int numSupplies) {
    
    // int i = blockIdx.x*blockDim.x + threadIdx.x;
    int offset = start[blockIdx.x];
    __shared__ int L;
    if (threadIdx.x == 0) {
        L = 0;
    }
    __syncthreads();

    if (blockIdx.x < V) {
            int j = threadIdx.x;
            while (j < V) {
                int idx = TREE_LOOKUP(blockIdx.x, j, V);
                if (d_adjMtx_ptr[idx] > 0) {
                    int location = atomicAdd(&L, 1);
                    Ea[offset + location] = j;
                }
                j = j + blockDim.x;
            }
    }
}

/*
DEBUG UTILITY : VIEW ADJACENCY LIST STRCTURE 
*/
__host__ void __debug_view_adjList(int * start, int * length, int * Ea, const int V) 
{        
        int * h_length = (int *) malloc(sizeof(int)*V);
        int * h_start = (int *) malloc(sizeof(int)*V);
        int * h_Ea = (int *) malloc(sizeof(int)*2*(V-1));

        cudaMemcpy(h_length, length, sizeof(int)*V, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_start, start, sizeof(int)*V, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ea, Ea, sizeof(int)*2*(V-1), cudaMemcpyDeviceToHost);

        std::cout<<"Str = [ ";
        for (int i =0; i < V; i++){
                std::cout<<h_start[i]<<", ";
        }
        std::cout<<"]"<<std::endl;
        std::cout<<"Len = [ ";
        for (int i =0; i < V; i++){
                std::cout<<h_length[i]<<", ";
        }
        std::cout<<"]"<<std::endl;
        std::cout<<"Ea = [ ";
        for (int i =0; i < 2*(V-1); i++){
                std::cout<<h_Ea[i]<<", ";
        }
        std::cout<<"]"<<std::endl;
        
        free(h_length);
        free(h_Ea);
        free(h_start);
        // *************** END OF DEBUG UTILITY ***************

}

__host__ void make_adjacency_list(Graph &graph, const int numSupplies, const int numDemands) {

        // Kernel Dimensions >> 
        dim3 __blockDim(64, 1, 1); 
        dim3 __gridDim(graph.V, 1, 1);

        determine_length <<< __gridDim, __blockDim >>> (graph.d_vertex_degree, graph.d_adjMtx_ptr, graph.V);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        thrust::inclusive_scan(thrust::device, graph.d_vertex_degree, graph.d_vertex_degree + graph.V, graph.d_vertex_start);
        
        // fill_Ea <<< __gridDim, __blockDim >>> (graph.d_vertex_start, graph.d_adjVertices, graph.d_adjMtx_ptr, graph.V, numSupplies);
        fill_Ea <<< __gridDim, __blockDim >>> (graph.d_vertex_start, graph.d_adjVertices, graph.d_adjMtx_ptr, graph.V, numSupplies);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Entries in Matrix >>
        // int _utm_entries = (V*(V+1))/2; // Number of entries in upper triangular matrix
        // auto result_end = thrust::copy_if(thrust::device, d_adjMtx_ptr, d_adjMtx_ptr + _utm_entries, 
        //                     Ea, is_nonzero_entry()); // --> need col indices of non-zeros

        // DEBUG :: 
        // __debug_view_adjList(graph.d_vertex_start, &graph.d_vertex_degree[1], graph.d_adjVertices, graph.V);
        // exit(0);
}

// ##################################################
// COMPUTING REDUCED COSTS >>
// ##################################################


/*
Kernel to compute Reduced Costs in the transportation table
*/
__global__ void computeReducedCosts(float * u_vars_ptr, float * v_vars_ptr, float * d_costs_ptr, float * d_reducedCosts_ptr, 
    const int numSupplies, const int numDemands)
{

        __shared__ float U[reducedcostBlock];
        __shared__ float V[reducedcostBlock];
        
        int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

        if (row_indx < numSupplies && col_indx < numDemands) {
            // r =  C_ij - (u_i + v_j);
            U[threadIdx.y] = u_vars_ptr[row_indx];
            V[threadIdx.x] = v_vars_ptr[col_indx];
        }
        
        __syncthreads();
        
        if (row_indx < numSupplies && col_indx < numDemands) {

            float r = d_costs_ptr[row_indx*numDemands+col_indx] - U[threadIdx.y] - V[threadIdx.x];
            d_reducedCosts_ptr[row_indx*numDemands+col_indx] = r;
        
        }
}


/*
Recursively explore pathMtx and store all the discovered cycles in the expanded form 
Can spped this up using a 3D grid? 
*/
__global__ void expand_all_cycles(int * d_pivot_cycles, int * d_adjMtx_transform, 
    int * d_pathMtx, MatrixCell * d_reducedCosts_ptr,  int * d_numNegativeCosts, 
    const int diameter, const int numSupplies, const int numDemands) 
{

    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx < *d_numNegativeCosts) {

        MatrixCell pivot_cell = d_reducedCosts_ptr[idx];
        int col_indx = pivot_cell.col;  // demand point
        int row_indx = pivot_cell.row;  // supply point 
        int offset_1 = diameter*idx; // offset for cycle store
        int V = numSupplies + numDemands;
        // Entering Edge is from supply - i to demand - j we discover a cycle by finding a shortest path from [j] -> [i]
        int offset_path = row_indx*V + (col_indx + numSupplies); // offset for pathMtx
        int offset_distance = row_indx*numDemands + col_indx;

        int depth = d_adjMtx_transform[offset_distance] + 1;
        int current_vtx = col_indx + numSupplies; // backtrack from - j
        int target_vtx = row_indx; // reach - i 
        d_pivot_cycles[offset_1] = target_vtx;
        
        int d = 1;

        while (d < depth) {
            d_pivot_cycles[offset_1 + d] = current_vtx;
            current_vtx = d_pathMtx[target_vtx*V + current_vtx];
            d++;
        }

        d_pivot_cycles[offset_1+depth] = target_vtx;

    }
}

__global__ void derive_cells_on_paths(int * d_pivot_cycles, int * d_adjMtx_transform, 
    int * d_pathMtx, MatrixCell * d_reducedCosts_ptr,  int * d_numNegativeCosts, 
    const int diameter, const int numSupplies, const int numDemands) 
{

    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx < *d_numNegativeCosts) {

        MatrixCell pivot_cell = d_reducedCosts_ptr[idx];
        int col_indx = pivot_cell.col;  // demand point
        int row_indx = pivot_cell.row;  // supply point 
        int offset_1 = diameter*idx; // offset for cycle store
        int V = numSupplies + numDemands;
        // Entering Edge is from supply - i to demand - j we discover a cycle by finding a shortest path from [j] -> [i]
        int offset_path = row_indx*V + (col_indx + numSupplies); // offset for pathMtx
        int offset_distance = row_indx*numDemands + col_indx;

        int depth = d_adjMtx_transform[offset_distance];
        int current_vtx = col_indx + numSupplies; // backtrack from - j
        int target_vtx = row_indx; // reach - i 
        int next_vtx;

        for (int d = 0; d < depth; d++) {
            next_vtx = d_pathMtx[target_vtx*V + current_vtx];
            d_pivot_cycles[offset_1 + d] = (1 - d%2)*(next_vtx*numDemands + (current_vtx-numSupplies)) +
                                        (d%2)*(current_vtx*numDemands + (next_vtx-numSupplies));
            current_vtx = next_vtx;
        }
    }
}


/*
Check if a cycle is still feasible if any of the edges of the cycle located at min_indx has been used
If it is infeasible then set the reduced cost of this cell as non-negative to deactivate pivot here
*/
__global__ void check_pivot_feasibility(MatrixCell * d_reducedCosts_ptr, const int min_indx,
                const int earlier_from, const int earlier_to, 
                int * d_adjMtx_transform, int * d_pivot_cycles,
                const int diameter, const int numSupplies, const int numDemands) {

    // Nomenclature : 
    // this_cycle - cycle located on this blockIdx.x
    // earlier_cycle - cycle located on the min_index 
    // Context : Edges in earlier cycle will be used by pivot (because min_reduced cost)
    //      Then if there are any common edges between this_cycle and earlier_cycle
    //      then this_cycle conflicts with earlier_cycle and it has to die
    //      Set reduced cost = non_negative for this cycle
    
    __shared__ int this_cycle[resolveBlockSize];
    __shared__ int earlier_cycle[resolveBlockSize];
    __shared__ bool conflict; // = false; 
    __shared__ MatrixCell this_cycleE;
    __shared__ int this_cycle_depth ,earlier_cycle_depth, num_tiles_earlier, y_offset;
    
    // This cycle is available at offset - blockIdx.x*diameter  
    // Earlier cycle is available at offset - min_indx*diameter
    if (threadIdx.x==0) {

        this_cycleE = d_reducedCosts_ptr[blockIdx.x];
        this_cycle_depth = d_adjMtx_transform[this_cycleE.row*numDemands + this_cycleE.col];
        earlier_cycle_depth = d_adjMtx_transform[earlier_from*numDemands + earlier_to]; 
        num_tiles_earlier = ceil(1.0*earlier_cycle_depth/resolveBlockSize);
    
    }
    
    __syncthreads();
    
    // First loaded this cycle in shared memory - 
    if (this_cycleE.cost < 0) {
        
        // One time load per block
        int offset = blockIdx.x*diameter + blockIdx.y*resolveBlockSize; 
        if (blockIdx.y*resolveBlockSize + threadIdx.x < this_cycle_depth) {
            this_cycle[threadIdx.x] = d_pivot_cycles[offset + threadIdx.x];
        }

        __syncthreads();

        // Tiling for earlier cycle >>
        // This cycle static
        for (int tile = 0; tile < num_tiles_earlier; tile++) {

            int load_from = min_indx*diameter + tile*resolveBlockSize;
            int load_upto = min_indx*diameter + earlier_cycle_depth;

            if (tile*resolveBlockSize + threadIdx.x < earlier_cycle_depth) {
                earlier_cycle[threadIdx.x] = d_pivot_cycles[load_from + threadIdx.x];
            }

            __syncthreads();

            // Comparison within block >> 
            // Each thread makes block-size comparisons
        
            for (int i = 0; i < resolveBlockSize; i++) {

                if (tile*resolveBlockSize + threadIdx.x < earlier_cycle_depth && blockIdx.y*resolveBlockSize + i < this_cycle_depth) {
                    bool compare = earlier_cycle[threadIdx.x] == this_cycle[i]; 
                    if (compare) {
                        d_reducedCosts_ptr[blockIdx.x].cost = epsilon; // atomic not required - race is healthy!
                    }
                }
            }
        }
    }            
}

/* Set initial values in adj Matrix and path matrix */
__global__ void initialize_parallel_pivot(vertexPin * empty_frontier, 
    int * d_vertex_start, int * d_vertex_degree, int * d_adjVertices,
    float * d_costs_ptr, const int numSupplies, const int numDemands) {
    
    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;

    if (row_indx < numSupplies) {
        int degree = d_vertex_degree[row_indx];
        if (col_indx < degree) {
            // Parallely load the neighbourhood of idx 
            int start = d_vertex_start[row_indx];
            int this_vertex = d_adjVertices[start + col_indx];
            float r_cost = 0; //-1.0f*d_costs_ptr[row_indx*numDemands + col_indx];
            vertexPin _pin = {.from = row_indx, .via = this_vertex, .to = this_vertex, .skip = row_indx, .recirculation = r_cost};
            // Load the pin in the frontier
            empty_frontier[start + col_indx] = _pin;
        }
    }    
}

// SM PROFILING >> LOAD IMBALANCE CHECKING 
__device__ uint get_smid(void) {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

/* BFS Kernel Launches 1  - block_per_vertex */
__global__ void update_distance_path_and_create_next_frontier(
                                        int * pathMtx,
                                        int * d_adjMtx_transform,
                                        int * d_vertex_start,
                                        int * d_vertex_length,
                                        int * d_adjVertices,
                                        vertexPin * currLevelNodes,
                                        vertexPin * nextLevelNodes,
                                        int * numCurrLevelNodes,
                                        int * numNextLevelNodes,
                                        float * d_costs_ptr,
                                        float * opportunity_cost,
                                        const int numSupplies,
                                        const int numDemands,
                                        const int iteration_number)  
{
    // SM PROFILING >> LOAD IMBALANCE CHECKING 
    // unsigned long long int * sm_profile
 
    // Initialize shared memory queue
    __shared__ int write_offset, read_offset, degree, skip_location;
    __shared__ float update_recirculation;
    __shared__ unsigned long long int time; 
    vertexPin this_vertex = currLevelNodes[blockIdx.x];
    int V = numSupplies + numDemands;

    if (threadIdx.x == 0) {

        read_offset = d_vertex_start[this_vertex.to];
        degree = d_vertex_length[this_vertex.to];
        write_offset = atomicAdd(numNextLevelNodes, degree-1);
        skip_location = V;
        // SM PROFILING >> LOAD IMBALANCE CHECKING 
        // time = clock64();

        // Update the pathMatrix and distance
        // Path for this vertex is updated and neighbors of this_vertex are queued >>
        int path_idx = this_vertex.from*V + this_vertex.to;
        int i = (int) iteration_number%2;
        pathMtx[path_idx] = this_vertex.skip;

        // Update reduced cost accumulator >>
        unsigned int cost_indx = (1 - i)*(this_vertex.skip*numDemands + (this_vertex.to - numSupplies)) 
            + (i)*(this_vertex.to*numDemands + (this_vertex.skip - numSupplies));
        update_recirculation = this_vertex.recirculation + ((int) pow(-1, i+1))*d_costs_ptr[cost_indx];

        if (i==0) { // No divergence here - all threads in iteration execute the same
            
            int dist_indx = this_vertex.from*numDemands + (this_vertex.to - numSupplies);
            d_adjMtx_transform[dist_indx] = iteration_number+1;
            opportunity_cost[dist_indx] = opportunity_cost[dist_indx] + update_recirculation;
        
        }
    }

    __syncthreads();

    // For all nodes in the curent level -> get all neighbors of the node
    // Except the skip add rest to the block queue
    // If full, add it to the global queue
    this_vertex.recirculation =  update_recirculation;
    int current = this_vertex.to;
    int current_skip = this_vertex.skip;
    
    for (int j = threadIdx.x; j < degree; j += blockDim.x) {

        int neighbor = d_adjVertices[read_offset + j];

        // check the position of skip ->
        if (neighbor==current_skip) {
            skip_location = j;
        }
    }

    __syncthreads();

    for (int j = threadIdx.x; j < degree; j += blockDim.x) {

        int neighbor = d_adjVertices[read_offset + j];

        // check the position of skip ->
        if (j < skip_location) {

            this_vertex.skip = current;
            this_vertex.to = neighbor;
            nextLevelNodes[write_offset+j] = this_vertex;
        
        }

        else if (j > skip_location) {

            this_vertex.skip = current;
            this_vertex.to = neighbor;
            nextLevelNodes[write_offset+j-1] = this_vertex;

        }

    }

    // SM PROFILING >> LOAD IMBALANCE CHECKING 
    // __syncthreads();
    // if (threadIdx.x == 0) {
    //     time = clock64() - time;
    //     atomicAdd(&sm_profile[get_smid()], time);
    // }

}

__global__ void collectNegativeReducedCosts(MatrixCell * d_reducedCosts_ptr, int * numNegativeCosts,
    float * opportunity_costs, const int numSupplies, const int numDemands) 
{
    
    int row_indx = threadIdx.y + blockIdx.y*blockDim.y;
    int col_indx = threadIdx.x + blockIdx.x*blockDim.x;

    if (row_indx < numSupplies && col_indx < numDemands) {

        int gid = row_indx*numDemands + col_indx;
        float r_cost = opportunity_costs[gid];
        if (r_cost < 0 && abs(r_cost) > epsilon2) {
            int position = atomicAdd(numNegativeCosts, 1);
            MatrixCell cost_obj = {.row = row_indx, .col = col_indx, .cost = r_cost};
             d_reducedCosts_ptr[position] = cost_obj;
        }
    }
}

#endif