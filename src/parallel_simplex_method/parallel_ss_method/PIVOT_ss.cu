#include "PIVOT_ss.h"

namespace SS_METHOD {
/* 
Setup necessary resources for pivoting 
these resources are static and to be shared/overwritten between iterations
*/
__host__ void pivotMalloc(PivotHandler &pivot, int numSupplies, int numDemands, char * pivoting_strategy) {

    int V = numSupplies + numDemands;
    
    gpuErrchk(cudaMalloc((void **) &pivot.d_adjMtx_transform, numSupplies*numDemands*sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &pivot.d_pathMtx, numSupplies*(V)*sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &pivot.current_frontier_length, sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &pivot.next_frontier_length, sizeof(int)));

    // Allocate resources for BFS 
    gpuErrchk(cudaMalloc((void **) &pivot.d_bfs_frontier_current, sizeof(vertexPin)*numSupplies*numDemands));
    gpuErrchk(cudaMalloc((void **) &pivot.d_bfs_frontier_next, sizeof(vertexPin)*numSupplies*numDemands));
    
    // Allocate for Pivoting
    gpuErrchk(cudaMalloc((void **) &pivot.d_reducedCosts_ptr, sizeof(MatrixCell)*numSupplies*numDemands));
    gpuErrchk(cudaMalloc((void **) &pivot.d_num_NegativeCosts, sizeof(int)));
    
    // Allocate mem for pivoting items
    gpuErrchk(cudaMalloc((void **) &pivot.opportunity_cost, numSupplies*numDemands*sizeof(float)));
    
    // Storing indexes of deconflicted cycles >>
    pivot.deconflicted_cycles = (int *) malloc(MAX_DECONFLICT_CYCLES(numSupplies, numDemands)*sizeof(int)); // upper bound = M + N - 1/3
    // Allocate appropriate memory for executing pivots
    pivot.deconflicted_cycles_depth = (int *) malloc(sizeof(int)); // Store depth of each cycle
    // Each cycle has a size less than max possible diameter
    pivot.deconflicted_cycles_backtracker = (int *) malloc(V*sizeof(int));

    // Todo >> Print size of memory allocated on device / Prior Estimatation
    // bytes 
    size_t size_of_memory = (numSupplies*numDemands*sizeof(int)) + numSupplies*(V)*sizeof(int) + 
        sizeof(vertexPin)*numSupplies*numDemands*2 + sizeof(int)*3 + sizeof(MatrixCell)*numSupplies*numDemands +
        numSupplies*numDemands*sizeof(float);
    
    // MegaBytes
    size_of_memory = size_of_memory/(1024*1024);
    std::cout<<"SS PIVOT MALLOC : "<<size_of_memory<<" MB of device memory allocated!";
 
}

/* 
Free up acquired resources for pivoting on host device 
*/
__host__ void pivotFree(PivotHandler &pivot, char * pivoting_strategy) {

    // Free up memory space >>
    gpuErrchk(cudaFree(pivot.d_adjMtx_transform));
    gpuErrchk(cudaFree(pivot.d_pathMtx));
    gpuErrchk(cudaFree(pivot.opportunity_cost));
    gpuErrchk(cudaFree(pivot.current_frontier_length));
    gpuErrchk(cudaFree(pivot.next_frontier_length));

    // De-allocate Resources for BFS >>
    gpuErrchk(cudaFree(pivot.d_bfs_frontier_current));
    gpuErrchk(cudaFree(pivot.d_bfs_frontier_next));
    
     // De-allocate for Pivoting
    gpuErrchk(cudaFree(pivot.d_reducedCosts_ptr));
    gpuErrchk(cudaFree(pivot.d_num_NegativeCosts));
    
    free(pivot.deconflicted_cycles);
    free(pivot.deconflicted_cycles_backtracker);
    free(pivot.deconflicted_cycles_depth);


}

}

__host__ static void do_flow_adjustment_on_host_device(int * h_adjMtx_ptr, float * h_flowMtx_ptr, 
        int * d_adjMtx_ptr, float * d_flowMtx_ptr, int * backtracker, float min_flow, int min_from, int min_to, int min_flow_id,
        int pivot_row, int pivot_col, int depth, int V, int numSupplies, int numDemands) {

    
    /* *************************** 
        DEBUG UTILITY // Print the discovered loop and pivoting parameters
    **************************** */
    // std::cout<<"Pivot Row : "<<pivot_row<<std::endl;
    // std::cout<<"Pivot Col : "<<pivot_col<<std::endl;
    // std::cout<<" ************** LOOP"<<std::endl;
    // std::cout<<"Depth : "<<depth<<" : ";
    // for (int j = 0; j < depth + 1; j++) {
    //     std::cout<<backtracker[j]<<" ";
    // }
    // std::cout<<std::endl<<" ************** PIVOT"<<std::endl;
    // std::cout<<"Min flow : "<<min_flow<<std::endl;
    // std::cout<<"Min from : "<<min_from<<std::endl;
    // std::cout<<"Min to : "<<min_to<<std::endl;
    // std::cout<<"Min index : "<<min_flow_id<<std::endl;
 
    int _from, _to, id;
    float _flow;
    int null_value = 0;
    int new_value = min_flow_id + 1;

    for (int i=1; i<depth; i++) 
    {
        _from = backtracker[i];
        _to = backtracker[i+1];
        id = h_adjMtx_ptr[TREE_LOOKUP(_from, _to, V)] - 1;
        _flow = ((int) pow(-1, (int)i%2))*min_flow;
        h_flowMtx_ptr[id] += _flow;
    }

    // Do the replacment between exiting i - entering j on both host and device
    // Also communicate device about the removal and addition of an extry-exit variable pair
    
    // Remove edge
    id = TREE_LOOKUP(min_from, min_to, V);
    h_adjMtx_ptr[id] = 0;
    gpuErrchk(cudaMemcpy(&d_adjMtx_ptr[id], &null_value, sizeof(int), cudaMemcpyHostToDevice));

    // Insert edge
    id = TREE_LOOKUP(pivot_row, pivot_col+ numSupplies, V);
    h_adjMtx_ptr[id] = min_flow_id + 1;
    gpuErrchk(cudaMemcpy(&d_adjMtx_ptr[id], &new_value, sizeof(int), cudaMemcpyHostToDevice));

    // Update new flow 
    h_flowMtx_ptr[min_flow_id] = min_flow;
}

__host__ static void execute_pivot_on_host_device(int * h_adjMtx_ptr, float * h_flowMtx_ptr, 
        int * d_adjMtx_ptr, float * d_flowMtx_ptr, int * backtracker, 
        int pivot_row, int pivot_col, int depth, int V, int numSupplies, int numDemands) {

    // *******************************************
    // STEP: Performing the pivot operation 
        // Step 1 - Find the Minimum flow
        // Step 2 - Adjust the Flow
    // *******************************************
            
    int id, _from = -1, _to = -1, min_flow_id = -1, min_from = -1, min_to = -1;
    float _flow, min_flow = INT_MAX;

    // ########### STEP 1 | Finding the minimum flow >>
    // Traverse the loop find the minimum flow that could be increased
    // on the incoming edge >> 
    for (int i=0; i<depth; i++) 
    {
        if (i%2==1) 
        {
            _from = backtracker[i];
            _to = backtracker[i+1];
            id = h_adjMtx_ptr[TREE_LOOKUP(_from, _to, V)] - 1;
            _flow = h_flowMtx_ptr[id];
            
            if (_flow < min_flow) 
            {
                min_flow = _flow;
                min_flow_id = id;
                min_from = _from;
                min_to = _to;
            }
        }
    }

    // ########### STEP 2 | Executing the flow adjustment >>

    // Skip the first edge (entering edge)
    // Exiting Edge will become automatically zero (min_from, min_to)
    // Note - minflow value is zero if there's a degenerate pivot!
    do_flow_adjustment_on_host_device(h_adjMtx_ptr, h_flowMtx_ptr, d_adjMtx_ptr, d_flowMtx_ptr, backtracker,
            min_flow, min_from, min_to, min_flow_id,
            pivot_row, pivot_col, depth, V, numSupplies, numDemands);

}


// ***********************************************************************

__host__ static void _debug_print_APSP(int * d_adjMtx, int * d_pathMtx, int numSupplies, int numDemands) {

    int V = numSupplies + numDemands;
    int * h_adjMtx_copy = (int *) malloc(sizeof(int)*numSupplies*numDemands);
    int * h_pathMtx = (int *) malloc(sizeof(int)*numSupplies*V);

    gpuErrchk(cudaMemcpy(h_adjMtx_copy, d_adjMtx, numSupplies*numDemands*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_pathMtx, d_pathMtx, numSupplies*V*sizeof(int), cudaMemcpyDeviceToHost));
	
    std::cout<<" ********* Distances >>"<<std::endl;
    for (int i=0; i < numSupplies; i++) {
        std::cout<<i<<" : ";
        for (int j=0; j< numDemands; j++) {
            std::cout<<h_adjMtx_copy[i*numDemands + j]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<" ********* Path >>"<<std::endl;
    for (int i=0; i<numSupplies; i++) {
        std::cout<<i<<" : ";
        for (int j=0; j<V; j++) {
            std::cout<<h_pathMtx[i*V + j]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout << "All point shortest path printed!"<<std::endl;

    free(h_adjMtx_copy);
    free(h_pathMtx);

}

__host__ static void _debug_view_discovered_cycles(PivotHandler &pivot, 
    MatrixCell * d_reducedCosts_ptr, int diameter, int h_numNegative_costs, 
    int numSupplies, int numDemands) {

    std::cout<<" ************* Viewing Expanded cycles!"<<std::endl;

    int simplex_gridDim = numSupplies*numDemands;

    // Printing all cycles
    int * h_pivot_cycles = (int *) malloc(h_numNegative_costs*(diameter)*sizeof(int));
    int * h_depth  = (int *) malloc(sizeof(int)*simplex_gridDim);
    MatrixCell * h_reducedCosts = (MatrixCell *) malloc(sizeof(MatrixCell)*h_numNegative_costs);

    gpuErrchk(cudaMemcpy(h_pivot_cycles, pivot.d_pivot_cycles, h_numNegative_costs*(diameter)*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_reducedCosts, d_reducedCosts_ptr, h_numNegative_costs*sizeof(MatrixCell), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_depth, pivot.d_adjMtx_transform, sizeof(int)*simplex_gridDim, cudaMemcpyDeviceToHost));
    
    for (int i=0; i < h_numNegative_costs; i++) 
    {
        int offset_1 = (i)*diameter;
        int offset_2 = h_reducedCosts[i].row*numDemands+ h_reducedCosts[i].col;
        int depth = h_depth[offset_2];
        std::cout<<"Path from From :"<<h_reducedCosts[i].row<<" To :"<<h_reducedCosts[i].col+numSupplies<<" | Length = "<<depth<<" ==> ";
        for (int d=0; d < depth; d++) {
            std::cout<<h_pivot_cycles[offset_1+d]<<", ";
        }
        std::cout<<std::endl;
    }
    
    free(h_depth);
    free(h_pivot_cycles);
    free(h_reducedCosts);
}

__host__ static void debug_viewOpportunity_costs(PivotHandler &pivot, float * d_costs_ptr, int numSupplies, int numDemands) {

    std::cout<<"\nViewing Computed Reduced Costs"<<std::endl;
    float * opp_costs = (float *) malloc(numSupplies*numDemands*sizeof(float));
    float * h_costs = (float *) malloc(numSupplies*numDemands*(sizeof(float)));
    cudaMemcpy(opp_costs, pivot.opportunity_cost, numSupplies*numDemands*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_costs, d_costs_ptr, numSupplies*numDemands*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<numSupplies; i++) {
        for (int j=0; j<numDemands; j++) {
            int key = i*numDemands + j;
            std::cout<<"ReducedCost["<<key<<"] = "<<opp_costs[key]<<std::endl;
            // std::cout<<"\tCost["<<key<<"] = "<<h_costs[key]<<std::endl;
        }
    }
}

/* Prints the current frontier in BFS intermediate iterations */
__host__ static void debug_viewCurrentFrontier(vertexPin * d_bfs_frontier, int length_of_frontier) {

    std::cout<<"Viewing Current Frontier"<<std::endl;

    vertexPin * h_bfs_frontier = (vertexPin *) malloc(sizeof(vertexPin)*length_of_frontier);
    cudaMemcpy(h_bfs_frontier, d_bfs_frontier, length_of_frontier*sizeof(vertexPin), cudaMemcpyDeviceToHost);

    for (int i=0; i < length_of_frontier; i++) {
        std::cout<<"Element : "<<i<<" ===> \n"<<h_bfs_frontier[i]<<std::endl;
    }
}

__host__ static void swap_frontier(vertexPin * &a, vertexPin * &b){
  vertexPin *temp = a;
  a = b;
  b = temp;
}

__host__ static void swap_lengths(int * &a, int * &b){
  int *temp = a;
  a = b;
  b = temp;
}

namespace SS_METHOD {
/*
Step 1: Find all point to all points shortest distance with Floyd Warshall using naive implementation 
    of Floyd Warshall algorithm in CUDA

- Step 2: For all negative reduced costs find the paths
- Step 3: Find edge disjoint paths among the ones obtained in 2
- Step 4: Perfrom flow adjustment on the paths
*/
__host__ void perform_a_parallel_pivot(PivotHandler &pivot, PivotTimer &timer, 
    Graph &graph, float * d_costs_ptr, bool &result, int numSupplies, int numDemands, int iteration, int &num_pivots) {
    
    // Find index of most negative reduced cost negative reduced cost >>

    auto _pivot_start = std::chrono::high_resolution_clock::now();
    auto _pivot_end = std::chrono::high_resolution_clock::now();
    auto _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);

    // Discover Cycles
    _pivot_start = std::chrono::high_resolution_clock::now();

	// Make a copy of adjacency matrix to make depth
    // IDEA: run my_signum all at once to get rid of that in the floyd warshall kernel - insted of memcpy run a kernel
    // Start with 
    //      all path values = -1 
    // 	    if there's an edge A->B : path transform A->B == 1 else INF and diagnoal elements zero
	thrust::fill(thrust::device, pivot.d_pathMtx, pivot.d_pathMtx + numSupplies*graph.V, -1);
    thrust::fill(thrust::device, pivot.d_adjMtx_transform, pivot.d_adjMtx_transform + numSupplies*numDemands, INT16_MAX);
    
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(1.0*numDemands/blockSize), ceil(1.0*numSupplies/blockSize),1);
    
    initialize_parallel_pivot <<< dimGrid, dimBlock >>> (pivot.d_bfs_frontier_current, 
            graph.d_vertex_start, &graph.d_vertex_degree[1], graph.d_adjVertices,
            d_costs_ptr, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(pivot.opportunity_cost, d_costs_ptr, sizeof(float)*numSupplies*numDemands, cudaMemcpyDeviceToDevice));

    // length of initial frontier (2*edges) - remember this is a vertex frontier
    // The BFS happens on the same tree from multiple source vertices
    int length_of_frontier = graph.V-1; 
    const int zero = 0;
    int iteration_number = 0;

    gpuErrchk(cudaMemcpy(pivot.current_frontier_length, &length_of_frontier, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pivot.next_frontier_length, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int this_blockSize = parallelBFSBlock;
    dim3 dimBFSBlock(this_blockSize, 1, 1);

    // BFS Kernel that updates the path Matrix and adjMatrix_transform //
    while (length_of_frontier > 0) {

        // debug_viewCurrentFrontier(pivot.d_bfs_frontier_current, length_of_frontier);

        dim3 dimBFSGrid(length_of_frontier, 1, 1);

        // SM PROFILING >> LOAD IMBALANCE CHECKING 
        // unsigned long long int * sm_profile;
        // gpuErrchk(cudaMalloc((void **) &sm_profile, sizeof(unsigned long long int)*68));
        // gpuErrchk(cudaMemset(sm_profile, 0, sizeof(unsigned long long int)*68));

        update_distance_path_and_create_next_frontier<<< dimBFSGrid, dimBFSBlock >>>(pivot.d_pathMtx, pivot.d_adjMtx_transform, 
            graph.d_vertex_start, &graph.d_vertex_degree[1], graph.d_adjVertices, 
            pivot.d_bfs_frontier_current, pivot.d_bfs_frontier_next,
            pivot.current_frontier_length, pivot.next_frontier_length,
            d_costs_ptr, pivot.opportunity_cost, 
            numSupplies, numDemands, iteration_number);
        //  sm_profile
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        // Post iteration >>
        iteration_number++;
        
        // Swap next and current >>
        swap_frontier(pivot.d_bfs_frontier_current, pivot.d_bfs_frontier_next);

        // Reset Next Length >>
        swap_lengths(pivot.current_frontier_length, pivot.next_frontier_length);
        gpuErrchk(cudaMemcpy(pivot.next_frontier_length, &zero, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(&length_of_frontier, pivot.current_frontier_length, sizeof(int), cudaMemcpyDeviceToHost));

        // std::cout<<"Length of Frontier = "<<length_of_frontier<<std::endl;
        // debug_viewCurrentFrontier(pivot.d_bfs_frontier_current, length_of_frontier);
        // debug_viewOpportunity_costs(pivot, d_costs_ptr, numSupplies, numDemands);
        // exit(0);
        
        // SM PROFILING >> LOAD IMBALANCE CHECKING 
        // unsigned long long int * h_sm_profile = (unsigned long long int *) malloc(sizeof(unsigned long long int)*68);
        // cudaMemcpy(h_sm_profile, sm_profile, sizeof(unsigned long long int)*68, cudaMemcpyDeviceToHost);
        // if (iteration_number==100) {
        //     for (int k = 0; k < 68; k++) {
        //         std::cout<<"SM["<<k<<"] = "<<h_sm_profile[k]<<std::endl;
        //     }
        //     exit(0);
        // }
        // cudaFree(sm_profile);
        // free(h_sm_profile);

    }

    
    // DEBUG UTILITY : view path output >>
    // _debug_print_APSP(pivot.d_adjMtx_transform, pivot.d_pathMtx, numSupplies, numDemands);
    // debug_viewOpportunity_costs(pivot, d_costs_ptr, numSupplies, numDemands);
    // exit(0);

    int h_num_NegativeCosts = 0;
    int diameter;

    gpuErrchk(cudaMemcpy(pivot.d_num_NegativeCosts, &h_num_NegativeCosts, sizeof(int), cudaMemcpyHostToDevice));

    // Get the diameter of tree  and number of potential pivots and allocate memory for storing cycles

    // Figure-out number of negative reduced costs    
    // Populate d_reduced_cost_ptr for this operation
    collectNegativeReducedCosts<<< dimGrid, dimBlock >>> (pivot.d_reducedCosts_ptr, pivot.d_num_NegativeCosts, pivot.opportunity_cost, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(&h_num_NegativeCosts, pivot.d_num_NegativeCosts, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_num_NegativeCosts > 0) {

        // Find diameter of graph >> 
        
        int * diameter_ptr = thrust::max_element(thrust::device, pivot.d_adjMtx_transform, pivot.d_adjMtx_transform + numSupplies*numDemands);
        gpuErrchk(cudaMemcpy(&diameter, diameter_ptr, sizeof(int), cudaMemcpyDeviceToHost));
        // diameter += 3;
        // std::cout<<"\tDiameter = "<<diameter<<std::endl;
        
        // Allocate memory for storing cycles >> 
        gpuErrchk(cudaMalloc((void **) &pivot.d_pivot_cycles, h_num_NegativeCosts*(diameter)*sizeof(int)));
        
        dim3 dimGrid3(ceil(1.0*h_num_NegativeCosts/blockSize), 1, 1);
        dim3 dimBlock3(blockSize, 1, 1);

        // expand_all_cycles<<< dimGrid3, dimBlock3 >>>(pivot.d_pivot_cycles, 
        //             pivot.d_adjMtx_transform, pivot.d_pathMtx, d_reducedCosts_ptr, 
        //             d_num_NegativeCosts, diameter, numSupplies, numDemands); 
        
        derive_cells_on_paths <<< dimGrid3, dimBlock3 >>> (pivot.d_pivot_cycles, 
                    pivot.d_adjMtx_transform, pivot.d_pathMtx, pivot.d_reducedCosts_ptr, 
                    pivot.d_num_NegativeCosts, diameter, numSupplies, numDemands);    
        
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // DEBUG UTILITY : view cycles in expanded form >>
        // _debug_view_discovered_cycles(pivot, d_reducedCosts_ptr, diameter, h_num_NegativeCosts, numSupplies, numDemands);
        // exit(0);

    }
    else {
        
        result = true;
        std::cout<<"Pivoting Complete!"<<std::endl;
        return;

    }
    
    _pivot_end = std::chrono::high_resolution_clock::now();
    _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    timer.cycle_discovery += _pivot_duration.count();

    // DEBUG : View computed opportunity costs
    _pivot_start = std::chrono::high_resolution_clock::now();

    // Get most opportunistic flow index - 
    bool search_complete = false; 
    int deconflicted_cycles_count = 0;
    // std::cout<<"Num Negative R Costs :"<<h_num_NegativeCosts<<std::endl;

    while (!search_complete) {

        // std::cout<<min_reduced_cost.cost<<std::endl;
        // Check if any cycles still remain untouched and if yes then get the best one

        MatrixCell min_opportunity_cost;
        int min_indx = thrust::min_element(thrust::device,
                pivot.d_reducedCosts_ptr, pivot.d_reducedCosts_ptr + (h_num_NegativeCosts), compareCells()) - pivot.d_reducedCosts_ptr;
        gpuErrchk(cudaMemcpy(&min_opportunity_cost, &pivot.d_reducedCosts_ptr[min_indx], sizeof(MatrixCell), cudaMemcpyDeviceToHost));

        if (min_opportunity_cost.cost >= 0) {
            search_complete = true;
        }

        else {

            pivot.deconflicted_cycles[deconflicted_cycles_count] = min_indx;
            min_opportunity_cost.cost = epsilon;
            gpuErrchk(cudaMemcpy(&pivot.d_reducedCosts_ptr[min_indx].cost, &min_opportunity_cost.cost, sizeof(float), cudaMemcpyHostToDevice));
        
            cudaDeviceProp prop;
            gpuErrchk(cudaGetDeviceProperties(&prop, 0));
            int num_strides = 1; //ceil(1.0*h_num_NegativeCosts/prop.maxGridSize[1]);

            dim3 cf_Block(resolveBlockSize, 1, 1);
            dim3 cf_Grid(h_num_NegativeCosts, ceil(1.0*diameter/resolveBlockSize), 1);

            check_pivot_feasibility <<< cf_Grid, cf_Block >>> (pivot.d_reducedCosts_ptr, min_indx,
                            min_opportunity_cost.row, min_opportunity_cost.col, 
                            pivot.d_adjMtx_transform, pivot.d_pivot_cycles,
                            diameter,numSupplies, numDemands);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
            deconflicted_cycles_count++;
        }

    }
    
    // std::cout<<"Found "<<deconflicted_cycles_count<<" deconflicted cycles to be pivoted"<<std::endl;

    _pivot_end = std::chrono::high_resolution_clock::now();
    _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    timer.resolve_time += _pivot_duration.count();

    // Now we have the indexes of feasible pivots in the array - deconflicted_cycles
    // Get these cycles on host and perform the pivot. To achieve this we do following
    // 1. Copy feasible cycles data to host memory 
    // 2. One by one perform the pivots

    _pivot_start = std::chrono::high_resolution_clock::now();

    MatrixCell pivot_Cell;

    for (int i=0; i < deconflicted_cycles_count; i++) {
        
        int pivot_indx = pivot.deconflicted_cycles[i];
        gpuErrchk(cudaMemcpy(&pivot_Cell, &pivot.d_reducedCosts_ptr[pivot_indx], sizeof(MatrixCell), cudaMemcpyDeviceToHost));
        
        int pivot_row = pivot_Cell.row;
        int pivot_col = pivot_Cell.col;
        
        gpuErrchk(cudaMemcpy(pivot.deconflicted_cycles_depth, &pivot.d_adjMtx_transform[pivot_row*numDemands + pivot_col], sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(pivot.deconflicted_cycles_backtracker, &pivot.d_pivot_cycles[pivot_indx*diameter], (*pivot.deconflicted_cycles_depth)*sizeof(int), cudaMemcpyDeviceToHost));

        int * this_backtracker = (int *) malloc(sizeof(int)*(*pivot.deconflicted_cycles_depth + 2));
        this_backtracker[0] = pivot_row;
        int from_vtx, to_vtx;
        
        // Parse edges >>

        for (int i = 0; i < *pivot.deconflicted_cycles_depth; i++) {
            
            int edge = pivot.deconflicted_cycles_backtracker[i];
            
            if (i%2 == 0) {
                to_vtx = edge/numDemands;
                from_vtx = edge - (to_vtx*numDemands) + numSupplies;
            }
            
            else {
                from_vtx = edge/numDemands;
                to_vtx = edge - (from_vtx*numDemands) + numSupplies;
            }

            this_backtracker[i+1] = from_vtx;
            this_backtracker[i+2] = to_vtx;

        }

        execute_pivot_on_host_device(graph.h_adjMtx_ptr, graph.h_flowMtx_ptr, graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, 
                    this_backtracker, pivot_row, pivot_col, *(pivot.deconflicted_cycles_depth) + 1, 
                    graph.V, numSupplies, numDemands);
        
    }

    _pivot_end = std::chrono::high_resolution_clock::now();
    _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    timer.adjustment_time += _pivot_duration.count();
    num_pivots = deconflicted_cycles_count;

    // Free memory allocated for cycles
    gpuErrchk(cudaFree(pivot.d_pivot_cycles));
}

} // End of namespace

