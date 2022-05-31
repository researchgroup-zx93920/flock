#include "PIVOT_ss.h"

namespace SS_METHOD {
/* 
Setup necessary resources for pivoting 
these resources are static and to be shared/overwritten between iterations
*/
__host__ void pivotMalloc(PivotHandler &pivot, int numSupplies, int numDemands) {

    int V = numSupplies + numDemands;

    if (PIVOTING_STRATEGY=="sequencial_dfs") {
        std::cout<<"sequencial DFS - Not Implemented for Stepping Stone Method, try parallel_fw!"<<std::endl;
        exit(-1);
    }
    else if (PIVOTING_STRATEGY=="parallel_dfs") {
        std::cout<<"parallel DFS - Not Implemented for Stepping Stone Method, try parallel_fw!"<<std::endl;
        exit(-1);
    }

    else if (PIVOTING_STRATEGY == "parallel_fw") {

        // Allocate Resources for floydwarshall cycle discovery strategy
        gpuErrchk(cudaMalloc((void **) &pivot.d_adjMtx_transform, V*V*sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &pivot.d_pathMtx, V*V*sizeof(int)));
        
        // Allocate mem for pivoting items
        gpuErrchk(cudaMalloc((void **) &pivot.opportunity_cost, numSupplies*numDemands*sizeof(float)));
        
        // Storing indexes of deconflicted cycles >>
        pivot.deconflicted_cycles = (int *) malloc(MAX_DECONFLICT_CYCLES(numSupplies, numDemands)*sizeof(int)); // upper bound = M + N - 1/3
        // Allocate appropriate memory for executing pivots
        pivot.deconflicted_cycles_depth = (int *) malloc(sizeof(int)); // Store depth of each cycle
        // Each cycle has a size less than max possible diameter
        pivot.deconflicted_cycles_backtracker = (int *) malloc(V*sizeof(int));

    }
}

/* 
Free up acquired resources for pivoting on host device 
*/
__host__ void pivotFree(PivotHandler &pivot) {

    if (PIVOTING_STRATEGY == "parallel_fw")
    {
        // Free up space >>
        gpuErrchk(cudaFree(pivot.d_adjMtx_transform));
        gpuErrchk(cudaFree(pivot.d_pathMtx));
        gpuErrchk(cudaFree(pivot.opportunity_cost));
        free(pivot.deconflicted_cycles);
        free(pivot.deconflicted_cycles_backtracker);
        free(pivot.deconflicted_cycles_depth);

    }
}

}

/*
Replaces the exiting basic flow with entering non basic flow
Does the necessary adjustments on the variables on device memory
*/
__host__ static void exit_i_and_enter_j(int * d_adjMtx_ptr, float * d_flowMtx_ptr, int exit_src, int exit_dest, 
        int enter_src, int enter_dest, int min_flow_indx, float min_flow, int V) {
            
    int id;
    int null_value = 0;
    int new_value = min_flow_indx + 1;

    // Set value for exiting in d
    id = TREE_LOOKUP(exit_src, exit_dest, V);
    gpuErrchk(cudaMemcpy(&d_adjMtx_ptr[id], &null_value, sizeof(int), cudaMemcpyHostToDevice));

    // Set value for entering to the appropriate
    id = TREE_LOOKUP(enter_src, enter_dest, V);
    gpuErrchk(cudaMemcpy(&d_adjMtx_ptr[id], &new_value, sizeof(int), cudaMemcpyHostToDevice));

    // The flow would have become zero - update it again
    if (PARALLEL_PIVOTING_METHOD=="delta") {
        gpuErrchk(cudaMemcpy(&d_flowMtx_ptr[min_flow_indx], &min_flow, sizeof(float), cudaMemcpyHostToDevice));
    }
}

/*
Do a copy from new value to device pointer
*/
__host__ static void modify_flowMtx_on_device(float * d_flowMtx_ptr, int id, float new_value) {
    gpuErrchk(cudaMemcpy(&d_flowMtx_ptr[id], &new_value, sizeof(float), cudaMemcpyHostToDevice));
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

    for (int i=1; i<depth; i++) 
    {
        _from = backtracker[i];
        _to = backtracker[i+1];
        id = h_adjMtx_ptr[TREE_LOOKUP(_from, _to, V)] - 1;
        _flow = ((int) pow(-1, (int)i%2))*min_flow;
        h_flowMtx_ptr[id] += _flow;
        if (PARALLEL_PIVOTING_METHOD=="delta") {
            modify_flowMtx_on_device(d_flowMtx_ptr, id, h_flowMtx_ptr[id]);
        }
    }

    // Do the replacment between exiting i - entering j on both host and device
    // Remove edge
    id = TREE_LOOKUP(min_from, min_to, V);
    h_adjMtx_ptr[id] = 0;
    // Insert edge
    id = TREE_LOOKUP(pivot_row, pivot_col+ numSupplies, V);
    h_adjMtx_ptr[id] = min_flow_id + 1;
    // Update new flow 
    h_flowMtx_ptr[min_flow_id] = min_flow;

    // Communicate device about the removal and addition of an extry-exit variable pair
    exit_i_and_enter_j(d_adjMtx_ptr, d_flowMtx_ptr, 
        min_from, min_to, 
        pivot_row, pivot_col + numSupplies, 
        min_flow_id, min_flow, V);
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

__host__ static void _debug_print_APSP(int * d_adjMtx, int * d_pathMtx, int V) {

    int * h_adjMtx_copy = (int *) malloc(sizeof(int)*V*V);
    int * h_pathMtx = (int *) malloc(sizeof(int)*V*V);

    gpuErrchk(cudaMemcpy(h_adjMtx_copy, d_adjMtx, V*V*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_pathMtx, d_pathMtx, V*V*sizeof(int), cudaMemcpyDeviceToHost));
	
    std::cout<<" ********* Distances >>"<<std::endl;
    for (int i=0; i<V; i++) {
        std::cout<<i<<" : ";
        for (int j=0; j<V; j++) {
            std::cout<<h_adjMtx_copy[i*V + j]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<" ********* Path >>"<<std::endl;
    for (int i=0; i<V; i++) {
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

__host__ static void _debug_view_discovered_cycles(PivotHandler &pivot, int diameter, int numSupplies, int numDemands) {

    std::cout<<" ************* Viewing Expanded cycles!"<<std::endl;

    int V = numSupplies + numDemands;
    int simplex_gridDim = V*V;

    // Printing all cycles
    int * h_pivot_cycles = (int *) malloc(numSupplies*numDemands*(diameter)*sizeof(int));
    int * h_depth  = (int *) malloc(sizeof(int)*simplex_gridDim);

    gpuErrchk(cudaMemcpy(h_pivot_cycles, pivot.d_pivot_cycles, numSupplies*numDemands*(diameter)*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_depth, pivot.d_adjMtx_transform, sizeof(int)*simplex_gridDim, cudaMemcpyDeviceToHost));
    
    for(int u=0; u<numSupplies; u++) {
        for (int v=0; v<numDemands; v++) {
            int offset_1 = (u*numSupplies + v)*diameter;
            int offset_2 = u*V + (v + numSupplies);
            int depth = h_depth[offset_2]+1;
            std::cout<<"Path from From :"<<u<<" To :"<<v+numSupplies<<" | Length = "<<depth<<" ==> ";
            for (int d=0; d<=depth; d++) {
                std::cout<<h_pivot_cycles[offset_1+d]<<", ";
            }
            std::cout<<std::endl;
        }
    }

    free(h_depth);
    free(h_pivot_cycles);
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
        }
    }
}


namespace SS_METHOD {
/*
Step 1: Find all point to all points shortest distance with Floyd Warshall using naive implementation 
    of Floyd Warshall algorithm in CUDA

- Step 2: For all negative reduced costs find the paths
- Step 3: Find edge disjoint paths among the ones obtained in 2
- Step 4: Perfrom flow adjustment on the paths
*/
__host__ void perform_a_parallel_pivot_floyd_warshall(PivotHandler &pivot, PivotTimer &timer, 
    Graph &graph, float * d_costs_ptr, bool &result, int numSupplies, int numDemands, int iteration) {
    
    // Find index of most negative reduced cost negative reduced cost >>

    auto _pivot_start = std::chrono::high_resolution_clock::now();
    auto _pivot_end = std::chrono::high_resolution_clock::now();
    auto _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);

    // Discover Cycles
    int simplex_gridDim = graph.V*graph.V;

    _pivot_start = std::chrono::high_resolution_clock::now();

	// Make a copy of adjacency matrix to make depth
    // IDEA: run my_signum all at once to get rid of that in the floyd warshall kernel - insted of memcpy run a kernel
    // Start with 
    //      all path values = -1 
    // 	    if there's an edge A->B : path transform A->B == 1 else INF and diagnoal elements zero
	thrust::fill(thrust::device, pivot.d_pathMtx, pivot.d_pathMtx + simplex_gridDim, -1);
    
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(1.0*graph.V/blockSize),ceil(1.0*graph.V/blockSize),1);
    
    fill_adjMtx <<< dimGrid, dimBlock >>> (pivot.d_adjMtx_transform, graph.d_adjMtx_ptr, pivot.d_pathMtx, graph.V);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Initialize the grid and block dimensions here
    dim3 dimGrid2((graph.V - 1) / blockSize + 1, (graph.V - 1) / blockSize + 1, 1);
    dim3 dimBlock2(blockSize, blockSize, 1);

    // /* cudaFuncSetCacheConfig(_naive_fw_kernel, cudaFuncCachePreferL1); */
    for (int vertex = 0; vertex < graph.V; ++vertex) {
        _naive_floyd_warshall_kernel <<< dimGrid2, dimBlock2 >>> (vertex, graph.V, pivot.d_adjMtx_transform, pivot.d_pathMtx);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    // DEBUG UTILITY : view floyd warshall output >>
    // _debug_print_APSP(pivot.d_adjMtx_transform, pivot.d_pathMtx, graph.V);

    // std::cout<<"Finding diameter of graph"<<std::endl;
    // Get the diameter of tree and allocate memory for storing cycles
    // Find diameter of graph >> 
    int diameter;
    int * diameter_ptr = thrust::max_element(thrust::device, pivot.d_adjMtx_transform, pivot.d_adjMtx_transform + simplex_gridDim);
    gpuErrchk(cudaMemcpy(&diameter, diameter_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    diameter++; // Length of cycle = length of path + 1
    // std::cout<<"Diameter = "<<diameter<<std::endl;
    
    // Allocate memory for cycles
    gpuErrchk(cudaMalloc((void **) &pivot.d_pivot_cycles, numSupplies*numDemands*(diameter)*sizeof(int)));
    
    // std::cout<<"Running cycle expansion kernel"<<std::endl;

    dim3 dimGrid3(ceil(1.0*numDemands/blockSize), ceil(1.0*numSupplies/blockSize), 1);
    dim3 dimBlock3(blockSize, blockSize, 1);

    expand_all_cycles<<<dimGrid3, dimBlock3>>>(pivot.d_adjMtx_transform, pivot.d_pathMtx, pivot.d_pivot_cycles, diameter, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // DEBUG UTILITY : view cycles in expanded form >>
    // _debug_view_discovered_cycles(pivot, diameter, numSupplies, numDemands);
    
    if (PARALLEL_PIVOTING_METHOD=="delta") {

        compute_opportunity_cost_and_delta<<<dimGrid3, dimBlock3>>>(graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, d_costs_ptr, 
                            pivot.d_adjMtx_transform, pivot.d_pivot_cycles, pivot.opportunity_cost, 
                            diameter, numSupplies, numDemands);
    
    }
    
    else if (PARALLEL_PIVOTING_METHOD=="r") {

        compute_opportunity_cost<<<dimGrid3, dimBlock3>>>(graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, d_costs_ptr, 
                            pivot.d_adjMtx_transform, pivot.d_pivot_cycles, pivot.opportunity_cost, 
                            diameter, numSupplies, numDemands);
    
    } 

    else {
    
        std::cout<<"Invalid parallel pivoting method! - try [r] or [delta]"<<std::endl;
    
    }


    _pivot_end = std::chrono::high_resolution_clock::now();
    _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    timer.cycle_discovery += _pivot_duration.count();

    // DEBUG : View computed opportunity costs
    _pivot_start = std::chrono::high_resolution_clock::now();

    // Get most opportunistic flow index - 
    bool search_complete = false;
    int shared_mem_requirement = sizeof(int)*diameter; // allocated cycle length 
    int deconflicted_cycles_count = 0;

    while (!search_complete) {

        // std::cout<<min_reduced_cost.cost<<std::endl;
        // Check if any cycles still remain untouched and if yes then get the best one
        float min_opportunity_cost;
        int min_indx = thrust::min_element(thrust::device,
                pivot.opportunity_cost, pivot.opportunity_cost + (numSupplies*numDemands)) - pivot.opportunity_cost;
        gpuErrchk(cudaMemcpy(&min_opportunity_cost, &pivot.opportunity_cost[min_indx], sizeof(float), cudaMemcpyDeviceToHost));

        if (!(min_opportunity_cost < 0 && std::abs(min_opportunity_cost) > epsilon2)) {
            search_complete = true;
            if (deconflicted_cycles_count == 0) {
                result = true;
                std::cout<<"Pivoting Complete!"<<std::endl;
                return;
            }
        }

        pivot.deconflicted_cycles[deconflicted_cycles_count] = min_indx;
        check_pivot_feasibility<<< dimGrid3, dimBlock3, shared_mem_requirement>>>(pivot.d_adjMtx_transform, pivot.d_pivot_cycles,
                 pivot.opportunity_cost, min_indx, diameter, numSupplies, numDemands);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        deconflicted_cycles_count++;

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

    for (int i=0; i < deconflicted_cycles_count; i++) {
        
        int pivot_indx = pivot.deconflicted_cycles[i];
        int pivot_row = pivot_indx/numDemands;
        int pivot_col = pivot_indx - (pivot_row*numDemands);
        int offset_1 = pivot_row*numDemands + pivot_col; // to retrieve cycle
        int offset_2 = pivot_row*graph.V + (pivot_col+numSupplies); // to retrieve depth

        gpuErrchk(cudaMemcpy(pivot.deconflicted_cycles_depth, &pivot.d_adjMtx_transform[offset_2], sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(pivot.deconflicted_cycles_backtracker, &pivot.d_pivot_cycles[offset_1*diameter], ((*pivot.deconflicted_cycles_depth)+2)*sizeof(int), cudaMemcpyDeviceToHost));

        execute_pivot_on_host_device(graph.h_adjMtx_ptr, graph.h_flowMtx_ptr, graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, 
                    pivot.deconflicted_cycles_backtracker, pivot_row, pivot_col, *(pivot.deconflicted_cycles_depth) + 1, 
                    graph.V, numSupplies, numDemands);
        
    }

    _pivot_end = std::chrono::high_resolution_clock::now();
    _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    timer.adjustment_time += _pivot_duration.count();

    // Free memory allocated for cycles
    gpuErrchk(cudaFree(pivot.d_pivot_cycles));
}

}

