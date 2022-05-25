#include "PIVOT_dfs.h"

/* 
Setup necessary resources for pivoting 
these resources are static and to be shared/overwritten between iterations
*/
__host__ void pivotMalloc(PivotHandler &pivot, int numSupplies, int numDemands) {

    int V = numSupplies + numDemands;

    if (PIVOTING_STRATEGY=="sequencial_dfs") {

        // Pivoting requires some book-keeping (for the DFS procedure)
        // BOOK 1: Stores the routes discovered for each thread -
        pivot.backtracker = (int *) malloc(sizeof(int)*V);    
        // BOOK 2: Stores the runtime stack for DFS running on each thread
        pivot.stack = (stackNode *) malloc(sizeof(stackNode)*V);    
        // BOOK 3: Keeps a track if any vertex was visited during DFS for each thread
        pivot.visited = (bool *) malloc(sizeof(bool)*V);
    }
    
    else if (PIVOTING_STRATEGY == "parallel_dfs") {

        // Allocate appropriate resources, Specific to parallel pivot >>
        int num_threads_launching = NUM_THREADS_LAUNCHING(numSupplies, numDemands, PARALLEL_PIVOT_IDEA);
        // BOOK 1: Stores the routes discovered for each thread
        gpuErrchk(cudaMalloc((void **) &pivot.backtracker, num_threads_launching * V * sizeof(int)));
        // BOOK 2: Stores the runtime stack for DFS running on each thread
        gpuErrchk(cudaMalloc((void **) &pivot.stack, num_threads_launching * V * sizeof(stackNode)));
        // BOOK 3: Keeps a track if any vertex was visited during DFS for each thread
        gpuErrchk(cudaMalloc((void **) &pivot.visited, num_threads_launching * V * sizeof(bool)));
        // BOOK 4: Stores the length of path discovered by each thread through DFS
        gpuErrchk(cudaMalloc((void **) &pivot.depth, num_threads_launching * sizeof(int)));
        
        // Following is temporarily removed 
        // gpuErrchk(cudaMalloc((void **) &pivot.v_conflicts, numSupplies * numDemands * sizeof(vertex_conflicts)));
    }
}

/* 
Free up acquired resources for pivoting on host device 
*/
__host__ void pivotFree(PivotHandler &pivot) {

    if (PIVOTING_STRATEGY == "sequencial_dfs") {
        
        free(pivot.backtracker);
        free(pivot.stack);
        free(pivot.visited);
    
    }

    else if (PIVOTING_STRATEGY == "parallel_dfs")
    {
        // Free up space >>
        gpuErrchk(cudaFree(pivot.backtracker));
        gpuErrchk(cudaFree(pivot.stack));
        gpuErrchk(cudaFree(pivot.visited));
        gpuErrchk(cudaFree(pivot.depth));

    }
}

/*
Push a node in the provided stack
*/
__host__ __device__ void stack_push(stackNode * stack, int &stack_top, int vtx, int depth)
{
    stack_top++;
    stackNode node = {.index = vtx, .depth = depth};
    stack[stack_top] = node;
}

/*
Pop a node from the provided stack
*/
__host__ __device__ stackNode stack_pop(stackNode * stack, int &stack_top)
{
    stackNode vtx;
    vtx = stack[stack_top];
    stack_top--;
    return vtx;
}

/*
Perform depth first search looking for route to execute the pivot
*/
__host__ __device__ void perform_dfs_sequencial_on_i(int * adjMtx, int * vertex_start, int * vertex_degree, int * adjVertices, 
        stackNode * stack, int * backtracker, bool * visited, 
        int * depth, int starting_vertex, int target_vertex, int V)
{   
    
    int key, current_depth = 1, stack_top = -1;
    stackNode current_vertex;
    stack_push(stack, stack_top, starting_vertex, current_depth);

    while(!(stack_top == -1))
    {
        current_vertex = stack_pop(stack, stack_top);

        // check if current vtx has been already visited in this search
        if (!visited[current_vertex.index])
        {
            // if not visited: >> 
            //  - mark this as visited 
            //  - see if current_vertex is adj to the starting point, 
            //        if not - queue the vertices that are adjacent to current vertex, increment depth
            visited[current_vertex.index]=true;

            // Do the book-keeping
            current_depth = current_vertex.depth + 1;
            backtracker[current_vertex.depth] = current_vertex.index;

            // check if target point is adjacent
            key = TREE_LOOKUP(target_vertex, current_vertex.index, V);
            if (adjMtx[key] > 0 && current_depth > 1)
            {
                // Leads back to origin - this completes the cycle - exit the loop
                *depth = current_depth;
                break;
            }
            else
            {
                // Append the ajacent nodes in stack
                int _s = vertex_start[current_vertex.index];
                for (int j = _s; j < _s + vertex_degree[current_vertex.index]; j++)
                {
                    stack_push(stack, stack_top, adjVertices[j], current_depth);
                }
            }
            
        }
        // else - move to next vertex : pop_next, Before that >>
        // Iterations have explored the childeren and now going up in the recursion tree 
        // to something that is still pending to be explored -
        if (stack_top == -1)
        {
            *depth=1;
        }
    }
}


/*
Replaces the exiting basic flow with entering non basic flow
Does the necessary adjustments on the variables on device memory
*/
__host__ void exit_i_and_enter_j(int * d_adjMtx_ptr, float * d_flowMtx_ptr, int exit_src, int exit_dest, 
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
    // gpuErrchk(cudaMemcpy(&d_flowMtx_ptr[min_flow_indx], &min_flow, sizeof(float), cudaMemcpyHostToDevice));

}

/*
Do a copy from new value to device pointer
*/
__host__ void modify_flowMtx_on_device(float * d_flowMtx_ptr, int id, float new_value) {
    gpuErrchk(cudaMemcpy(&d_flowMtx_ptr[id], &new_value, sizeof(float), cudaMemcpyHostToDevice));
}

__host__ void do_flow_adjustment_on_host_device(int * h_adjMtx_ptr, float * h_flowMtx_ptr, 
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
        // modify_flowMtx_on_device(d_flowMtx_ptr, id, h_flowMtx_ptr[id]);
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

__host__ void execute_pivot_on_host_device(int * h_adjMtx_ptr, float * h_flowMtx_ptr, 
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

/*
Pivoting Operation in Transport Simplex. A pivot is complete in following 3 Steps
    Step 1: Check if already optimal 
    Step 2: If not, Traverse tree and find a route (using DFS)
    Step 3: Perform the pivot and adjust the flow
    Step 4/0: Repeat!
*/
__host__ void perform_a_sequencial_pivot(PivotHandler &pivot, PivotTimer &timer,
    Graph &graph, MatrixCell * d_reducedCosts_ptr, bool &result, int numSupplies, int numDemands) {

    MatrixCell min_reduced_cost;
    
    // Find index of most negative reduced cost negative reduced cost >>
    int min_indx = thrust::min_element(thrust::device,
                d_reducedCosts_ptr, d_reducedCosts_ptr + (numSupplies*numDemands), compareCells()) - d_reducedCosts_ptr;
    gpuErrchk(cudaMemcpy(&min_reduced_cost, &d_reducedCosts_ptr[min_indx], sizeof(MatrixCell), cudaMemcpyDeviceToHost));

    if (min_reduced_cost.cost < 0 && std::abs(min_reduced_cost.cost) > 10e-3) {

        int cell_index = min_reduced_cost.row*numDemands + min_reduced_cost.col;
        int pivot_row =  cell_index/numDemands;
        int pivot_col = cell_index - (pivot_row*numDemands);

        // Preprocess before sequencial pivot
        if (!(CALCULATE_DUAL=="host_bfs")) {
                
                // Copy Adjacency list on host - reuse the same struct for efficient DFS >> 
                gpuErrchk(cudaMemcpy(graph.h_vertex_degree, &graph.d_vertex_degree[1], sizeof(int)*graph.V, cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(graph.h_vertex_start, graph.d_vertex_start, sizeof(int)*graph.V, cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(graph.h_adjVertices, graph.d_adjVertices, sizeof(int)*2*(graph.V-1), cudaMemcpyDeviceToHost)); 

            }
        
        auto _pivot_start = std::chrono::high_resolution_clock::now();
        auto _pivot_end = std::chrono::high_resolution_clock::now();
        auto _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    
        // *******************************************
        // STEP: Traverse tree and find a cycle
        // *******************************************
        int _depth = 1; // Stores length of cycle discovered for each thread
        pivot.backtracker[0] = pivot_row;
        memset(pivot.visited, 0, graph.V*sizeof(bool));

        // Find a path by performing DFS from pivot_col reaching pivot row to complete cycle >>
        // SEQUENCIAL PROCEDURE to find An incoming edge to vertex = pivot_row from vertex = numSupplies + pivot_col        
        _pivot_start = std::chrono::high_resolution_clock::now();

        perform_dfs_sequencial_on_i(graph.h_adjMtx_ptr, graph.h_vertex_start, graph.h_vertex_degree, graph.h_adjVertices, 
            pivot.stack, pivot.backtracker, pivot.visited, &_depth, 
            pivot_col+numSupplies, pivot_row, graph.V);
    
        _pivot_end = std::chrono::high_resolution_clock::now();
        _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
        timer.cycle_discovery += _pivot_duration.count();

        // If loop not discovered, this is a foolproof check
        // Beacuse graph is a tree, this should not happen anytime 
        // BUT just in case u know this many not be your day :D
        _pivot_start = std::chrono::high_resolution_clock::now();

        if (_depth <= 1) {
            std::cout<<" !! Error !! : Pivot cannot be performed, this is probably not a tree but forest!"<<std::endl;
            std::cout<<"Solution IS NOT OPTIMAL!"<<std::endl;
            result = true;
            return;
        }
        
        // As expected cycle was discovered and stored in backtracker array
        else {
            // *******************************************
            // STEP : Performing the pivot operation 
            // *******************************************
            pivot.backtracker[_depth] = pivot_row;

            // std::cout<<"Printing Cycle :: [ ";
            // for (int i=0; i<= _depth; i++){
            //     std::cout<<pivot.backtracker[i]<<", ";         
            // }
            // std::cout<<"]"<<std::endl;
            // exit(0);

            execute_pivot_on_host_device(graph.h_adjMtx_ptr, graph.h_flowMtx_ptr, 
                    graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, 
                    pivot.backtracker, pivot_row, pivot_col, _depth, 
                    graph.V, numSupplies, numDemands);
        }
        
        _pivot_end = std::chrono::high_resolution_clock::now();
        _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
        timer.adjustment_time += _pivot_duration.count();
    }
    else
    {
        result = true;
        std::cout<<"Pivoting Complete!"<<std::endl;
        return;
    }
}

/*
KERNEL 1 =>
Parallel version of DFS on Device -
On a negative reduced cost cell find a alternating path that improves the objective function
*/
__global__ void find_loops(MatrixCell * d_reducedCosts_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr, 
        int * d_vertex_start, int * d_vertex_degree, int * d_adjVertices,
        stackNode * stack, bool * visited, int * backtracker, int * depth, int numSupplies, int numDemands, int bound) {

    int local_id = blockIdx.x*blockDim.x + threadIdx.x;
    MatrixCell c = d_reducedCosts_ptr[local_id];

    // Check bounds and if this reduced cost is negative
    // Bound is number of parallel pivots that need to be performed
    if (local_id < bound && c.cost < -10e-3 ) { 

        int V = numSupplies + numDemands;
        int offset = V * local_id;
        int local_row = c.row;
        int local_col = c.col;
        int _depth = 1;
        backtracker[offset] = local_row;
        
        // then pivot row is - local_row
        // and  pivot col is - local_col

        perform_dfs_sequencial_on_i(d_adjMtx_ptr, d_vertex_start, d_vertex_degree, d_adjVertices,
                &stack[offset], &backtracker[offset], 
                &visited[offset], &_depth, local_col + numSupplies, local_row, V);
    
        if (_depth > 1) {
    
            // A loop was found - complete the book-keeping
            backtracker[offset + _depth] = local_row;
    
            // Update depth and savings for referncing in subsequent kernel //
            depth[local_id] = _depth;
    
        }
        // else depth[local_id] = 0 (remains default)
    }
}


/*
Fetch and view all parallel discovered cycles 
Function: Copy depth, backtrack from device and print
*/
__host__ void __debug_utility_1(MatrixCell * d_reducedCosts_ptr, int * backtracker, int * depth,  
    int iteration, int numSupplies, int numDemands, int num_threads_launching) 
{
    std::cout<<"DEBUG UTIITY - 1 | Viewing Discovered Loops"<<std::endl;
    int V = numSupplies + numDemands;

    int * h_backtracker = (int *) malloc(num_threads_launching * V * sizeof(int));
    int * h_depth = (int *) malloc(num_threads_launching * sizeof(int));
    MatrixCell * h_reduced_costs = (MatrixCell *) malloc(num_threads_launching * sizeof(MatrixCell));

    int num_cycles = 0;
    
    cudaMemcpy(h_backtracker, backtracker, num_threads_launching * V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_depth, depth, num_threads_launching * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reduced_costs, d_reducedCosts_ptr, num_threads_launching * sizeof(MatrixCell), cudaMemcpyDeviceToHost);

    for (int i=0; i < num_threads_launching; i++) {
        int offset = V*i;
        if (h_depth[i] > 0) {
            std::cout<<"Iteration : "<<iteration<<" : Thread : "<<i<<" : Depth : "<<h_depth[i]<<" : ";
            for (int j = 0; j <= h_depth[i]; j++) {
                std::cout<<h_backtracker[offset+j]<<" ";
            }
            std::cout<<std::endl;
            // std::cout<<"\t Loop Minimum = "<<h_loop_minimum[i]<<" From :"<<h_loop_min_from[i]<<" To : "<<h_loop_min_to[i]<<std::endl;
            std::cout<<"\t Reduced Cost Row = "<<h_reduced_costs[i].row<<std::endl;
            std::cout<<"\t Reduced Cost Col = "<<h_reduced_costs[i].col<<std::endl;
            std::cout<<"\t Reduced Cost = "<<h_reduced_costs[i].cost<<std::endl;
            num_cycles++;
        }
    }

    free(h_backtracker);
    free(h_depth);
    free(h_reduced_costs);

    std::cout<<"\n"<<num_cycles<<" cycles were discovered!"<<std::endl;
    // *********************** END OF DEBUG UTILITY - 1 *************** //
}

/*
API to execute parallel pivot, this uses DFS for dicovering the loops and then
gets the loops to execute pivot on the host (dehati way) - ask Mohit why is this a dehati way

This is meant primarily for testing and deriving insights on parallel cycles
*/
__host__ void perform_a_parallel_pivot(PivotHandler &pivot, PivotTimer &timer, 
    Graph &graph, MatrixCell * d_reducedCosts_ptr, bool &result, int numSupplies, int numDemands, int iteration) {
    
    // Check if termination criteria achieved 
    // (lowest reduced cost is positive)
    MatrixCell min_reduced_cost;
    // have all the reduced costs in the d_reducedCosts_ptr on device
    thrust::sort(thrust::device,
            d_reducedCosts_ptr, d_reducedCosts_ptr + (numSupplies*numDemands), compareCells());
    gpuErrchk(cudaMemcpy(&min_reduced_cost, &d_reducedCosts_ptr[0], sizeof(MatrixCell), cudaMemcpyDeviceToHost));
    // Termination criteria achieved
    if (!(min_reduced_cost.cost < 0 && std::abs(min_reduced_cost.cost) > 10e-3)) {    
        result = true;
        std::cout<<"Pivoting Complete!"<<std::endl;
        return;
    }

    auto _pivot_start = std::chrono::high_resolution_clock::now();
    auto _pivot_end = std::chrono::high_resolution_clock::now();
    auto _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    
    /*
    Strategy is to execute multiple pivots at the same time
        Step 1 : Go to all the cells with negative reduced costs -> find the cycles (Discover cycles)
        Step 2 : Whatever cycles were discovered, get them on host
        Step 3 : Execute cycles one by one and discard the ones that conflict with the ones discovered earlier 
    */

    // Discover Cycles
    _pivot_start = std::chrono::high_resolution_clock::now();
    
    int num_threads_launching = NUM_THREADS_LAUNCHING(numSupplies, numDemands, PARALLEL_PIVOT_IDEA);

    dim3 __blockDim(blockSize, 1, 1);
    dim3 __gridDim(ceil(1.0*num_threads_launching/blockSize), 1, 1);
    
    // Set initial values
    thrust::fill(thrust::device, pivot.depth, pivot.depth + (num_threads_launching), 0);
    thrust::fill(thrust::device, pivot.visited, pivot.visited + (graph.V * num_threads_launching), false);

    find_loops <<<__gridDim, __blockDim>>> (d_reducedCosts_ptr, graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, // Lookups 
        graph.d_vertex_start, graph.d_vertex_degree, graph.d_adjVertices,
        pivot.stack, pivot.visited,  // Intermediates
        pivot.backtracker, pivot.depth, // book-keeping
        numSupplies, numDemands, num_threads_launching); // Params
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); 
        // xxxxxx - Barrier 1 - xxxxxx
    
    // DEBUG UTILITY 1 ::
    // __debug_utility_1(d_reducedCosts_ptr, pivot.backtracker, pivot.depth, iteration, 
    // numSupplies, numDemands, num_threads_launching);

    _pivot_end = std::chrono::high_resolution_clock::now();
    _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    timer.cycle_discovery += _pivot_duration.count();
    
    /* ******************************
        Multi-Pivot Method
    ******************************* */

    // Copy Discovered cycles to host and sequencially execute pivots on the host 
    // Making sure no edge is used twice // 
    bool * edge_visited = (bool *) malloc(numSupplies*numDemands*sizeof(bool)); 
    
    int * h_backtracker = (int *) malloc(num_threads_launching * graph.V * sizeof(int));
    int * h_depth = (int *) malloc(num_threads_launching * sizeof(int));
    MatrixCell * h_reduced_costs = (MatrixCell *) malloc(num_threads_launching * sizeof(MatrixCell));

    _pivot_start = std::chrono::high_resolution_clock::now();

    int num_cycles_pivoted = 0;
    thrust::fill(thrust::host, edge_visited, edge_visited + (numSupplies*numDemands), false);
    
    gpuErrchk(cudaMemcpy(h_backtracker, pivot.backtracker, num_threads_launching * graph.V * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_depth, pivot.depth, num_threads_launching * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_reduced_costs, d_reducedCosts_ptr, num_threads_launching * sizeof(MatrixCell), cudaMemcpyDeviceToHost));
    
    // In the running workflow, we start with most negative reduced cost and proceed thereafter 
    for (int i=0; i < num_threads_launching; i++) {

        int offset = graph.V*i;
        if (h_depth[i] > 0) {
            // check if all the edges are available >> 
            bool cycle_valid = true; 
            int _edge_from, _edge_to, _id;

            for (int j = 0; j <= h_depth[i]-1; j++) {
                
                _edge_from = h_backtracker[offset+j] - numSupplies*(j%2);
                _edge_to = h_backtracker[offset+j+1] - numSupplies*((j+1)%2);
                _id = (_edge_from*numDemands + _edge_to)*((j+1)%2) + (_edge_to*numDemands + _edge_from)*(j%2);
                cycle_valid = (cycle_valid && !(edge_visited[_id]));
                // No need to check further if already found an edge that has been used
                if (!cycle_valid) {
                    // std::cout<<"break"<<std::endl;
                    break;
                }
            }

            if (cycle_valid) {

                // Mark edges in thie cycles as used >>
                #pragma omp parallel
                #pragma omp for
                for (int j = 0; j <= h_depth[i]-1; j++) {
                
                    _edge_from = h_backtracker[offset+j] - numSupplies*(j%2);
                    _edge_to = h_backtracker[offset+j+1] - numSupplies*((j+1)%2);
                    _id = (_edge_from*numDemands + _edge_to)*((j+1)%2) + (_edge_to*numDemands + _edge_from)*(j%2);
                    edge_visited[_id] = true;
                
                }

                #pragma omp barrier

                execute_pivot_on_host_device(graph.h_adjMtx_ptr, graph.h_flowMtx_ptr, graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, 
                    &h_backtracker[offset], h_reduced_costs[i].row, h_reduced_costs[i].col, h_depth[i], 
                    graph.V, numSupplies, numDemands);

                num_cycles_pivoted++;

            }
        }
    }

    _pivot_end = std::chrono::high_resolution_clock::now();
    _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    timer.adjustment_time += _pivot_duration.count();

    free(h_backtracker);
    free(h_depth);
    free(h_reduced_costs);

    // End of multi-pivot
    // std::cout<<"Iteration : "<<iteration<<" | Number of cycles pivoted : "<<num_cycles_pivoted<<std::endl;

}


// ***********************************************************************

__host__ void _debug_print_APSP(int * d_adjMtx, int * d_pathMtx, int V) {

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
}

/*
Step 1: Find all point to all points shortest distance with Floyd Warshall using naive implementation 
    of Floyd Warshall algorithm in CUDA

- Step 2: For all negative reduced costs find the paths
- Step 3: Find edge disjoint paths among the ones obtained in 2
- Step 4: Perfrom flow adjustment on the paths
*/
__host__ void perform_a_parallel_pivot_floyd_warshall(PivotHandler &pivot, PivotTimer &timer, 
    Graph &graph, MatrixCell * d_reducedCosts_ptr, bool &result, int numSupplies, int numDemands, int iteration) {
    
    // Here no need to sort reduced costs - check for termination criteria

    MatrixCell min_reduced_cost;
    
    // Find index of most negative reduced cost negative reduced cost >>
    int min_indx = thrust::min_element(thrust::device,
                d_reducedCosts_ptr, d_reducedCosts_ptr + (numSupplies*numDemands), compareCells()) - d_reducedCosts_ptr;
    gpuErrchk(cudaMemcpy(&min_reduced_cost, &d_reducedCosts_ptr[min_indx], sizeof(MatrixCell), cudaMemcpyDeviceToHost));

    if (!(min_reduced_cost.cost < 0 && std::abs(min_reduced_cost.cost) > 10e-3)) {
    
        // Terminate
        result = true;
        std::cout<<"Pivoting Complete!"<<std::endl;
        return;
    
    }

    auto _pivot_start = std::chrono::high_resolution_clock::now();
    auto _pivot_end = std::chrono::high_resolution_clock::now();
    auto _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);

    // Discover Cycles
    int * d_adjMtx_transform, * d_pathMtx;
    int simplex_gridDim = graph.V*graph.V;

    // Allocate memory for floyd warshall
	gpuErrchk(cudaMalloc((void **) &d_adjMtx_transform, simplex_gridDim*sizeof(int)));
	gpuErrchk(cudaMalloc((void **) &d_pathMtx, simplex_gridDim*sizeof(int)));
	
    _pivot_start = std::chrono::high_resolution_clock::now();

	// Make a copy of adjacency matrix to make depth
    // IDEA: run my_signum all at once to get rid of that in the floyd warshall kernel - insted of memcpy run a kernel	
	thrust::fill(thrust::device, d_pathMtx, d_pathMtx + simplex_gridDim, -1);
    
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(1.0*graph.V/blockSize),ceil(1.0*graph.V/blockSize),1);
    
    fill_adjMtx <<< dimGrid, dimBlock >>> (d_adjMtx_transform, graph.d_adjMtx_ptr, d_pathMtx, graph.V);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Initialize the grid and block dimensions here
    dim3 dimGrid2((graph.V - 1) / blockSize + 1, (graph.V - 1) / blockSize + 1, 1);
    dim3 dimBlock2(blockSize, blockSize, 1);

    // /* cudaFuncSetCacheConfig(_naive_fw_kernel, cudaFuncCachePreferL1); */
    for (int vertex = 0; vertex < graph.V; ++vertex) {
        _naive_floyd_warshall_kernel <<< dimGrid2, dimBlock2 >>> (vertex, graph.V, d_adjMtx_transform, d_pathMtx);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    // DEBUG UTILITY >>
    // _debug_print_APSP(d_adjMtx_ptr_copy, d_pathMtx, V);

    _pivot_end = std::chrono::high_resolution_clock::now();
    _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    timer.cycle_discovery += _pivot_duration.count();

    std::cout<<"APSP Completed in "<<timer.cycle_discovery<<" microseconds"<<std::endl;
    exit(0);

}


/******************** DUMP ***********************

/*
Fetch and view v_owner and v_savings 
Function: Copy arrays from device and print
*/
__host__ void __debug_utility_2(vertex_conflicts * v_conflicts, int numSupplies, int numDemands)
{
    std::cout<<"DEBUG UTIITY - 2 | Viewing Loop Owners"<<std::endl;

    vertex_conflicts * h_v_savings = (vertex_conflicts *) malloc(numSupplies * numDemands * sizeof(vertex_conflicts));
    cudaMemcpy(h_v_savings, v_conflicts,  numSupplies * numDemands * sizeof(vertex_conflicts), cudaMemcpyDeviceToHost);

    for (int i=0; i < numSupplies; i++) {
        for (int j=0; j < numDemands; j++) {
            std::cout << "Edge - (" <<i<<", "<<j+numSupplies<< ") by Thread : " << h_v_savings[i*numDemands + j].ints[1]<< std::endl;
        }
    }
    // *********************** END OF DEBUG UTILITY - 2 *************** //
}

__host__ void __debug_utility_3(int * backtracker, int * depth, 
    int numSupplies, int numDemands, int num_threads_launching) 
{
    // // *********************** DEBUG UTILITY - 3 *************** //
    // // Fetch and view the loops that do not conflict and maximize savings 
    
    std::cout<<"DEBUG UTIITY - 3 | Viewing Non-Conflicting loops"<<std::endl;
    int V = numSupplies + numDemands;
    int * h_backtracker = (int *) malloc(num_threads_launching * V * sizeof(int));
    int * h_depth = (int *) malloc(num_threads_launching * sizeof(int));
    int num_cycles = 0;
    
    cudaMemcpy(h_backtracker, backtracker, num_threads_launching * V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_depth, depth, num_threads_launching * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i < num_threads_launching; i++) {
        int offset = V*i;
       if (h_depth[i] > 0){
            std::cout<<"Thread : "<<i<<" : Depth : "<<h_depth[i]<<" : ";
            for (int j = 0; j < h_depth[i]; j++) {
                std::cout<<h_backtracker[offset+j]<<" ";
            }
            std::cout<<std::endl;
           num_cycles++;
       }
    }

    free(h_backtracker);
    free(h_depth);

    std::cout<<"\n"<<num_cycles<<" non conflicting cycles were discovered!"<<std::endl;
    // *********************** END OF DEBUG UTILITY - 3 *************** //
}




/*
The Novel Conflict Selector >>
Reference: https://stackoverflow.com/questions/17411493/how-can-i-implement-a-custom-atomic-function-involving-several-variables 
*/
__device__ unsigned long long int atomicMinAuxillary(unsigned long long int* address, float val1, int val2)
{
    vertex_conflicts loc, loctest;
    loc.floats[0] = val1;
    loc.ints[1] = val2;
    loctest.ulong = *address;
    while (val1  < loctest.floats[0] || (val1 == loctest.floats[0] && val2 < loctest.ints[1])) {
        // condition and tie-braker (bland's rule)
        loctest.ulong = atomicCAS(address, loctest.ulong,  loc.ulong);
    } 
    return loctest.ulong;
}

/*
Kernel 2 :: Step 1
Resolve conflicts |  Search for vertices that fall under conflicting loops
*/
__global__ void resolve_conflicts_step_1(int * depth, int * backtracker, MatrixCell * d_reducedCosts_ptr, 
        vertex_conflicts * v_conflicts, int numSupplies, int numDemands, int bound) {
            
    int local_id = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies + numDemands;
    int offset = V * local_id;
    
    if (local_id < bound) {

        // Check if this is a cell along which pivoting is performing
        int _depth = depth[local_id];
        // Real loops exists on this edge (cell) -
        if (_depth > 0) {

            // Find Savings
            float r = d_reducedCosts_ptr[local_id].cost;
            int _edge_from, _edge_to, _id;
            // If this loop is pivoted then this is the savings you get 

            for (int i=0; i<_depth-1; i++) {

                _edge_from = backtracker[offset+i] - numSupplies*(i%2);
                _edge_to = backtracker[offset+i+1] - numSupplies*((i+1)%2);
                
                /* 
                Atomically make the comparison and assign

                Essentially the following is performed in an atomic sense
                if (_savings < v_savings[_vtx]) {
                    v_savings[_vtx] = _savings;
                    v_owner[_vtx] = local_id;
                } 
                */

                _id = (_edge_from*numDemands + _edge_to)*((i+1)%2) + (_edge_to*numDemands + _edge_from)*(i%2);
                atomicMinAuxillary(&(v_conflicts[_id].ulong), r, local_id);
            }
        }
    }
}

/*
Resolve conflicts | Step 2 : Kill threads => discard the loops => Set depth = 0
*/
__global__ void resolve_conflicts_step_2(int * depth, int * backtracker, vertex_conflicts * v_conflicts, 
        int numSupplies, int numDemands, int num_threads_launching) {
            
    int local_id = blockIdx.x*blockDim.x + threadIdx.x;
    int V = numSupplies + numDemands;
    int offset = V * local_id;
    
    if (local_id < num_threads_launching) {
        // Check if this is a cell along which pivoting is performing
        int _depth = depth[local_id];
        // Real loops exists on this edge (cell) -
        if (_depth > 0) {
            // Check continuity along all vertices in loop if v_owner is this thread itself >>
            bool _continuity = true;
            int _edge_from, _edge_to, _id, i=0;
            while (i < _depth-1 && _continuity) {
                _edge_from = backtracker[offset+i] - numSupplies*(i%2);
                _edge_to = backtracker[offset+i+1] - numSupplies*((i+1)%2);
                _id = (_edge_from*numDemands + _edge_to)*((i+1)%2) + (_edge_to*numDemands + _edge_from)*(i%2);
                _continuity = (_continuity && (v_conflicts[_id].ints[1] == local_id));
                i++;
            }

            if (!_continuity) { // Kill this thread in this case
                depth[local_id] = 0;
            }
        }
    }
}

/*
Kernel to execute the flow adjustments in parallel >>
*/
__global__ void run_flow_adjustments(int * d_adjMtx_ptr, float * d_flowMtx_ptr, int * depth, 
            int * backtracker, float * loop_minimum,
            int * loop_min_from, int * loop_min_to, int * loop_min_id,
            int numSupplies, int numDemands) {
            
    int local_row = blockIdx.y*blockDim.y + threadIdx.y;
    int local_col = blockIdx.x*blockDim.x + threadIdx.x;
    int local_id = local_row*numDemands + local_col;
    int V = numSupplies + numDemands;
    int offset = V * local_id;
    
    if (local_row < numSupplies && local_col < numDemands) {
        // Check if this is a cell along which pivoting is performing
        int _depth = depth[local_id];
        // Real loops exists on this edge (cell) -
        if (_depth > 0) {

            int _from, _to, id, j=-1, min_from = loop_min_from[local_id], 
                min_to = loop_min_to[local_id], min_flow_id = loop_min_id[local_id];
            float _flow, _min_flow = loop_minimum[local_id];

            for (int i=1; i<_depth; i++)
            {
                _from = backtracker[offset+i];
                _to = backtracker[offset+i+1];
                id = d_adjMtx_ptr[TREE_LOOKUP(_from, _to, V)] - 1;
                _flow = j*_min_flow;
                d_flowMtx_ptr[id] += _flow;
                j *= -1;
            }

            // Do the replacment between exiting i - entering j on both host and device
            // Remove edge
            id = TREE_LOOKUP(min_from, min_to, V);
            d_adjMtx_ptr[id] = 0;
            // Insert edge
            id = TREE_LOOKUP(local_row, local_col+numSupplies, V);
            d_adjMtx_ptr[id] = min_flow_id + 1;
            // Update new flow 
            d_flowMtx_ptr[min_flow_id] = _min_flow;
        }
    }
}

/*

// Resolve Conflicts >> 
    _pivot_start = std::chrono::high_resolution_clock::now();
    // std::cout<<"Parallel Pivoiting : Discovered Loops!"<<std::endl;
    // std::cout<<"Parallel Pivoiting : Resolving Conflicts | Running Step 1 (Discover conflicts) ..."<<std::endl;        
    vertex_conflicts _vtx_conflict_default;
    _vtx_conflict_default.floats[0] = FLT_MAX;
    _vtx_conflict_default.ints[1] = -1;
    thrust::fill(thrust::device, v_conflicts, v_conflicts + (numSupplies*numDemands), _vtx_conflict_default);

    resolve_conflicts_step_1 <<<__gridDim, __blockDim>>> (depth, backtracker, d_reducedCosts_ptr, 
        v_conflicts, numSupplies, numDemands, num_threads_launching);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
        // xxxxxx - Barrier 2 - xxxxxx

    // DEBUG UTILITY 2 ::
    // __debug_utility_2(v_conflicts, numSupplies, numDemands);
    
    // std::cout<<"Parallel Pivoiting : Completed Step 1 | Running Step 2 (Resolve Conflicts) ..."<<std::endl;
    resolve_conflicts_step_2 <<<__gridDim, __blockDim>>> (depth, backtracker, v_conflicts, numSupplies, numDemands, num_threads_launching);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // xxxxxx - Barrier 3 - xxxxxx

    // DEBUG UTILITY 3 ::
    __debug_utility_3(backtracker, depth, numSupplies, numDemands, num_threads_launching);
    exit(0);
    _pivot_end = std::chrono::high_resolution_clock::now();
    _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    resolve_time += _pivot_duration.count();

    // std::cout<<"Parallel Pivoiting : Conflicts Resolved | Running flow adjustments ..."<<std::endl;        
    // Check if any conflicting pivots still exist >>
    _pivot_start = std::chrono::high_resolution_clock::now();
    int _conflict_flag = thrust::reduce(thrust::device, depth, depth + (numSupplies*numDemands), 0);
        
    std::cout<<"THIS PIVOTING METHOD IS OUT DATED, TRY - hybrid!"<<std::endl;
    exit(-1);
    run_flow_adjustments <<<__gridDim, __blockDim>>> (d_adjMtx_ptr, d_flowMtx_ptr, depth, backtracker, loop_minimum, 
        loop_min_from, loop_min_to, loop_min_id, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
        // xxxxxx - Barrier 4 - xxxxxx

    _pivot_end = std::chrono::high_resolution_clock::now();
    _pivot_duration = std::chrono::duration_cast<std::chrono::microseconds>(_pivot_end - _pivot_start);
    adjustment_time += _pivot_duration.count();
*/
