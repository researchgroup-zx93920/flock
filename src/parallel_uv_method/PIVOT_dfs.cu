#include "PIVOT_dfs.h"


__host__ void Initialize_pivoting() {
    if (PIVOTING_STRATEGY == "parallel") {

        // Allocate appropriate resources >>
        int num_threads_launching = data->numSupplies*data->numDemands;
        cudaMalloc((void **) &backtracker, num_threads_launching * V * sizeof(int));
        cudaMalloc((void **) &depth, num_threads_launching * sizeof(int));
        cudaMalloc((void **) &loop_minimum, num_threads_launching * sizeof(float));
        cudaMalloc((void **) &loop_min_from, num_threads_launching * sizeof(int));
        cudaMalloc((void **) &loop_min_to, num_threads_launching * sizeof(int));
        cudaMalloc((void **) &loop_min_id, num_threads_launching * sizeof(int));
        cudaMalloc((void **) &stack, num_threads_launching * V * sizeof(stackNode));
        cudaMalloc((void **) &visited, num_threads_launching * V * sizeof(bool));
        cudaMalloc((void **) &v_conflicts, V * sizeof(vertex_conflicts));
        _vtx_conflict_default.floats[0] = FLT_MAX;
        _vtx_conflict_default.ints[1] = -1;
        std::cout<<"\tParallel Pivoting : Allocated Resources on Device"<<std::endl;
    
    }

}

__host__ void terminate_PIVOT() {
    if (PIVOTING_STRATEGY == "parallel")
     {
        // Free up space >>
        cudaFree(stack);
        cudaFree(visited);
        cudaFree(v_conflicts);
        cudaFree(backtracker);
        cudaFree(depth);
        cudaFree(loop_minimum);
        cudaFree(loop_min_from);
        cudaFree(loop_min_to);
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
__host__ __device__ void perform_dfs_sequencial_on_i(int * adjMtx, stackNode * stack, int * backtracker, bool * visited, 
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
                // std::cout<<"Loop Breaks"<<std::endl;
                *depth = current_depth;
                break;
            }
            else
            {
                // Append the ajacent nodes in stack
                for(int j=0; j < V; j++)
                {
                    key = TREE_LOOKUP(current_vertex.index, j, V);
                    // queue neighbors
                    if(adjMtx[key] > 0)
                    {
                        stack_push(stack, stack_top, j, current_depth);
                    }
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
Do a copy from new value to device pointer
*/
__host__ void modify_flowMtx_on_device(float * d_flowMtx_ptr, int id, float new_value) {
    cudaMemcpy(&d_flowMtx_ptr[id], &new_value, sizeof(float), cudaMemcpyHostToDevice);
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
    cudaMemcpy(&d_adjMtx_ptr[id], &null_value, sizeof(int), cudaMemcpyHostToDevice);

    // Set value for entering to the appropriate
    id = TREE_LOOKUP(enter_src, enter_dest, V);
    cudaMemcpy(&d_adjMtx_ptr[id], &new_value, sizeof(int), cudaMemcpyHostToDevice);

    // The flow would have become zero - update it again
    cudaMemcpy(&d_flowMtx_ptr[min_flow_indx], &min_flow, sizeof(float), cudaMemcpyHostToDevice);

}

/*
Parallel version of DFS on Device
*/
__global__ void find_loops_and_savings(float * d_reducedCosts_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr, 
        stackNode * stack, bool * visited, int * backtracker, int * depth, float * loop_minimum,
        int * loop_min_from, int * loop_min_to, int * loop_min_id,
        int numSupplies, int numDemands) {

    int local_row = blockIdx.y*blockDim.y + threadIdx.y;
    int local_col = blockIdx.x*blockDim.x + threadIdx.x;
    int local_id = local_row*numDemands + local_col;
    int V = numSupplies + numDemands;
    int offset = V * local_id;

    if (local_row < numSupplies && local_col < numDemands) {
        float r = d_reducedCosts_ptr[local_id];    
        // Check if this reduced cost is negative
        if (r < 0) {
            
            int _depth = 1;
            backtracker[offset + 0] = local_row;
            // then pivot row is - local_row
            // and  pivot col is - local_col
            perform_dfs_sequencial_on_i(d_adjMtx_ptr, &stack[offset], &backtracker[offset], 
                &visited[offset], &_depth, numSupplies + local_col, local_row, V);
            
            if (!(_depth <= 1)) {
                
                // A loop was found - complete the book-keeping
                backtracker[offset + _depth] = local_row;

                // Traverse the loop find the minimum flow that could be increased
                // on the incoming edge >>
                int id, _from = -1, _to = -1, _min_flow_id = -1, _min_from = -1, _min_to = -1;
                float _flow, min_flow = FLT_MAX;

                for (int i=0; i<_depth; i++) 
                {
                    if (i%2==1)
                    {
                        _from = backtracker[offset+i];
                        _to = backtracker[offset+i+1];
                        id = d_adjMtx_ptr[TREE_LOOKUP(_from, _to, V)] - 1;
                        _flow = d_flowMtx_ptr[id];
                        
                        if (_flow < min_flow) 
                        {
                            min_flow = _flow;
                            _min_flow_id = id;
                            _min_from = _from;
                            _min_to = _to;
                        }
                    }
                }

                // Update depth and savings for referncing in subsequent kernel //
                depth[local_id] = _depth;
                loop_minimum[local_id] = min_flow;
                loop_min_from[local_id] = _min_from;
                loop_min_to[local_id] = _min_to;
                loop_min_id[local_id] = _min_flow_id;
            }
            // Otherwise depth[local_id] = 0 (remains default)
        }
    }
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
Resolve conflicts | Step 1 : Search for vertices that fall under conflicting loops
*/
__global__ void resolve_conflicts_step_1(int * depth, int * backtracker, float * loop_minimum, float * d_reducedCosts_ptr, 
        vertex_conflicts * v_conflicts, int numSupplies, int numDemands) {
            
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

            // Find Savings
            float r = d_reducedCosts_ptr[local_id];
            // If this loop is pivoted then this is the savings you get
            float _savings = r*loop_minimum[local_id]; 
            int _vtx;

            for (int i=0; i<_depth; i++) {
                _vtx = backtracker[offset+i];
                // Atomically make the comparison and assign
                /* Essentially the following is performed in an atomic sense
                if (_savings < v_savings[_vtx]) {
                    v_savings[_vtx] = _savings;
                    v_owner[_vtx] = local_id;
                } 
                */   
                atomicMinAuxillary(&(v_conflicts[_vtx].ulong), _savings, local_id);
            }
        }
    }
}

/*
Resolve conflicts | Step 2 : Kill threads => discard the loops => Set depth = 0
*/
__global__ void resolve_conflicts_step_2(int * depth, int * backtracker, vertex_conflicts * v_conflicts, int numSupplies, int numDemands) {
            
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
            // Check continuity along all vertices in loop if v_owner is this thread itself >>
            bool _continuity = true;
            int _vtx, i=0;
            while (i < _depth && _continuity) {
                _vtx = backtracker[offset+i];
                _continuity = (_continuity && (v_conflicts[_vtx].ints[1] == local_id));
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

__host__ void do_flow_adjustment_on_host(int * h_adjMtx_ptr, float * h_flowMtx_ptr, 
        int * d_adjMtx_ptr, float * d_flowMtx_ptr, int * backtracker, float min_flow, int min_from, int min_to, int min_flow_id,
        int pivot_row, int pivot_col, int depth, int V, int numSupplies, int numDemands) {

    // std::cout<<"Depth : "<<depth<<" : ";
    // for (int j = 0; j < depth + 1; j++) {
    //     std::cout<<backtracker[j]<<" ";
    // }
    // std::cout<<std::endl;
    // std::cout<<"Min flow : "<<min_flow<<std::endl;
    // std::cout<<"Min from : "<<min_from<<std::endl;
    // std::cout<<"Min to : "<<min_to<<std::endl;
    // std::cout<<"Min id : "<<min_flow_id<<std::endl;
    // std::cout<<"Pivot Row : "<<pivot_row<<std::endl;
    // std::cout<<"Pivot Col : "<<pivot_col<<std::endl;

    int j=-1, _from, _to, id;
    float _flow;
    for (int i=1; i<depth; i++) 
    {
        _from = backtracker[i];
        _to = backtracker[i+1];
        id = h_adjMtx_ptr[TREE_LOOKUP(_from, _to, V)] - 1;
        _flow = j*min_flow;
        h_flowMtx_ptr[id] += _flow;
        modify_flowMtx_on_device(d_flowMtx_ptr, id, h_flowMtx_ptr[id]);
        j *= -1;
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

    exit_i_and_enter_j(d_adjMtx_ptr, d_flowMtx_ptr, 
        min_from, min_to, 
        pivot_row, pivot_col + numSupplies, 
        min_flow_id, min_flow, V);
}

__host__ void execute_pivot_on_host(int * h_adjMtx_ptr, float * h_flowMtx_ptr, 
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
    do_flow_adjustment_on_host(h_adjMtx_ptr, h_flowMtx_ptr, d_adjMtx_ptr, d_flowMtx_ptr, backtracker,
            min_flow, min_from, min_to, min_flow_id,
            pivot_row, pivot_col, depth, V, numSupplies, numDemands);

}

void perform_a_sequencial_pivot() {

    // An incoming edge from vertex = pivot_row to vertex = numSupplies + pivot_col
            
            // *******************************************
            // STEP 2: If not, Traverse tree and find a route
            // *******************************************

            // BOOK 1: Stores the routes discovered for each thread -
            int * backtracker = (int *) malloc(sizeof(int)*V);
            
            // BOOK 2: Stores the runtime stack for DFS running on each thread
            stackNode * stack = (stackNode *) malloc(sizeof(stackNode)*V);
            
            // BOOK 3: Keeps a track if any vertex was visited during DFS for each thread
            bool * visited = (bool *) malloc(sizeof(bool)*V);
            memset(visited, 0, V*sizeof(bool));

            // BOOK 4: Stores length of routes discovered for each thread
            int depth;

            // SEQUENCIAL PROCEDURE >>
            backtracker[0] = pivot_row;
            depth = 1;
            
            // Find a loop by performing DFS from pivot_col upto pivot row >>
            perform_dfs_sequencial_on_i(h_adjMtx_ptr, stack, backtracker, visited, &depth, 
                pivot_col+data->numSupplies, pivot_row, V);
            
            // If loop not discovered >>
            if (depth <= 1) {
                std::cout<<" !! Error !! : Degenerate pivot cannot be performed, this is probably not a tree but forest!"<<std::endl;
                std::cout<<"Solution IS NOT OPTIMAL!"<<std::endl;
                // view_uvra();
                // std::cout<<"From : "<<pivot_row<<" | To : "<<pivot_col+data->numSupplies<<std::endl;
                result = true;
                return;
            }

            backtracker[depth] = pivot_row;

            // *******************************************
            // STEP 3: Performing the pivot operation 
            // *******************************************
            // std::cout<<"Pivot Row : "<<pivot_row<<std::endl;
            // std::cout<<"Pivot Col : "<<pivot_col<<std::endl;

            execute_pivot_on_host(h_adjMtx_ptr, h_flowMtx_ptr, d_adjMtx_ptr, d_flowMtx_ptr, backtracker,
            pivot_row, pivot_col, depth, V, data->numSupplies, data->numDemands);

            free(backtracker);
            free(stack);
            free(visited);

}



void perform_a_parallel_pivot() {
    /*
            Strategy is to execute multiple pivots at the same time
            Resolve conflicts through a barrier
            
            KERNEL 1: Go to all the cells with negative reduced costs -> find the pivots -> evaluate savings
            KERNEL 2: 
            */

            dim3 __blockDim(blockSize, blockSize, 1);
            dim3 __gridDim(ceil(1.0*data->numDemands/blockSize), ceil(1.0*data->numSupplies/blockSize), 1);
    
            thrust::fill(thrust::device, depth, depth + data->numSupplies*data->numDemands, 0);
            thrust::fill(thrust::device, visited, visited + (V * data->numSupplies*data->numDemands), false);

            find_loops_and_savings <<<__gridDim, __blockDim>>> (d_reducedCosts_ptr, d_adjMtx_ptr, d_flowMtx_ptr, // Lookups 
                    stack, visited, backtracker,  // Intermediates
                    depth, loop_minimum,  // Outputs
                    loop_min_from, loop_min_to, loop_min_id, // Book-keeping sake
                    data->numSupplies, data->numDemands); // Params
            cudaDeviceSynchronize(); // xxxxxx - Barrier 1 - xxxxxx
            
            // *********************** DEBUG UTILITY - 1 *************** //
            // Fetch and view dicsovered cycles 
            // Function: Copy depth, backtrack from device and print
            
            // std::cout<<"DEBUG UTIITY - 1 | Viewing Discovered Loops"<<std::endl;
            // int num_threads_launching = data->numSupplies*data->numDemands;

            // int * h_backtracker = (int *) malloc(num_threads_launching * V * sizeof(int));
            // int * h_depth = (int *) malloc(num_threads_launching * sizeof(int));
            // float * h_loop_minimum = (float *) malloc(num_threads_launching * sizeof(float));
            // int * h_loop_min_from = (int *) malloc(num_threads_launching * sizeof(int));
            // int * h_loop_min_to = (int *) malloc(num_threads_launching * sizeof(int));
            // int * h_loop_min_id = (int *) malloc(num_threads_launching * sizeof(int));
            // h_reduced_costs = (float *) malloc(num_threads_launching * sizeof(float));

            // int num_cycles = 0;
            
            // cudaMemcpy(h_backtracker, backtracker, num_threads_launching * V * sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(h_depth, depth, num_threads_launching * sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(h_loop_minimum, loop_minimum, num_threads_launching * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(h_loop_min_from, loop_min_from, num_threads_launching * sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(h_loop_min_to, loop_min_to, num_threads_launching * sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(h_loop_min_id, loop_min_id, num_threads_launching * sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(h_reduced_costs, d_reducedCosts_ptr, num_threads_launching * sizeof(float), cudaMemcpyDeviceToHost);

            // for (int i=0; i < num_threads_launching; i++) {
            //     int offset = V*i;
            //     if (h_depth[i] > 0){
            //         std::cout<<"Thread - "<<i<<" : Depth : "<<h_depth[i]<<" : ";
            //         for (int j = 0; j <= h_depth[i]; j++) {
            //             std::cout<<h_backtracker[offset+j]<<" ";
            //         }
            //         std::cout<<std::endl;
            //         std::cout<<"\t Loop Minimum = "<<h_loop_minimum[i]<<" From :"<<h_loop_min_from[i]<<" To : "<<h_loop_min_to[i]<<std::endl;
            //         std::cout<<"\t Reduced Costs = "<<h_reduced_costs[i]<<std::endl;
            //         num_cycles++;
            //     }
            // }

            // free(h_backtracker);
            // free(h_depth);
            // free(h_loop_minimum);
            // free(h_loop_min_from);
            // free(h_loop_min_to);
            // free(h_loop_min_id);
            // free(h_reduced_costs);

            // std::cout<<"\n"<<num_cycles<<" cycles were discovered!"<<std::endl;

            // *********************** END OF DEBUG UTILITY - 1 *************** //

            // std::cout<<"Parallel Pivoiting : Discovered Loops!"<<std::endl;

            // Resolve Conflicts >>

            thrust::fill(thrust::device, v_conflicts, v_conflicts + V, _vtx_conflict_default);

            // std::cout<<"Parallel Pivoiting : Resolving Conflicts | Running Step 1 (Discover conflicts) ..."<<std::endl;
            
            resolve_conflicts_step_1 <<<__gridDim, __blockDim>>> (depth, backtracker, loop_minimum, d_reducedCosts_ptr, 
                    v_conflicts, data->numSupplies, data->numDemands);
            cudaDeviceSynchronize(); // xxxxxx - Barrier 2 - xxxxxx
            
            // *********************** DEBUG UTILITY - 2 *************** //
            // Fetch and view v_owner and v_savings 
            // Function: Copy arrays from device and print
            
            // std::cout<<"DEBUG UTIITY - 2 | Viewing Loop Owners"<<std::endl;

            // vertex_conflicts * h_v_savings = (vertex_conflicts *) malloc(V * sizeof(vertex_conflicts));
            // cudaMemcpy(h_v_savings, v_conflicts,  V * sizeof(vertex_conflicts), cudaMemcpyDeviceToHost);

            // for (int i=0; i < V; i++) {
            //     std::cout << "Vertex - " << i << " by Thread : " << h_v_savings[i].ints[1]<< std::endl;
            // }

            // *********************** END OF DEBUG UTILITY - 2 *************** //
            
            // std::cout<<"Parallel Pivoiting : Completed Step 1 | Running Step 2 (Resolve Conflicts) ..."<<std::endl;
            
            resolve_conflicts_step_2 <<<__gridDim, __blockDim>>> (depth, backtracker, v_conflicts, data->numSupplies, data->numDemands);
            cudaDeviceSynchronize(); // xxxxxx - Barrier 3 - xxxxxx

            // // *********************** DEBUG UTILITY - 3 *************** //
            // // Fetch and view the loops that do not conflict and maximize savings 
            
            // std::cout<<"DEBUG UTIITY - 3 | Viewing Non-Conflicting loops"<<std::endl;

            // int num_threads_launching2 = data->numSupplies*data->numDemands;
            // int * h_backtracker2 = (int *) malloc(num_threads_launching2 * V * sizeof(int));
            // int * h_depth2 = (int *) malloc(num_threads_launching2 * sizeof(int));
            // float * h_loop_minimum2 = (float *) malloc(num_threads_launching2 * sizeof(float));
            // int num_cycles2 = 0;
            
            // cudaMemcpy(h_backtracker2, backtracker, num_threads_launching2 * V * sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(h_depth2, depth, num_threads_launching2 * sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(h_loop_minimum2, loop_minimum, num_threads_launching2 * sizeof(float), cudaMemcpyDeviceToHost);

            // for (int i=0; i < num_threads_launching2; i++) {
            //     int offset2 = V*i;
            //    if (h_depth2[i] > 0){
            //         std::cout<<"Thread - "<<i<<" : Depth : "<<h_depth2[i]<<" : ";
            //         for (int j = 0; j < h_depth2[i]; j++) {
            //             std::cout<<h_backtracker2[offset2+j]<<" ";
            //         }
            //         std::cout<<std::endl;
            //         std::cout<<"\t Loop Minimum = "<<h_loop_minimum2[i]<<std::endl;
            //        num_cycles2++;
            //    }
            // }

            // free(h_backtracker2);
            // free(h_depth2);
            // free(h_loop_minimum2);

            // std::cout<<"\n"<<num_cycles2<<" non conflicting cycles were discovered!"<<std::endl;

            // *********************** END OF DEBUG UTILITY - 3 *************** //

            // std::cout<<"Parallel Pivoiting : Conflicts Resolved | Running flow adjustments ..."<<std::endl;
            
            // Check if any conflicting pivots exist
            int _conflict_flag = thrust::reduce(thrust::device, depth, depth + data->numSupplies*data->numDemands, 0);
            if (_conflict_flag > 0) {
                
                // METHOD 1 : RUN ADJUSTMENTS IN PARALLEL
                run_flow_adjustments <<<__gridDim, __blockDim>>> (d_adjMtx_ptr, d_flowMtx_ptr, depth, backtracker, loop_minimum, 
                    loop_min_from, loop_min_to, loop_min_id, data->numSupplies, data->numDemands);
                cudaDeviceSynchronize(); // xxxxxx - Barrier 4 - xxxxxx

                // METHOD 2 : RUN FLOW ADJUSTMENTS IN SEQ on host (for all independent loops)
                // int * _h_depth = (int *) malloc(data->numSupplies*data->numDemands * sizeof(int));
                // int * _h_backtracker = (int *) malloc(sizeof(int)*V);
                // float min_flow;
                // int min_from, min_to, min_flow_id;
                // cudaMemcpy(_h_depth, depth, data->numSupplies*data->numDemands * sizeof(int), cudaMemcpyDeviceToHost);
                
                // for (int i=0; i < (data->numSupplies*data->numDemands); i++) {
                //     if (_h_depth[i] > 0) {

                //         int offset = V*i;
                //         pivot_row =  i/data->numDemands;
                //         pivot_col = i - (pivot_row*data->numDemands);

                //         cudaMemcpy(_h_backtracker, &backtracker[offset], (_h_depth[i]+1)*sizeof(int), cudaMemcpyDeviceToHost);
                //         cudaMemcpy(&min_flow, &loop_minimum[i], sizeof(float), cudaMemcpyDeviceToHost);
                //         cudaMemcpy(&min_from, &loop_min_from[i], sizeof(int), cudaMemcpyDeviceToHost);
                //         cudaMemcpy(&min_to, &loop_min_to[i], sizeof(int), cudaMemcpyDeviceToHost);
                //         cudaMemcpy(&min_flow_id, &loop_min_id[i], sizeof(int), cudaMemcpyDeviceToHost);
                        
                //         do_flow_adjustment_on_host(h_adjMtx_ptr, h_flowMtx_ptr, d_adjMtx_ptr, d_flowMtx_ptr, _h_backtracker,
                //             min_flow, min_from, min_to, min_flow_id,
                //             pivot_row, pivot_col, _h_depth[i], V, data->numSupplies, data->numDemands);
                        
                //         // std::cout<<"Adjusted!"<<std::endl;
                //    }
                // }
                // free(_h_depth);
                // free(_h_backtracker);
        else 
        {
            std::cout<<"No independent cycles found!"<<std::endl;
        }
}