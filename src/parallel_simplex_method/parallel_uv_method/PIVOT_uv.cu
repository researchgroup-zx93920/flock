#include "PIVOT_uv.h"

namespace UV_METHOD {
/* 
Setup necessary resources for pivoting 
these resources are static and to be shared/overwritten between iterations
*/
__host__ void pivotMalloc(PivotHandler &pivot, int numSupplies, int numDemands, char * pivoting_strategy) {

    int V = numSupplies + numDemands;

    // Pivoting requires some book-keeping (for the DFS procedure)
    // BOOK 1: Stores the routes discovered for each thread -
    pivot.backtracker = (int *) malloc(sizeof(int)*V);    
    // BOOK 2: Stores the runtime stack for DFS running on each thread
    pivot.stack = (stackNode *) malloc(sizeof(stackNode)*V);    
    // BOOK 3: Keeps a track if any vertex was visited during DFS for each thread
    pivot.visited = (bool *) malloc(sizeof(bool)*V);

    if (REDUCED_COST_MODE == "parallel") {
        gpuErrchk(cudaMalloc((void **) &pivot.opportunity_cost, sizeof(float)*numSupplies*numDemands));
    }

}

/* 
Free up acquired resources for pivoting on host device 
*/
__host__ void pivotFree(PivotHandler &pivot, char * pivoting_strategy) {

    // Free up resources acquired for sequencial DFS
    free(pivot.backtracker);
    free(pivot.stack);
    free(pivot.visited);

    if (REDUCED_COST_MODE == "parallel") {

        gpuErrchk(cudaFree(pivot.opportunity_cost));
    
    }

}

}

/*
Push a node in the provided stack
*/
__host__ static void stack_push(stackNode * stack, int &stack_top, int vtx, int depth)
{
    stack_top++;
    stackNode node = {.index = vtx, .depth = depth};
    stack[stack_top] = node;
}

/*
Pop a node from the provided stack
*/
__host__ static stackNode stack_pop(stackNode * stack, int &stack_top)
{
    stackNode vtx;
    vtx = stack[stack_top];
    stack_top--;
    return vtx;
}

/*
Perform depth first search looking for route to execute the pivot
*/
__host__ void perform_dfs_sequencial_on_i(int * adjMtx, int * vertex_start, int * vertex_degree, int * adjVertices, 
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

__host__ static void do_flow_adjustment_on_host(int * h_adjMtx_ptr, float * h_flowMtx_ptr, 
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

    }

    // Do the replacment between exiting i - entering j on host

    // Remove edge >>
    id = TREE_LOOKUP(min_from, min_to, V);
    h_adjMtx_ptr[id] = 0;
    // Insert edge >>
    id = TREE_LOOKUP(pivot_row, pivot_col+ numSupplies, V);
    h_adjMtx_ptr[id] = min_flow_id + 1;
    // Update new flow >>
    h_flowMtx_ptr[min_flow_id] = min_flow;

}

__host__ static void execute_pivot_on_host(int * h_adjMtx_ptr, float * h_flowMtx_ptr, 
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

namespace UV_METHOD {
/*
Pivoting Operation in Transport Simplex. A pivot is complete in following 3 Steps
    Step 1: Check if already optimal 
    Step 2: If not, Traverse tree and find a route (using DFS)
    Step 3: Perform the pivot and adjust the flow
    Step 4/0: Repeat!
*/
__host__ void perform_a_sequencial_pivot(PivotHandler &pivot, PivotTimer &timer,
    Graph &graph, bool &result, int numSupplies, int numDemands) {

    float min_reduced_cost = pivot.reduced_cost;
    
    if (min_reduced_cost < 0 && std::abs(min_reduced_cost) > epsilon2) {

        int pivot_row =  pivot.pivot_row;   // cell_index/numDemands;
        int pivot_col = pivot.pivot_col;    // cell_index - (pivot_row*numDemands);

        // Preprocess before sequencial pivot
        
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

            execute_pivot_on_host(graph.h_adjMtx_ptr, graph.h_flowMtx_ptr, 
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

}
