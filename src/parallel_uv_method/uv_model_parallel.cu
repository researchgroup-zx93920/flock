/*
DISCLAIMER : 
1. INTENTIONALLY AVOIDING THE USE OF BOOST LOGGER UNLESS THERE'S A BETTER ALTERNATIVE
2. Arrangement of Functions : Find the kernels used just above the corresponding method

TODO:
1. Testing methods for both kinds of DFS (CPU and GPU)
2. Efficient Data Structure for micro_DFS instead of adjMtx
3. Can avoid memcpy opertion of device_CostMatrix_ptr between constructor and vogel call
4. int to Unsigned int
5. Change adjacency matrix and flow matrix struct documentation
6. VOGEL is missing some flows in some specific degenerate cases !! - Need a diagnosis here!
*/

#include "uv_model_parallel.h"

/*
Kernel to convert float cost matrix to the MatrixCell objects
*/
__global__ void createCostMatrix(MatrixCell *d_costMtx, float * d_costs_ptr, int n_supplies, int n_demands)
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
Constructor - Maintaining consistancy across algorithm modes
*/
uvModel_parallel::uvModel_parallel(ProblemInstance *problem, flowInformation *flows)
{

    std::cout << std::endl <<"Initializing UV Model (parallel)" << std::endl;
    data = problem; // Get problem instance in this class
    optimal_flows = flows; // Flows are still on the CPU memory 

    // Allocate memory based on the problem size # >>
    
        // feasible_flows - holds the edges of the spanning tree
    feasible_flows = (flowInformation *) malloc(sizeof(flowInformation) * ((data->numSupplies) + (data->numDemands) - 1));
    
        // costMatrix - costMatrix in CPU
    costMatrix = (MatrixCell *) malloc(sizeof(MatrixCell) * (data->numSupplies) * (data->numDemands));
    
    // Creating cost Matrix Cells objects =>
    //  Following process - prepare costMatrix cells using a small GPU function and copy back to CPU
    cudaMalloc((void **) &d_costs_ptr, sizeof(float) * (data->numSupplies) * (data->numDemands));
    cudaMalloc((void **) &device_costMatrix_ptr, sizeof(MatrixCell) * (data->numSupplies) * (data->numDemands));
    cudaMemcpy(d_costs_ptr, data->costs, sizeof(float) * (data->numSupplies) * (data->numDemands), cudaMemcpyHostToDevice);

    dim3 _dimBlock(blockSize,blockSize,1);
    dim3 _dimGrid(ceil(1.0*data->numDemands/blockSize), ceil(1.0*data->numSupplies/blockSize), 1);
    createCostMatrix<<<_dimGrid, _dimBlock>>>(device_costMatrix_ptr, d_costs_ptr, data->numSupplies, data->numDemands);
    cudaDeviceSynchronize();
    cudaMemcpy(costMatrix, device_costMatrix_ptr, sizeof(MatrixCell) * (data->numSupplies) * (data->numDemands), cudaMemcpyDeviceToHost);
    /*
    for (int i = 0; i < data->numSupplies; i++)
    {
        for (int j = 0; j < data->numDemands; j++)
        {
            int _key = i * data->numDemands + j;
            std::cout<<costMatrix[_key]<<std::endl;
        }
    }
    */
    V = data->numSupplies+data->numDemands;
    std::cout << "An uv_model_parallel object was successfully created" << std::endl;
}

/*
Destructor - Low Prioirity Issue (Handle Later)
*/
uvModel_parallel::~uvModel_parallel()
{
    // On thrust layer - replace by thrust eqv. of free
    // free(costMatrix);
    // free(feasible_flows);
}

/*
Generate initial basic feasible solution using the selected method
This function populates the feasible_flows attribute of the parent class, that is subject
to updates by the subsequent improvement techniques
*/
void uvModel_parallel::generate_initial_BFS()
{

    // Data is available on the class objects - Call one of the IBFS methods on these
    auto start = std::chrono::high_resolution_clock::now();
    if (BFS_METHOD == "nwc")
    {
        // Approach 1: Northwest Corner (Naive BFS - sequential)
        // --------------------------------------------------------
        // Utilize NW Corner method to determine basic feasible solution, (uncomment below)
        find_nw_corner_bfs_seq(data->supplies, data->demands, costMatrix, feasible_flows, flow_indexes,
                               data->numSupplies, data->numDemands);
    }
    else if (BFS_METHOD == "vam")
    {
        // Approach 2: Vogel's Approximation - parallel
        // --------------------------------------------------------
        // Utilitze vogel's approximation to determine basic fesible solution using CUDA kernels
        find_vogel_bfs_parallel(data->supplies, data->demands, costMatrix, feasible_flows,
                                flow_indexes, data->numSupplies, data->numDemands);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double solution_time = duration.count();
    std::cout << BFS_METHOD << " BFS Found in : " << solution_time << " millisecs." << std::endl;
    // The array feasible flows at this stage holds the initial basic feasible solution
}

__global__ void copy_row_shadow_prices(Variable * U_vars, float * u_vars_ptr, int numSupplies) 
{    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < numSupplies) {
        u_vars_ptr[gid] = U_vars[gid].value;
    }
}

__global__ void copy_col_shadow_prices(Variable * V_vars, float * v_vars_ptr, int numDemands) 
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < numDemands) {
        v_vars_ptr[gid] = V_vars[gid].value;
    }
}

__global__ void initialize_U_vars(Variable * U_vars, int numSupplies) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    Variable default_var;
    if (gid < numSupplies) {
        U_vars[gid] = default_var;
    }
}

__global__ void initialize_V_vars(Variable * V_vars, int numDemands) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    Variable default_var;
    if (gid < numDemands) {
        V_vars[gid] = default_var;
    }
}

/*
Given a u and v vector on device - computes the dual costs for all the constraints. There could be multiple ways 
to solve the dual costs for a given problem. Packaged method derive u's and v's and load them onto 
the U_vars and V_vars attributes
    1. Use a tree method for dual (trickle down approach)
    2. Use a off-the-shelf solver for linear equation 
*/
void uvModel_parallel::solve_uv()
{
    /* 
    solve_uv method - Need to populate u_vars_ptr and v_vars_ptr attributes of this class 
    This is an internal function can't be API'ed - Executes successfully in a specific situation
    */

    // Initialize u and v and then solve them using the adj matrix provided
    if (CALCULATE_DUAL=="tree") 
    {
        
        initialize_U_vars<<<ceil(1.0*data->numSupplies/blockSize), blockSize>>>(U_vars, data->numSupplies);
        initialize_V_vars<<<ceil(1.0*data->numDemands/blockSize), blockSize>>>(V_vars, data->numDemands);
        
        // Set u[0] = 0 on device >> // This can be done more smartly - low prioirity
        Variable default_variable;
        default_variable.assigned = true;
        cudaMemcpy(U_vars, &default_variable, sizeof(Variable), cudaMemcpyHostToDevice);
        
        // Perform the assignment
        dim3 __blockDim(blockSize, blockSize, 1); 
        dim3 __gridDim(ceil(1.0*data->numDemands/blockSize), ceil(1.0*data->numSupplies/blockSize), 1);
        for (int i=0; i<(data->numSupplies+data->numDemands-1); i++) {
            assign_next<<< __gridDim, __blockDim >>> (d_adjMtx_ptr, d_costs_ptr, 
                U_vars, V_vars, data->numSupplies, data->numDemands);
            cudaDeviceSynchronize(); // There might be some performance degrade here
        }

        // Once done - copy the final values to u_vars_ptr and v_vars_ptr and free device memory
        // This one dumps the unnecessary data associated with equation solve
        copy_row_shadow_prices<<<ceil(1.0*data->numSupplies/blockSize), blockSize>>>(U_vars, u_vars_ptr, data->numSupplies);
        copy_row_shadow_prices<<<ceil(1.0*data->numDemands/blockSize), blockSize>>>(V_vars, v_vars_ptr, data->numDemands);
        cudaDeviceSynchronize();
    }
    else 
    {
        std::cout<<"Invalid method of dual calculation!"<<std::endl;
        std::exit(-1); 
    }

}

/*
Kernel to compute Reduced Costs in the transportation table
*/
__global__ void computeReducedCosts(float * u_vars_ptr, float * v_vars_ptr, float * d_costs_ptr, float * d_reducedCosts_ptr, 
    int numSupplies, int numDemands) 
{

        int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

        if (row_indx < numSupplies && col_indx < numDemands) {
            // r =  C_ij - (u_i + v_j);
            float r = d_costs_ptr[row_indx*numDemands+col_indx] - u_vars_ptr[row_indx] - v_vars_ptr[col_indx];
            d_reducedCosts_ptr[row_indx*numDemands+col_indx] = r;
        }
}

/*
Pretty generic method to compute reduced 
costs provided a cost-matrix and u_vars, v_vars and cost Matrix on device
*/
void uvModel_parallel::get_reduced_costs() 
{
    // std::cout<<"\t\tComputing Reduced Costs ..."<<std::endl;
    dim3 __dimBlock(blockSize, blockSize, 1); // Refine this based on device query
    dim3 __dimGrid(ceil(1.0*data->numDemands/blockSize), ceil(1.0*data->numSupplies/blockSize), 1);
    computeReducedCosts<<< __dimGrid, __dimBlock >>>(u_vars_ptr, v_vars_ptr, d_costs_ptr, d_reducedCosts_ptr, 
        data->numSupplies, data->numDemands);
    cudaDeviceSynchronize();
    // std::cout<<"\t\tComputing Reduced Costs - complete!"<<std::endl;
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

/* 
Pivoting Operation in Trnasport Simplex. A pivot is complete in following 3 Steps
    Step 1: Check if already optimal 
    Step 2: If not, Traverse tree and find a route (using DFS)
    Step 3: Perform the pivot and adjust the flow
    Step 4/0: Repeat!
*/
void uvModel_parallel::perform_pivot(bool &result) 
{
    // *******************************************
    // STEP 1: Check if already optimal
    // *******************************************
    
    // view_uvra();

    // Find the position of the most negative reduced cost >>
    int min_index = thrust::min_element(thrust::device, d_reducedCosts_ptr, 
            d_reducedCosts_ptr + (data->numSupplies*data->numDemands)) - d_reducedCosts_ptr;
    // have all the reduced costs in the d_reducedCosts_ptr on device
    float min_reduced_cost = 0;
    cudaMemcpy(&min_reduced_cost, d_reducedCosts_ptr + min_index, sizeof(float), cudaMemcpyDeviceToHost);

    if (!(min_reduced_cost >= 0))
    {
        // Found a negative reduced cost >>
        if (PIVOTING_STRATEGY == "sequencial") {
        
            // pivot row and pivot col are declared private attributes
            pivot_row =  min_index/data->numDemands;
            pivot_col = min_index - (pivot_row*data->numDemands);
            
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
            execute_pivot_on_host(h_adjMtx_ptr, h_flowMtx_ptr, d_adjMtx_ptr, d_flowMtx_ptr, backtracker,
            pivot_row, pivot_col, depth, V, data->numSupplies, data->numDemands);

            free(backtracker);
            free(stack);
            free(visited);
        }

        else if (PIVOTING_STRATEGY == "parallel") {
            
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
                
            }

            else {
                std::cout<<"No independent cycles found!"<<std::endl;
            }
        }
        else 
        {
          std::cout<<"Invalid selection of pivoting strategy"<<std::endl;
          exit(-1);
        }
    }
    else 
    {
        result = true;
        std::cout<<"Pivoting Complete!"<<std::endl;
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
        int numSupplies, int numDemands) 
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

void uvModel_parallel::execute() 
{
    
    // **************************************
    // Finding BFS >>
    // **************************************
    generate_initial_BFS();

    // **************************************
    // Modified Distribution Method (u-v method) - parallel (improve the BFS solution)
    // **************************************
    
    // STEP 1 : Allocate relevant memory and transfer the necessary to GPU Memory >>
    // **************************************
    std::cout<<"SIMPLEX PASS 1 :: creating the necessary data structures on global memory"<<std::endl;

        // 1.1 Create and Initialize u and v varaibles 
    cudaMalloc((void **) &u_vars_ptr, sizeof(float)*data->numSupplies);
    cudaMalloc((void **) &v_vars_ptr, sizeof(float)*data->numDemands);
    cudaMalloc((void **) &d_reducedCosts_ptr, sizeof(float)*data->numSupplies*data->numDemands);
    std::cout<<"\tGenerated shadow price vectors ..."<<std::endl;

        // 1.2 Transfer flows on device and prepare an adjacency and flow matrix >>
    int _utm_entries = (V*(V+1))/2; // Number of entries in upped triangular matrix 
    cudaMalloc((void **) &d_adjMtx_ptr, sizeof(int)*_utm_entries); 
    thrust::fill(thrust::device, d_adjMtx_ptr, d_adjMtx_ptr + _utm_entries, 0);

    cudaMalloc((void **) &d_flowMtx_ptr, sizeof(float)*(V-1));
    thrust::fill(thrust::device, d_flowMtx_ptr, d_flowMtx_ptr + (V-1), 0);

    cudaMalloc((void **) &d_flows_ptr, sizeof(flowInformation)*(V-1));
    cudaMemcpy(d_flows_ptr, feasible_flows, sizeof(flowInformation)*(V-1), cudaMemcpyHostToDevice);

        // 1.3 Small kernel to parallely create a tree using the flows
    create_initial_tree <<< ceil(1.0*(V-1)/blockSize), blockSize >>> (d_flows_ptr, d_adjMtx_ptr, d_flowMtx_ptr, 
                            data->numSupplies, data->numDemands);
    std::cout<<"\tGenerated initial tree ..."<<std::endl;
    // Now device_flows are useless; All information about graph is now contained within d_adjMatrix, d_flowMatrix on device =>
    cudaFree(d_flows_ptr);
    
    // **********************************************************
    // Pre-Process Operation before pivoting 
    // **********************************************************
    // Initialize empty u and v equations using the Variable Data Type >>
    if (CALCULATE_DUAL=="tree") {
        cudaMalloc((void **) &U_vars, sizeof(Variable)*data->numSupplies);
        cudaMalloc((void **) &V_vars, sizeof(Variable)*data->numSupplies);
    }

    // In case of sequencial pivoting - one would need a copy of adjMatrix on the host to traverse the graph
    // IMPORTANT: The sequencial function should ensure that the change made on host must be also made on device
    // if (PIVOTING_STRATEGY == "sequencial") {
    
    h_adjMtx_ptr = (int *) malloc(sizeof(int)*(_utm_entries));
    cudaMemcpy(h_adjMtx_ptr, d_adjMtx_ptr, sizeof(int)*(_utm_entries), cudaMemcpyDeviceToHost);
    
    h_flowMtx_ptr = (float *) malloc(sizeof(float)*(V-1));
    cudaMemcpy(h_flowMtx_ptr, d_flowMtx_ptr, sizeof(float)*(V-1), cudaMemcpyDeviceToHost);
    
    // }
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

    /* STEP 2 : SIMPLEX IMPROVEMENT 
    // **************************************
        LOOP THORUGH - 
            2.1 Use the current tree on device and solve u's and v's
            2.2 Compute Reduced costs
                2.3.1 If no - negative reduced costs - break the loop
                2.3.2 If there exist negative reduced costs -
                    Perform a pivot operation (more details on the corresponding function)
    */
    bool result = false;
    int iteration_counter = 0;
    auto start = std::chrono::high_resolution_clock::now();

    // STEP 2 Implemented Here ->
    std::cout<<"SIMPLEX PASS 2 :: find the dual -> reduced -> pivots -> repeat!"<<std::endl;
    result = false;
    while ((!result) && iteration_counter < MAX_ITERATIONS) {
        // std::cout<<"Iteration :"<<iteration_counter<<std::endl;

        // 2.1 
        solve_uv();  
        // NOTE:: This method cannot be called before this step because of memory allocations above
        // u_vars_ptr and v_vars ptr were populated on device

        // 2.2 
        get_reduced_costs();
        // d_reducedCosts_ptr was populated on device
        
        // DEBUG :: 
        // view_uvra();

        // 2.3
        perform_pivot(result);

        // view_uvra();

        iteration_counter++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	double solution_time = duration.count();
    std::cout<<"\tSimplex completed in : "<<solution_time<<" millisecs. and "<<iteration_counter<<" iterations."<<std::endl;

    std::cout<<"SIMPLEX PASS 3 :: Clearing the device memory and transfering the necessary data on CPU"<<std::endl;
    
    // Recreate device flows using the current adjMatrix
    if (CALCULATE_DUAL=="tree") {
        cudaFree(U_vars);
        cudaFree(V_vars);
    }
    
    cudaFree(u_vars_ptr);
    cudaFree(v_vars_ptr);
    cudaFree(d_reducedCosts_ptr);

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

    flowInformation default_flow;
    default_flow.qty = 0;

    cudaMalloc((void **) &d_flows_ptr, sizeof(flowInformation)*(data->numSupplies*data->numDemands));
    thrust::fill(thrust::device, d_flows_ptr, d_flows_ptr + (data->numSupplies*data->numDemands), default_flow);

    dim3 __blockDim(blockSize, blockSize, 1);
    int grid_size = ceil(1.0*(V)/blockSize); // VxV grid
    dim3 __gridDim(grid_size, grid_size, 1);
    retrieve_final_tree <<< __gridDim, __blockDim >>> (d_flows_ptr, d_adjMtx_ptr, d_flowMtx_ptr, 
        data->numSupplies, data->numDemands);
    cudaDeviceSynchronize();
    
    // Copy the (flows > 0) back on the host >>
    auto flow_end = thrust::remove_if(thrust::device,
        d_flows_ptr, d_flows_ptr + (data->numSupplies*data->numDemands), is_zero());
    int flow_count = flow_end - d_flows_ptr;
    std::cout<<"\tFound "<<flow_count<<" active flows in the final result"<<std::endl;
    data->active_flows = flow_count;
    cudaMemcpy(feasible_flows, d_flows_ptr, (data->active_flows)*sizeof(flowInformation), cudaMemcpyDeviceToHost);

    double objval = 0.0;
    for (int i=0; i< (data->active_flows); i++){
        int _row = feasible_flows[i].source;
        int _col = feasible_flows[i].destination;
        int _key = _row*data->numDemands + _col;
        objval += feasible_flows[i].qty*data->costs[_key];
    }
    data->totalFlowCost = objval;
    std::cout<<"Objective Value = "<<objval<<std::endl;

}

void uvModel_parallel::create_flows()
{
    memcpy(optimal_flows, feasible_flows, (data->active_flows)*sizeof(flowInformation));
}


/*
For Debugging purposes view - u, v, r_costs and adjMatrix >> 
*/
void uvModel_parallel::view_uvra() 
{

    std::cout<<"Viewing U, V, R and adjMatrix \n *******************************"<<std::endl;
    
    // Print reduced costs
    float * h_reduced_costs;
    h_reduced_costs = (float *) malloc(data->numDemands*data->numSupplies*sizeof(float));
    cudaMemcpy(h_reduced_costs, d_reducedCosts_ptr, data->numDemands*data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < data->numDemands*data->numSupplies; i++) {
        std::cout << "ReducedCosts[" << i << "] = " << h_reduced_costs[i] << std::endl;
    }

    std::cout<<"*****************************"<<std::endl;

    // Print U >>
    float * u_vector;
    u_vector = (float *) malloc(data->numSupplies*sizeof(float));
    cudaMemcpy(u_vector, u_vars_ptr, data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < data->numSupplies; i++) {
        std::cout << "U[" << i << "] = " << u_vector[i] << std::endl;
    }

    std::cout<<"*****************************"<<std::endl;

    // Print V >>
    float * v_vector;
    v_vector = (float *) malloc(data->numDemands*sizeof(float));
    cudaMemcpy(v_vector, v_vars_ptr, data->numDemands*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < data->numDemands; i++) {
        std::cout << "V[" << i << "] = " << v_vector[i] << std::endl;
    }

    std::cout<<"*****************************"<<std::endl;

    // Print adjMatrix & flowMatrix >>
    int * h_adjMtx;
    float * h_flowMtx;
    int _V = data->numSupplies+data->numDemands;
    
    h_adjMtx = (int *) malloc(((_V*(_V+1))/2)*sizeof(int));
    cudaMemcpy(h_adjMtx, d_adjMtx_ptr, ((_V*(_V+1))/2)*sizeof(int), cudaMemcpyDeviceToHost);

    h_flowMtx = (float *) malloc((_V-1)*sizeof(float));
    cudaMemcpy(h_flowMtx, d_flowMtx_ptr, (_V-1)*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < _V; i++) {
        for (int j = i; j < _V; j++) {
                int indx = TREE_LOOKUP(i, j, _V);
                if (h_adjMtx[indx] > 0) {
                    std::cout << "adjMatrix[" << i << "]["<< j << "] = "<<indx<<" = "<< h_adjMtx[indx]<< std::endl;
            }
        }
    }

    std::cout<<"*****************************"<<std::endl;

    for (int i = 0; i < _V-1; i++) {
        std::cout << "flowMatrix[" << i << "] = " << h_flowMtx[i] << std::endl;
    }

    // // Useless later >> 
    // if (depth > 0)
    // {
    //     for (int i=0; i <= depth; i++)
    //     {
    //     std::cout<<"Loop : "<<backtracker[i]<<std::endl;
    //     }
    // }
    // else {
    //     std::cout<<"No Cycle was detected!"<<std::endl;
    // }
    // exit(0);

    // Finish the flow pivoting >>
    // _from = pivot_row;
    // _to = backtracker[depth];
    // id = _from*V + _to;
    // _flow = -1*min_flow;
    // h_adjMtx_ptr[id] += _flow;
    // modify_adjMtx_on_device(d_adjMtx_ptr, id, h_adjMtx_ptr[id]);
    // id = _to*V + _from;
    // h_adjMtx_ptr[id] += _flow;
    // modify_adjMtx_on_device(d_adjMtx_ptr, id, h_adjMtx_ptr[id]);

}