/*
DISCLAIMER : 
1. INTENTIONALLY AVOIDING THE USE OF BOOST LOGGER UNLESS THERE'S A BETTER ALTERNATIVE
2. Arrangement of Functions : Find the kernels used just above the corresponding method

TODO:
1. Testing methods for both kinds of DFS (CPU and GPU)
2. Efficient Data Structure for micro_DFS instead of adjMtx
3. Can avoid memcpy opertion of device_CostMatrix_ptr between constructor and vogel call
4. int to Unsigned int
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
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    double solution_time = duration.count();
    std::cout << BFS_METHOD << " BFS Found in : " << solution_time << " secs." << std::endl;
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
        Variable * U_vars, * V_vars;
        // Initialize empty u and v equations using the Variable Data Type >>
        cudaMalloc((void **) &U_vars, sizeof(Variable)*data->numSupplies);
        cudaMalloc((void **) &V_vars, sizeof(Variable)*data->numSupplies);
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
        cudaFree(U_vars);
        cudaFree(V_vars);
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

void stack_push(stackNode * stack, int &stack_top, int vtx, int depth)
{
    stack_top++;
    stackNode node = {.index = vtx, .depth = depth};
    stack[stack_top] = node;
}

stackNode stack_pop(stackNode * stack, int &stack_top)
{
    stackNode vtx;
    vtx = stack[stack_top];
    stack_top--;
    return vtx;
}

/* End of DFS */
void perform_dfs_sequencial_on_i(float * adjMtx, stackNode * stack, int * backtracker, 
        int &depth, int starting_vertex, int target_vertex, int V)
{   
    // Initialized visited flag
    bool * visited = (bool *) malloc(sizeof(bool)*V);
    for(int j=0; j<V; j++) {
        visited[j]=false;
    }

    int key, stack_top = -1;
    stackNode current_vertex;
    stack_push(stack, stack_top, starting_vertex, depth);

    while(!(stack_top == -1))
    {
        current_vertex = stack_pop(stack, stack_top);
        // std::cout<<"Current Vertex : "<<current_vertex.index<<std::endl;
        // std::cout<<"Depth : "<<current_vertex.depth<<std::endl;
        
        // check if current vtx has been already visited in this search
        if (!visited[current_vertex.index])
        {
            // if not visited: >> 
            //  - mark this as visited 
            //  - see if current_vertex is adj to the starting point, 
            //        if not - queue the vertices that are adjacent to current vertex, increment depth
            visited[current_vertex.index]=true;

            // Do the book-keeping
            backtracker[current_vertex.depth] = current_vertex.index;
            depth = current_vertex.depth + 1;

            // check if target point is adjacent
            key = target_vertex*V + current_vertex.index;
            if (adjMtx[key] > 0 && depth > 1)
            {
                // Leads back to origin - this completes the cycle - exit the loop
                // std::cout<<"Loop Breaks"<<std::endl;
                break;
            }
            else
            {
                // Append the ajacent nodes in stack
                for(int j=0; j < V; j++)
                {
                    key = current_vertex.index*V + j;
                    // queue neighbors
                    if(adjMtx[key] > 0)
                    {
                        stack_push(stack, stack_top, j, depth);
                    }
                }
                // // Increment depth
                // depth++;
            }
        }
        // else - move to next vertex : pop_next, Before that >>
        // Iterations have explored the childeren and now going up in the recursion tree 
        // to something that is still pending to be explored -
        if (stack_top == -1)
        {
            depth=1;
        }
    }
}

__host__ void modify_adjMtx_on_device(float * d_adjMtx_ptr, int id, float new_value) {
    // Do a copy from new value to device pointer >>
    cudaMemcpy(d_adjMtx_ptr + id, &new_value, sizeof(float), cudaMemcpyHostToDevice);
}

void uvModel_parallel::perform_pivot(bool &result) 
{
    // have all the reduced costs in the d_reducedCosts_ptr on device
    
    if (PIVOTING_STRATEGY == "sequencial") 
    {
        // Find the position of the most negative reduced cost >>
        
        int min_index = thrust::min_element(thrust::device, d_reducedCosts_ptr, 
            d_reducedCosts_ptr + (data->numSupplies*data->numDemands)) - d_reducedCosts_ptr;
        float min_reduced_cost = 0;
        cudaMemcpy(&min_reduced_cost, d_reducedCosts_ptr + min_index, sizeof(float), cudaMemcpyDeviceToHost);
        
        // std::cout<<"Minimum = "<<min_reduced_cost<<std::endl;
        // std::cout<<"Min-Index = "<<min_index<<std::endl;

        if (min_reduced_cost >= 0) 
        {
            result = true;
        }
        else
        {   
            // Found a negative reduced cost >>
            // pivot row and pivot col are declared private attributes
            pivot_row =  min_index/data->numDemands;
            pivot_col = min_index - (pivot_row*data->numDemands);
            
            // An incoming edge from vertex = pivot_row to vertex = numSupplies + pivot_col
            std::cout<<"Pivot Row = "<<pivot_row<<std::endl;
            std::cout<<"Pivot Col ="<<pivot_col<<std::endl; 
            
            int id;
            int backtracker[V];
            stackNode stack[V-1];
            
            backtracker[0] = pivot_row;
            int depth = 1;
            // may have to bring visited here - todo for Mohit
            // Attempt 1 : 
            perform_dfs_sequencial_on_i(h_adjMtx_ptr, stack, backtracker, depth, 
                pivot_col+data->numSupplies, pivot_row, V);
            
            // If still loop not discovered >>
            if (depth <= 1) {
                std::cout<<"Error : Degenerate pivot cannot be performed!"<<std::endl;
                std::cout<<"Solution NOT OPTIMAL!"<<std::endl;
                // view_uvra();
                // std::cout<<"From : "<<pivot_row<<" | To : "<<pivot_col+data->numSupplies<<std::endl;
                result = true;
                return;
            }

            // Traverse the loop find the minimum flow that could be increased
            // on the incoming edge
            // Look into adjacency matrix >>
            
            int _from = -1, _to = -1;
            float _flow, min_flow = INT_MAX;
            backtracker[depth] = pivot_row;

            // Performing the pivot operation >> 
            // Finding the minimum flow >>
            for (int i=0; i<depth; i++) 
            {
                if (i%2==1) 
                {
                    _from = backtracker[i];
                    _to = backtracker[i+1];
                    id = _from*V + _to;
                    _flow = h_adjMtx_ptr[id];
                    // if (_flow == min_flow){
                    //     std::cout<<"Tie!"<<std::endl;
                    //     std::cout<<"Min Flow: "<<min_flow<<std::endl;
                    //     exit(0);
                    // }
                    if (_flow < min_flow) 
                    {
                        min_flow = _flow;
                    }
                }
            }

            // Before 
            // view_uvra();

            // std::cout<<"min_flow : "<<min_flow<<std::endl;
            // for (int i=0; i<depth; i++) 
            // {
            //     _from = backtracker[i];
            //     _to = backtracker[i+1];
            //     std::cout<<"From : "<<_from<<" | To : "<<_to<<std::endl;
            // }

            // Executing the flow adjustment >>
            int j=1;
            for (int i=0; i<depth; i++) 
            {
                _from = backtracker[i];
                _to = backtracker[i+1];
                id = _from*V + _to;
                _flow = j*min_flow;
                h_adjMtx_ptr[id] += _flow;
                modify_adjMtx_on_device(d_adjMtx_ptr, id, h_adjMtx_ptr[id]);
                id = _to*V + _from;
                h_adjMtx_ptr[id] += _flow;
                modify_adjMtx_on_device(d_adjMtx_ptr, id, h_adjMtx_ptr[id]);
                j *= -1;
            }

            // After 
            // view_uvra();

        }
    }
}

/*
Generate a tree on the global memory using the initial set of feasible flows
*/
__global__ void create_initial_tree(flowInformation * d_flows_ptr, float * d_adjMtx_ptr, 
    int numSupplies, int numDemands) 
{
    
    int V = numSupplies+numDemands;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    if (gid < V - 1) {
    
        flowInformation _this_flow = d_flows_ptr[gid];
        int row = _this_flow.source;
        int column =  _this_flow.destination;
        float _qty = 1.0*_this_flow.qty;
        if (_qty==0){
            // Handling degeneracy - 
            _qty=epsilon;
        }
        d_adjMtx_ptr[row*V + (numSupplies + column)] = _qty;
        d_adjMtx_ptr[(column + numSupplies) * V + row] = _qty;

    }
}

/*
Reverse operation of generating a tree from the feasible flows - unordered allocation
*/
__global__ void retrieve_final_tree(flowInformation * d_flows_ptr, float * d_adjMtx_ptr, int numSupplies, int numDemands) 
{

    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies+numDemands;
    
    // Upper triangle scope of adj matrix
    if (col_indx < V && col_indx >= numSupplies && row_indx < numSupplies) {
        // Check if this is a flow edge - 
        int gid = row_indx*V + col_indx;
    
        if (d_adjMtx_ptr[gid] > 0) {

            int flow_indx = row_indx*numDemands + (col_indx - numSupplies);
            flowInformation _this_flow;
            _this_flow.qty = round(d_adjMtx_ptr[gid]);
            _this_flow.source = row_indx;
            _this_flow.destination = col_indx - numSupplies;
            d_flows_ptr[flow_indx] = _this_flow; 
        
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

        // 1.2 Transfer flows on device and prepare an adjacency matrix >>
    cudaMalloc((void **) &d_adjMtx_ptr, sizeof(float)*(V*V)); 
    cudaMemset(d_adjMtx_ptr, 0, sizeof(float)*(V*V));

    cudaMalloc((void **) &d_flows_ptr, sizeof(flowInformation)*(V-1));
    cudaMemcpy(d_flows_ptr, feasible_flows, sizeof(flowInformation)*(V-1), cudaMemcpyHostToDevice);

        // 1.3 Small kernel to parallely create a tree using the flows
    create_initial_tree <<< ceil(1.0*(V-1)/blockSize), blockSize >>> (d_flows_ptr, d_adjMtx_ptr, data->numSupplies, data->numDemands);
    std::cout<<"\tGenerated initial tree ..."<<std::endl;
    // Now device_flows are useless; All information about graph is now contained within d_adjMatrix on device =>
    cudaFree(d_flows_ptr);
    
    // **********************************************************
    // Pre-Process Operation before pivoting 
    // **********************************************************

    // In case of sequencial pivoting - one would need a copy of adjMatrix on the host to traverse the graph
    // IMPORTANT: The sequencial function should ensure that the change made on host must be also made on device
    if (PIVOTING_STRATEGY == "sequencial") {
        h_adjMtx_ptr = (float *) malloc(sizeof(float)*(V*V));
        cudaMemcpy(h_adjMtx_ptr, d_adjMtx_ptr, sizeof(float)*(V*V), cudaMemcpyDeviceToHost);
    }


    /* STEP 2 : SIMPLEX IMPROVEMENT 
    // **************************************
        LOOP THORUGH - 
            2.1 Use the current tree on device and solve u's and v's
            2.2 Compute Reduced costs
            2.3.1 If no - negative reduced costs - break the loop
            2.3.2 If there exist negative reduced costs -
                Perform a pivot operation
    */
    bool result = false;
    int iteration_counter = 0;
    auto start = std::chrono::high_resolution_clock::now();

    // STEP 2 Implemented Here ->
    std::cout<<"SIMPLEX PASS 2 :: find the dual -> reduced -> pivots -> repeat!"<<std::endl;
    result = false;
    while (!result) {
        std::cout<<"Iteration :"<<iteration_counter<<std::endl;

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
        iteration_counter++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	double solution_time = duration.count();
    std::cout<<"\tSimplex completed in : "<<solution_time<<" secs. and "<<iteration_counter<<" iterations."<<std::endl;

    std::cout<<"PASS 3 :: Clearing the device memory and transfering the necessary data on CPU"<<std::endl;
    // Recreate device flows using the current adjMatrix
    cudaFree(u_vars_ptr);
    cudaFree(v_vars_ptr);
    cudaFree(d_reducedCosts_ptr);

    flowInformation default_flow;
    default_flow.qty = 0;
    thrust::device_vector<flowInformation> device_flows(data->numSupplies*data->numDemands, default_flow);
    d_flows_ptr = thrust::raw_pointer_cast(device_flows.data());

    dim3 __blockDim(blockSize, blockSize, 1);
    int grid_size = ceil(1.0*(data->numDemands+data->numSupplies)/blockSize);
    dim3 __gridDim(grid_size, grid_size, 1);
    retrieve_final_tree <<< __gridDim, __blockDim >>> (d_flows_ptr, d_adjMtx_ptr, data->numSupplies, data->numDemands);
    
    // Life is good, well sometimes!
    thrust::device_vector<flowInformation>::iterator flows_end = thrust::remove_if(
        device_flows.begin(), device_flows.end(), is_zero());
    device_flows.resize(flows_end - device_flows.begin());
    std::cout<<"\tFound "<<device_flows.size()<<" active flows in the final result"<<std::endl;
    data->active_flows = device_flows.size();
    // Assuming M+N-1 edges still exist 
    cudaMemcpy(feasible_flows, d_flows_ptr, (data->active_flows)*sizeof(flowInformation), cudaMemcpyDeviceToHost);
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
    
    // // Print reduced costs
    // float * h_reduced_costs;
    // h_reduced_costs = (float *) malloc(data->numDemands*data->numSupplies*sizeof(float));
    // cudaMemcpy(h_reduced_costs, d_reducedCosts_ptr, data->numDemands*data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < data->numDemands*data->numSupplies; i++) {
    //     std::cout << "ReducedCosts[" << i << "] = " << h_reduced_costs[i] << std::endl;
    // }

    // std::cout<<"*****************************"<<std::endl;

    // // Print U >>
    // float * u_vector;
    // u_vector = (float *) malloc(data->numSupplies*sizeof(float));
    // cudaMemcpy(u_vector, u_vars_ptr, data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < data->numSupplies; i++) {
    //     std::cout << "U[" << i << "] = " << u_vector[i] << std::endl;
    // }

    // std::cout<<"*****************************"<<std::endl;

    // // Print V >>
    // float * v_vector;
    // v_vector = (float *) malloc(data->numDemands*sizeof(float));
    // cudaMemcpy(v_vector, v_vars_ptr, data->numDemands*sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < data->numDemands; i++) {
    //     std::cout << "V[" << i << "] = " << v_vector[i] << std::endl;
    // }

    // std::cout<<"*****************************"<<std::endl;

    // Print adjMatrix >>
    float * h_adjMtx;
    int _V = data->numSupplies+data->numDemands;
    h_adjMtx = (float *) malloc((_V*_V)*sizeof(float));
    cudaMemcpy(h_adjMtx, d_adjMtx_ptr, (_V*_V)*sizeof(float), cudaMemcpyDeviceToHost);
    int r = 0;
    for (int i = 0; i < _V; i++) {
        for (int j = 0; j < _V; j++) {
            if (h_adjMtx[i*_V +j] > 0) {
                std::cout << "adjMatrix[" << i << "]["<< j << "] = " << h_adjMtx[i*_V +j] << std::endl;
                r++;
            }
        }
    }
    std::cout<<"Found "<<r<<" flows > 0"<<std::endl;
    std::cout<<"*****************************"<<std::endl;

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