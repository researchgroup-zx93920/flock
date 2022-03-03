/*
DISCLAIMER : 
1. INTENTIONALLY AVOIDING THE USE OF BOOST LOGGER UNLESS THERE'S A BETTER ALTERNATIVE
2. Arrangement of Functions : Find the kernels used just above the corresponding method

TODO:
1. Testing methods for both kinds of DFS (CPU and GPU)
2. Efficient Data Structure for micro_DFS instead of adjMtx
3. Can avoid memcpy opertion of device_CostMatrix_ptr between constructor and vogel call
4. ...
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
Perform DFS on the graph and returns true if any back-edge is found in the graph
At a thread level this may be inefficient given the data structures used, thus 
another method exists for the thread specific DFS

__host__ bool modern_DFS(Graph const &graph, int v, std::vector<bool> &discovered, std::vector<int> &loop, int parent)
{

    discovered[v] = true; // mark the current node as discovered
 
    // do for every edge (v, w)
    for (int w: graph.adjList[v])
    {
        // if `w` is not discovered
        if (!discovered[w])
        {
            if (modern_DFS(graph, w, discovered, loop, v)) {
                loop.push_back(w);
                return true;
            }
        }
        // if `w` is discovered, and `w` is not a parent
        else if (w != parent)
        {
            // we found a back-edge (cycle -> v-w is a back edge)
            loop.push_back(w);
            return true;
        }
    }
    // No back-edges were found in the graph
    discovered[v] = false; // Reset
    return false;
}
*/

/* 
Perform parallel DFS at the thread level using adjacency matrix - this one makes a stack called loop 
that stores the traversal \n
Note that:  This is a device function, expected to be light on memory, for well connected graphs, 
this will be helped thorugh cache, but that needs to checked through experiments.

__device__ bool micro_DFS(int * visited, int * adjMtx, pathEdge * loop, int V, int i, int parent) {
    
    atomicAdd(&visited[i], 1); // perform atomic add on visited - i - so that other threads aren't exploring this
    // visited[i] = 1; // >> sequential testing/debugging

    // For every neighbor of i 
    for(int j=0; j<V; j++) {
        if (adjMtx[i*V + j]>0) {

            // If it is not visited by anybody else =>
            if(visited[j]==0) {
                // Check if there's a forward edge 
                pathEdge * _loop = (pathEdge *) malloc(sizeof(pathEdge));
                _loop->index = j;
                _loop->next = NULL;
                if (micro_DFS(visited, adjMtx, V, j, i, _loop)) {
                    // loop_push
                    loop->next = _loop;
                    return true;
                }
                else {
                    free(_loop);
                    // Jumps to return false; 
                }
            }
            // We found a backward edge
            else if (j != parent) {
                // loop_push
                pathEdge * _loop = (pathEdge *) malloc(sizeof(pathEdge));
                _loop->index = j;
                _loop->next = NULL;
                loop->next = _loop;
                return true;
            }
        }
    }
    
    atomicSub(&visited[i], 1); // Reset visited flag to let other threads explore
    // visited[i] = 0; // >> sequential testing/debugging
    return false;
}
*/

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

__global__ void copy_row_shadow_prices(Variable * U_vars, float * u_vars_ptr, int numSupplies) {    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < numSupplies) {
        u_vars_ptr[gid] = U_vars[gid].value;
    }
}

__global__ void copy_col_shadow_prices(Variable * V_vars, float * v_vars_ptr, int numDemands) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < numDemands) {
        v_vars_ptr[gid] = V_vars[gid].value;
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
    if (CALCULATE_DUAL=="tree") {

        // Initialize empty u and v equations using the Variable Data Type >>
        Variable default_variable;
        Variable * U_vars, * V_vars;

        // This can be done slightly better given a superMemset function that can memset any struct
        thrust::device_vector<Variable> U_vars_vector(data->numSupplies);
        thrust::fill(U_vars_vector.begin(), U_vars_vector.end(), default_variable);
        U_vars = thrust::raw_pointer_cast(U_vars_vector.data());

        thrust::device_vector<Variable> V_vars_vector(data->numDemands);
        thrust::fill(V_vars_vector.begin(), V_vars_vector.end(), default_variable);
        V_vars = thrust::raw_pointer_cast(V_vars_vector.data());
        
        // Set u[0] = 0 on device >> // This can be done more smartly - low prioirity
        default_variable.value = 0;
        default_variable.assigned = true;
        U_vars_vector[0] = default_variable;
        
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
    else {
        std::cout<<"Invalid method of dual calculation!"<<std::endl;
        std::exit(-1); 
    }

}

/*
Kernel to compute Reduced Costs in the transportation table
*/
__global__ void computeReducedCosts(float * u_vars_ptr, float * v_vars_ptr, float * d_costs_ptr, float * d_reducedCosts_ptr, 
    int numSupplies, int numDemands) {

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
void uvModel_parallel::get_reduced_costs() {

    std::cout<<"\t\tComputing Reduced Costs ..."<<std::endl;
    dim3 __dimBlock(blockSize, blockSize, 1); // Refine this based on device query
    dim3 __dimGrid(ceil(1.0*data->numDemands/blockSize), ceil(1.0*data->numSupplies/blockSize), 1);
    computeReducedCosts<<< __dimGrid, __dimBlock >>>(u_vars_ptr, v_vars_ptr, d_costs_ptr, d_reducedCosts_ptr, 
        data->numSupplies, data->numDemands);
    cudaDeviceSynchronize();
    std::cout<<"\t\tComputing Reduced Costs - complete!"<<std::endl;

}

void uvModel_parallel::perform_pivot(bool &result) {
    // Void Functionality 
    result = true;
}

/*
Generate a tree on the global memory using the initial set of feasible flows
*/
__global__ void create_initial_tree(flowInformation * d_flows_ptr, int * d_adjMtx_ptr, 
    int numSupplies, int numDemands) {
    
    int V = numSupplies+numDemands;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    if (gid < V - 1) {
    
        flowInformation _this_flow = d_flows_ptr[gid];
        int row = _this_flow.source;
        int column =  _this_flow.destination;
        d_adjMtx_ptr[row*V + (numSupplies + column)] = _this_flow.qty;
        d_adjMtx_ptr[(column + numSupplies) * V + row] = _this_flow.qty;

    }
}

/*
Reverse operation of generating a tree from the feasible flows - unordered allocation
*/
__global__ void retrieve_final_tree(flowInformation * d_flows_ptr, int * d_adjMtx_ptr, int numSupplies, int numDemands) {

    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies+numDemands;
    int gid = row_indx*V + col_indx;

    // Upper triangle scope of adj matrix
    if (col_indx < V && row_indx < V && row_indx <= col_indx) {
        // Check if this is a flow edge - 
        if (d_adjMtx_ptr[gid] > 0) {

            int flow_indx = row_indx*numDemands + (col_indx - numSupplies);
            flowInformation _this_flow;
            _this_flow.qty = d_adjMtx_ptr[gid];
            _this_flow.source = row_indx;
            _this_flow.destination = col_indx - numSupplies;
            d_flows_ptr[flow_indx] = _this_flow; 
        
        }
    }
}

void uvModel_parallel::execute() {
    
    // **************************************
    // Finding BFS >>
    // **************************************
    generate_initial_BFS();

    // **************************************
    // Modified Distribution Method (u-v method) - parallel (improve the BFS solution)
    // **************************************
    
    // STEP 1 : Allocate relevant memory and transfer the necessary to GPU Memory >>
    // **************************************
    std::cout<<"PASS 1 :: creating the necessary data structures on global memory"<<std::endl;

        // 1.1 Create and Initialize u and v varaibles 
    cudaMalloc((void **) &u_vars_ptr, sizeof(float)*data->numSupplies);
    cudaMalloc((void **) &v_vars_ptr, sizeof(float)*data->numDemands);
    cudaMalloc((void **) &d_reducedCosts_ptr, sizeof(float)*data->numSupplies*data->numDemands);
    std::cout<<"\tGenerated shadow price vectors ..."<<std::endl;

        // 1.2 Transfer flows on device and prepare an adjacency matrix >>
    int V = data->numSupplies+data->numDemands;  // Number of vertices in the tree 
    cudaMalloc((void **) &d_adjMtx_ptr, sizeof(int)*(V*V)); 
    cudaMemset(d_adjMtx_ptr, 0, sizeof(int)*(V*V));

    cudaMalloc((void **) &d_flows_ptr, sizeof(flowInformation)*(V-1));
    cudaMemcpy(d_flows_ptr, feasible_flows, sizeof(flowInformation)*(V-1), cudaMemcpyHostToDevice);

        // 1.3 Small kernel to parallely create a tree using the flows
    create_initial_tree <<< ceil(1.0*(V-1)/blockSize), blockSize >>> (d_flows_ptr, d_adjMtx_ptr, data->numSupplies, data->numDemands);
    std::cout<<"\tGenerated initial tree ..."<<std::endl;
    // Now device_flows are useless; All information about graph is now contained within d_adjMatrix on device =>
    cudaFree(d_flows_ptr);

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
    std::cout<<"PASS 2 : Simplex => find the dual -> reduced -> pivots -> repeat!"<<std::endl;
    while (!result) {

        // 2.1 
        solve_uv();  
        // NOTE:: This method cannot be called before this step because of memory allocations above
        // u_vars_ptr and v_vars ptr were populated on device

        // 2.2 
        get_reduced_costs();
        // d_reducedCosts_ptr was populated on device
        
        // DEBUG :: 
        view_uvra();

        // 2.3
        perform_pivot(result);
        iteration_counter++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	double solution_time = duration.count();
    std::cout<<"Simplex completed in : "<<solution_time<<" secs. and "<<iteration_counter<<" iterations."<<std::endl;

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
    thrust::remove_if(device_flows.begin(), device_flows.end(), is_zero());
    // Assuming M+N-1 edges still exist 
    cudaMemcpy(feasible_flows, d_flows_ptr, (data->numSupplies+data->numDemands-1)*sizeof(flowInformation), cudaMemcpyDeviceToHost);
}

void uvModel_parallel::create_flows()
{
    memcpy(optimal_flows, feasible_flows, (data->numSupplies+data->numDemands-1)*sizeof(flowInformation));
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

    // Print adjMatrix >>
    int * h_adjMtx;
    int _V = data->numSupplies+data->numDemands;
    h_adjMtx = (int *) malloc((_V*_V)*sizeof(int));
    cudaMemcpy(h_adjMtx, d_adjMtx_ptr, (_V*_V)*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < _V; i++) {
        for (int j = 0; j < _V; j++) {
            std::cout << "adjMatrix[" << i << "]["<< j << "] = " << h_adjMtx[i*_V +j] << std::endl;
        }
    }
    
    std::cout<<"*****************************"<<std::endl;
}

