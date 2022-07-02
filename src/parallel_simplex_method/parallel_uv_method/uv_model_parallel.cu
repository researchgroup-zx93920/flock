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
    gpuErrchk(cudaMalloc((void **) &d_costs_ptr, sizeof(float) * (data->numSupplies) * (data->numDemands)));
    gpuErrchk(cudaMalloc((void **) &device_costMatrix_ptr, sizeof(MatrixCell) * (data->numSupplies) * (data->numDemands)));
    gpuErrchk(cudaMemcpy(d_costs_ptr, data->costs, sizeof(float) * (data->numSupplies) * (data->numDemands), cudaMemcpyHostToDevice));

    dim3 _dimBlock(blockSize,blockSize,1);
    dim3 _dimGrid(ceil(1.0*data->numDemands/blockSize), ceil(1.0*data->numSupplies/blockSize), 1);
    createCostMatrix<<<_dimGrid, _dimBlock>>>(device_costMatrix_ptr, d_costs_ptr, data->numSupplies, data->numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(costMatrix, device_costMatrix_ptr, sizeof(MatrixCell) * (data->numSupplies) * (data->numDemands), cudaMemcpyDeviceToHost));
    // Why we copy it back to host? because NWC and Vogel can interchangably use this struct
    
    // !! Setup Constants !!
    graph.V = data->numSupplies+data->numDemands;

    // Initialize model statistics >>
    uv_time = 0.0; 
    reduced_cost_time = 0.0; 
    pivot_time = 0.0;
    cycle_discovery_time = 0.0;
    resolve_time = 0.0;
    adjustment_time = 0.0;
    totalIterations = 0;
    objVal = 0.0;
    totalIterations = 0;
    totalSolveTime = 0.0;
    std::cout << "An uv_model_parallel object was successfully created" << std::endl; 
}

/*
Destructor - Low Prioirity Issue (Handle Later)
*/
uvModel_parallel::~uvModel_parallel()
{
    // On thrust layer - replace by thrust eqv. of free
    free(costMatrix);
    free(feasible_flows);

    // FREE GPU MEMORY CREATED IN INITIALIZATION
    gpuErrchk(cudaFree(d_costs_ptr));
    gpuErrchk(cudaFree(device_costMatrix_ptr)); 
}

/*
Generate initial basic feasible solution using the selected method
This function populates the feasible_flows attribute of the parent class, that is subject
to updates by the subsequent improvement techniques
*/
void uvModel_parallel::generate_initial_BFS()
{
    // Data is available on the class objects - Call one of the IBFS methods on these
    
    if (BFS_METHOD == "nwc_host")
    {
        // Approach 1: Northwest Corner (Naive BFS - sequential)
        // --------------------------------------------------------
        // Utilize NW Corner method to determine basic feasible solution, (uncomment below)
        find_nw_corner_bfs_seq(data->supplies, data->demands, costMatrix, feasible_flows,
                               data->numSupplies, data->numDemands);
    }
    else if (BFS_METHOD == "vam_host") 
    {
        // Approach 2: Vogel's Approximation - sequencial
        // --------------------------------------------------------
        // Utilitze vogel's approximation to determine basic fesible solution using CUDA kernels
        find_vogel_bfs_sequencial(data->supplies, data->demands, costMatrix, feasible_flows,
                                data->numSupplies, data->numDemands);
    }
    else if (BFS_METHOD == "vam_device")
    {
        // Approach 3: Vogel's Approximation - parallel
        // --------------------------------------------------------
        // Utilitze vogel's approximation to determine basic fesible solution using CUDA kernels
        find_vogel_bfs_parallel(data->supplies, data->demands, costMatrix, feasible_flows,
                                data->numSupplies, data->numDemands);
    }
    // The array feasible flows at this stage holds the initial basic feasible solution
    else 
    {
        std::cout<<"Invalid BFS_METHOD"<<std::endl;
        exit(-1);
    }
}

/*
Given a u and v vector on device - computes the dual costs for all the constraints. There could be multiple ways 
to solve the dual costs for a given problem. Packaged method derive u's and v's and load them onto 
the U_vars and V_vars attributes
    1. Use a bfs method for dual (trickle down approach)
    2. Use a off-the-shelf solver for linear equation 
*/
void uvModel_parallel::solve_uv()
{
    /* 
    solve_uv method - Need to populate u_vars_ptr and v_vars_ptr attributes of this class 
    This is an internal function can't be API'ed - Executes successfully in a specific situation
    This is special kernel - classified in DUAL_solver.h module
    */

    /* 
    Initialize u and v and then solve them through 
    Breadth first search using the adj matrix provided
    BFS is performed on the host 
    */
    UV_METHOD::find_dual_using_host_bfs(dual, graph, data->costs, data->numSupplies, data->numDemands);
    
}


/*
Pretty generic method to compute reduced 
costs provided a cost-matrix and u_vars, v_vars and cost Matrix on device
*/
void uvModel_parallel::get_reduced_costs() 
{
    if (REDUCED_COST_MODE == "parallel") {

        dim3 __dimBlock(reducedcostBlock, reducedcostBlock, 1); // Refine this based on device query
        dim3 __dimGrid(ceil(1.0*data->numDemands/reducedcostBlock), ceil(1.0*data->numSupplies/reducedcostBlock), 1);
        
        // Compute >> 
        computeReducedCosts<<< __dimGrid, __dimBlock >>>(dual.u_vars_ptr, dual.v_vars_ptr, d_costs_ptr, pivot.opportunity_cost, 
            data->numSupplies, data->numDemands);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Reduce >> 
        int min_indx = thrust::min_element(thrust::device,
                pivot.opportunity_cost, pivot.opportunity_cost + data->numSupplies*data->numDemands) - pivot.opportunity_cost;
        gpuErrchk(cudaMemcpy(&pivot.reduced_cost, &pivot.opportunity_cost[min_indx], sizeof(float), cudaMemcpyDeviceToHost));

        pivot.pivot_row = min_indx/data->numDemands;
        pivot.pivot_col = min_indx - (pivot.pivot_row*data->numDemands);
    
    }

    else if (REDUCED_COST_MODE == "sequencial") {

        float * h_u_vars = &dual.h_variables[0];
        float * h_v_vars = &dual.h_variables[data->numSupplies];
        float min_r = INT_MAX*1.0f;
        int selected_pivot_row = -1;
        int selected_pivot_column = -1;

        for (int i = 0; i < data->numSupplies; i ++) {
            for (int j = 0; j < data-> numDemands; j++) {
                int indx = i*data->numDemands + j;
                float r = costMatrix[indx].cost - h_u_vars[i] - h_v_vars[j];
                if (r < min_r) {
                    min_r = r;
                    selected_pivot_row = i;
                    selected_pivot_column = j;
                }
            }
        }
        
        pivot.reduced_cost = min_r;
        pivot.pivot_row = selected_pivot_row;
        pivot.pivot_col = selected_pivot_column;

    }
}


/* Perform pivot using a appropriate opterations as selected */
void uvModel_parallel::perform_pivot(bool &result, int iteration) 
{
    // Find a negative reduced cost and pivot along >>
    UV_METHOD::perform_a_sequencial_pivot(pivot, timer, graph, result,
            data->numSupplies, data->numDemands);

}

void uvModel_parallel::setup_host_graph() {

    gpuErrchk(cudaMemcpy(graph.h_vertex_degree, &graph.d_vertex_degree[1], sizeof(int)*graph.V, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(graph.h_vertex_start, graph.d_vertex_start, sizeof(int)*graph.V, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(graph.h_adjVertices, graph.d_adjVertices, sizeof(int)*2*(graph.V-1), cudaMemcpyDeviceToHost));
    
    for (int i=0; i < graph.V; i++) {

        for (int j=0; j < graph.h_vertex_degree[i]; j++) {
            graph.h_Graph[i].push_back(graph.h_adjVertices[graph.h_vertex_start[i]+j]);
        }
    
    }

}

void uvModel_parallel::execute() 
{
    // SIMPLEX ALGORITHM >>
    std::cout<<"------------- PARAMS L1 -------------\nBFS: "<<BFS_METHOD<<"\nCALCULATE_DUAL: ";
    std::cout<<CALCULATE_DUAL<<"\nPIVOTING STRATEGY: "<<PIVOTING_STRATEGY<<"\n-------------------------------------"<<std::endl;

    // **************************************
    // STEP 1: Finding BFS
    // **************************************
    auto start = std::chrono::high_resolution_clock::now();

    generate_initial_BFS();
    // The array feasible flows at this stage holds the initial basic feasible solution

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double solution_time = duration.count();
    std::cout << BFS_METHOD << " BFS Found in : " << solution_time << " millisecs." << std::endl;
    totalSolveTime += solution_time;
    
    /* **************************************
    STEP 2: Modified Distribution Method (u-v method) - parallel/hybrid (improve the BFS solution)
    **************************************
    LOOP THORUGH - 
        2.1 Use the current tree on device and solve u's and v's
        2.2 Compute Reduced costs
            2.3.1 If no - negative reduced costs - break the loop
            2.3.2 If there exist negative reduced costs -
                Perform a pivot operation (more details on the corresponding function) */
    start = std::chrono::high_resolution_clock::now();

    bool result = false;
    int iteration_counter = 0;
    result = false;

    // **************************************
    // INITIALIZATION AND PREPROCESS : 
    // Allocate relevant GPU memory and transfer the necessary data to DEVICE Memory >>
    // Perform Pre-Process Operation before pivoting
    // **************************************
    std::cout<<"SIMPLEX PASS 1 :: creating the necessary data structures on global memory"<<std::endl;
    
    // Follow DUAL_solver for the following
    UV_METHOD::dualMalloc(dual, data->numSupplies, data->numDemands);
    std::cout<<"\tSuccessfully allocated Resources for DUAL ..."<<std::endl;

    // Follow PIVOTING_dfs for the following
    UV_METHOD::pivotMalloc(pivot, data->numSupplies, data->numDemands, PIVOTING_STRATEGY);
    std::cout<<"\tSuccessfully allocated Resources for PIVOTING ..."<<std::endl;
    
    // Create tree structure on host and device (for pivoting)
    create_IBF_tree_on_host_device(graph, feasible_flows, data->numSupplies, data->numDemands);
    std::cout<<"\tGenerated initial tree (on host & device) ..."<<std::endl;
    
    // This might be just on time operation 
    /* This transformation step generate adj list for the current tree in each iteration
        the output is used by dual as well as pivot because DFS will improve by elimination of zeros in sparse matrix
        [ Note that this step is anyway performed in DUAL BFS but 
        this is expected to be helpful in parallel pivot to supercharge DFS !! ]
    */
    make_adjacency_list(graph, data->numSupplies, data->numDemands); 
    
    // Max level pivoting flexibility, STL saved somebody a lifetime
    setup_host_graph();

    // **************************************
    // LOOP STEP 2 : SIMPLEX PROCEDURE
    // **************************************
    std::cout<<"SIMPLEX PASS 2 :: find the dual -> reduced -> pivots -> repeat!"<<std::endl;
    auto iter_start = std::chrono::high_resolution_clock::now();
    auto iter_end = std::chrono::high_resolution_clock::now();
    auto iter_duration = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start);

    while ((!result) && iteration_counter < MAX_ITERATIONS) {

        // std::cout<<"Iteration :"<<iteration_counter<<std::endl;
        // view_tree();
        
        // 2.1 
        iter_start = std::chrono::high_resolution_clock::now();
        
        solve_uv();
        // view_uv();
        // u_vars_ptr and v_vars ptr were populated on device

        iter_end = std::chrono::high_resolution_clock::now();
        iter_duration = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start);
        uv_time += iter_duration.count();

        // 2.2 
        iter_start = std::chrono::high_resolution_clock::now();
        
        get_reduced_costs();
        // view_reduced_costs();
        // count_negative_reduced_costs();

        iter_end = std::chrono::high_resolution_clock::now();
        iter_duration = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start);
        reduced_cost_time += iter_duration.count();
        
        // DEBUG ::
        // view_uv();
        // view_tree();
        // view_reduced_costs();
        
        // 2.3
        iter_start = std::chrono::high_resolution_clock::now();
        
        perform_pivot(result, iteration_counter);

        iter_end = std::chrono::high_resolution_clock::now();
        iter_duration = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start);
        pivot_time += iter_duration.count();

        iteration_counter++;
    }

    std::cout<<"SIMPLEX PASS 3 :: Clearing the device memory and transfering the necessary data on CPU"<<std::endl;
    
    // **************************************
    // TERMINATION AND POST PROCESS :
    // Free up gpu memory and transfer the necessary back to HOST Memory >>
    // Post process operation after pivoting
    // **************************************

    UV_METHOD::dualFree(dual);
    std::cout<<"\tSuccessfully de-allocated resources for DUAL ..."<<std::endl;
    UV_METHOD::pivotFree(pivot, PIVOTING_STRATEGY);
    std::cout<<"\tSuccessfully de-allocated Resources for PIVOT ..."<<std::endl;

    std::cout<<"\tProcessing Solution ..."<<std::endl;
    cudaMemcpy(graph.d_flowMtx_ptr, graph.h_flowMtx_ptr, (graph.V-1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(graph.d_adjMtx_ptr, graph.h_adjMtx_ptr, ((graph.V*(graph.V+1))/2)*sizeof(float), cudaMemcpyHostToDevice);    
    retrieve_solution_on_current_tree(feasible_flows, graph, data->active_flows, data->numSupplies, data->numDemands);

    std::cout<<"Found "<<data->active_flows<<" active flows in the final result"<<std::endl;

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    solution_time = duration.count();
    std::cout<<" ============ Simplex completed in : "<<solution_time<<" millisecs. and "<<iteration_counter<<" iterations."<<std::endl;
    totalSolveTime += solution_time;
    totalIterations = iteration_counter;

    // Clean up!
    close_solver(graph);

    double objval = 0.0;
    for (int i=0; i< (data->active_flows); i++) {

        int _row = feasible_flows[i].source;
        int _col = feasible_flows[i].destination;
        int _key = _row*data->numDemands + _col;
        objval += feasible_flows[i].qty*data->costs[_key];
    
    }
    
    objVal = objval;
    data->totalFlowCost = objVal;
    data->solveTime = totalSolveTime;
    // Load back timer details
    cycle_discovery_time = timer.cycle_discovery;
    resolve_time = timer.resolve_time;
    adjustment_time = timer.adjustment_time;

    std::cout<<" ============ Current Objective Value = "<<objval<<std::endl<<std::endl;
    
}

void uvModel_parallel::create_flows()
{
    memcpy(optimal_flows, feasible_flows, (data->active_flows)*sizeof(flowInformation));
}


/*
For Debugging purposes view - u, v, r_costs and adjMatrix >> 
*/
void uvModel_parallel::view_uv() 
{

    std::cout<<"Viewing DUAL COSTS - U, V \n *******************************"<<std::endl;

    // Print U >>
    float * u_vector;
    
    if (REDUCED_COST_MODE == "parallel") {

        u_vector = (float *) malloc(data->numSupplies*sizeof(float));
        gpuErrchk(cudaMemcpy(u_vector, dual.u_vars_ptr, data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost));

    }
    else {
        u_vector = &dual.h_variables[0];
    }
    
    for (int i = 0; i < data->numSupplies; i++) {
        std::cout << "U[" << i << "] = " << u_vector[i] << std::endl;
    }

    std::cout<<" *****************************"<<std::endl;

    // Print V >>
    float * v_vector;

    if (REDUCED_COST_MODE == "parallel") {

        v_vector = (float *) malloc(data->numDemands*sizeof(float));
        gpuErrchk(cudaMemcpy(v_vector, dual.v_vars_ptr, data->numDemands*sizeof(float), cudaMemcpyDeviceToHost));

    }
    else {
        v_vector = &dual.h_variables[data->numSupplies];
    }

    for (int i = 0; i < data->numDemands; i++) {
        std::cout << "V[" << i << "] = " << v_vector[i] << std::endl;
    }
}

void uvModel_parallel::view_reduced_costs() 
{

    std::cout<<"Viewing Reduced Costs \n *******************************"<<std::endl;

    if (REDUCED_COST_MODE == "parallel") {

        // Print reduced costs
        float * h_reduced_costs;
        h_reduced_costs = (float *) malloc(data->numDemands*data->numSupplies*sizeof(float));
        gpuErrchk(cudaMemcpy(h_reduced_costs, pivot.opportunity_cost, data->numDemands*data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < data->numDemands*data->numSupplies; i++) {
    
            std::cout << "ReducedCosts[" << i << "] = " << h_reduced_costs[i] << std::endl;
    
        }
    
    }
    
    else {
    
        std::cout<<" !!!!!!!!! REDUCED COST COMPUTED ON HOST ARE NOT STORED !!!!!!!!!!!! "<<std::endl;
    
    }

}

void uvModel_parallel::count_negative_reduced_costs() 
{


    if (REDUCED_COST_MODE == "parallel") {

        // Print reduced costs
        float * h_reduced_costs;
        int _count = 0;
        h_reduced_costs = (float *) malloc(data->numDemands*data->numSupplies*sizeof(float));
        gpuErrchk(cudaMemcpy(h_reduced_costs, pivot.opportunity_cost, data->numDemands*data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < data->numDemands*data->numSupplies; i++) {
            if (h_reduced_costs[i] < 0 && abs(h_reduced_costs[i]) > 10e-3) {
                _count++;
            }
        }
        std::cout<<"Number of Negative Reduced Costs = "<<_count<<std::endl;
    }
    
    else {
    
        std::cout<<" !!!!!!!!! REDUCED COST COMPUTED ON HOST ARE NOT STORED !!!!!!!!!!!! "<<std::endl;
    
    }
}

void uvModel_parallel::view_tree() 
{

    std::cout<<"Viewing tree - adjMatrix & flowMatrix \n *******************************"<<std::endl;

    // Print adjMatrix & flowMatrix >>
    int * h_adjMtx;
    float * h_flowMtx;
    int _V = data->numSupplies+data->numDemands;
    
    h_adjMtx = graph.h_adjMtx_ptr; // (int *) malloc(((_V*(_V+1))/2)*sizeof(int));
    // gpuErrchk(cudaMemcpy(h_adjMtx, graph.d_adjMtx_ptr, ((_V*(_V+1))/2)*sizeof(int), cudaMemcpyDeviceToHost));

    h_flowMtx = graph.h_flowMtx_ptr; // (float *) malloc((_V-1)*sizeof(float));
    // gpuErrchk(cudaMemcpy(h_flowMtx, graph.d_flowMtx_ptr, (_V-1)*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < _V; i++) {
        for (int j = i; j < _V; j++) {
                int indx = TREE_LOOKUP(i, j, _V);
                if (h_adjMtx[indx] > 0) {
                    std::cout << "adjMatrix[" << i << "]["<< j << "] = "<<indx<<" = "<< h_adjMtx[indx]<< std::endl;
            }
        }
    }

    std::cout<<" *****************************"<<std::endl;
    
    for (int i = 0; i < _V-1; i++) {
        std::cout << "flowMatrix[" << i << "] = " << h_flowMtx[i] << std::endl;
    }

}