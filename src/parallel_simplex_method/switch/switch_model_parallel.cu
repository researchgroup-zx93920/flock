/*
DISCLAIMER : 
1. INTENTIONALLY AVOIDING THE USE OF BOOST LOGGER UNLESS THERE'S A BETTER ALTERNATIVE
2. Arrangement of Functions : Find the kernels used just above the corresponding method

TODO:
1. Testing methods for both kinds of DFS (CPU and GPU)
2. Can avoid memcpy opertion of device_CostMatrix_ptr between constructor and vogel call
*/

#include "switch_model_parallel.h"

/*
Constructor - Maintaining consistancy across algorithm modes
*/
switchModel_parallel::switchModel_parallel(ProblemInstance *problem, flowInformation *flows)
{

    std::cout << std::endl <<"Initializing Stepping Stone Model (parallel)" << std::endl;
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
switchModel_parallel::~switchModel_parallel()
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
void switchModel_parallel::generate_initial_BFS()
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

void switchModel_parallel::solve_uv()
{
    // Use the facility method avilable form UV
    UV_METHOD::find_dual_using_host_bfs(dual, graph, data->costs, data->numSupplies, data->numDemands);
}


/*
Pretty generic method to compute reduced 
costs provided a cost-matrix and u_vars, v_vars and cost Matrix on device
*/
void switchModel_parallel::get_reduced_costs() 
{
    if (REDUCED_COST_MODE == "parallel") {

        // Transfer dual costs to GPU 
        gpuErrchk(cudaMemcpy(dual.u_vars_ptr, &dual.h_variables[0], sizeof(int)*data->numSupplies, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dual.v_vars_ptr, &dual.h_variables[data->numSupplies], sizeof(int)*data->numDemands, cudaMemcpyHostToDevice));

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

    else { 

        std::cout<<"ERROR : for switching model - enfored reduced cost mode to parallel!"<<std::endl;
    
    }

        
}



/* Perform pivot using a appropriate opterations as selected */
void switchModel_parallel::perform_pivot(bool &result, int iteration, int &mode) 
{
    // Find a negative reduced cost and pivot along >>
    // Fixed Pivoting Strategy 
    if (mode == 1) 
    {
        UV_METHOD::perform_a_sequencial_pivot(pivot, timer, graph, result,
            data->numSupplies, data->numDemands);
    }
    else if (mode == 0) 
    {
        int num_pivots = 0;
        SS_METHOD::perform_a_parallel_pivot(pivot, timer, graph, d_costs_ptr, result,
            data->numSupplies, data->numDemands, iteration, num_pivots);

        // Switching >>
        if (num_pivots < 10) {
            mode = 1;
        }
    }
    else
    {
        std::cout<<"Invalid selection of pivoting strategy"<<std::endl;
        exit(-1);
    }    
}

void switchModel_parallel::setup_host_graph() {

    gpuErrchk(cudaMemcpy(graph.h_vertex_degree, &graph.d_vertex_degree[1], sizeof(int)*graph.V, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(graph.h_vertex_start, graph.d_vertex_start, sizeof(int)*graph.V, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(graph.h_adjVertices, graph.d_adjVertices, sizeof(int)*2*(graph.V-1), cudaMemcpyDeviceToHost));
    
    for (int i=0; i < graph.V; i++) {

        for (int j=0; j < graph.h_vertex_degree[i]; j++) {
            graph.h_Graph[i].push_back(graph.h_adjVertices[graph.h_vertex_start[i]+j]);
        }
    
    }

}



void switchModel_parallel::execute() 
{
    // SIMPLEX ALGORITHM >>
    std::cout<<"------------- PARAMS L1 -------------\nBFS: "<<BFS_METHOD<<"\nCALCULATE_DUAL: ";
    std::cout<<CALCULATE_DUAL<<"\nPIVOTING STRATEGY: switch-ParallelToSeq\n-------------------------------------"<<std::endl;

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
    STEP 2: Stepping Stone Method (ss-method) - parallel
    **************************************
    LOOP THORUGH - 
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
    SS_METHOD::pivotMalloc(pivot, data->numSupplies, data->numDemands, "parallel_bfs");
    std::cout<<"\tSuccessfully allocated Resources for PIVOTING ..."<<std::endl;
    
    // Create tree structure on host and device (for pivoting)
    create_IBF_tree_on_host_device(graph, feasible_flows, data->numSupplies, data->numDemands);
    std::cout<<"\tGenerated initial tree (on host & device) ..."<<std::endl;
    
    // **************************************
    // LOOP STEP 2 : SIMPLEX PROCEDURE
    // **************************************
    std::cout<<"SIMPLEX PASS 2 :: compute loops with cost improvement -> perform pivots -> repeat!"<<std::endl;
    auto iter_start = std::chrono::high_resolution_clock::now();
    auto iter_end = std::chrono::high_resolution_clock::now();
    auto iter_duration = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start);
    int mode = 0;
    int swiched_iteration = false;

    while ((!result) && iteration_counter < MAX_ITERATIONS) {

        // std::cout<<"Iteration :"<<iteration_counter<<std::endl;
        // view_tree();
        
        // 2.1 

        if (mode == 0) {

            make_adjacency_list(graph, data->numSupplies, data->numDemands);

            iter_start = std::chrono::high_resolution_clock::now();
            
            perform_pivot(result, iteration_counter, mode);
            
            iter_end = std::chrono::high_resolution_clock::now();
            iter_duration = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start);
            pivot_time += iter_duration.count();     
            swiched_iteration = (mode == 1);

        }

        // switch is guaranteed - the following is a one time operation
        if (swiched_iteration) {

            // Follow PIVOTING_dfs for the following
            // Make some interchange
            SS_METHOD::pivotFree(pivot, "parallel_bfs");
            UV_METHOD::pivotMalloc(pivot, data->numSupplies, data->numDemands, "sequencial_dfs");
            std::cout<<"\tSuccessfully re-allocated Resources for PIVOTING ..."<<std::endl;

            // Follow DUAL_solver for the following
            UV_METHOD::dualMalloc(dual, data->numSupplies, data->numDemands);
            std::cout<<"\tSuccessfully allocated Resources for DUAL ..."<<std::endl;
            make_adjacency_list(graph, data->numSupplies, data->numDemands);
            setup_host_graph();
            swiched_iteration = false;
        }

        if (mode == 1) {

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
            
            // d_reducedCosts_ptr was populated on device
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
            
            perform_pivot(result, iteration_counter, mode);

            iter_end = std::chrono::high_resolution_clock::now();
            iter_duration = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start);
            pivot_time += iter_duration.count();

        }

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
    UV_METHOD::pivotFree(pivot, "sequencial_dfs");
    std::cout<<"\tSuccessfully de-allocated Resources for PIVOT ..."<<std::endl;

    std::cout<<"\tProcessing Solution ..."<<std::endl;

    cudaMemcpy(graph.d_flowMtx_ptr, graph.h_flowMtx_ptr, (graph.V-1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(graph.d_adjMtx_ptr, graph.h_adjMtx_ptr, ((graph.V*(graph.V+1))/2)*sizeof(int), cudaMemcpyHostToDevice);
    
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
    for (int i=0; i< (data->active_flows); i++){
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

void switchModel_parallel::create_flows()
{
    memcpy(optimal_flows, feasible_flows, (data->active_flows)*sizeof(flowInformation));
}