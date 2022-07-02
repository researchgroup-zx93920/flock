/*
DISCLAIMER : 
1. INTENTIONALLY AVOIDING THE USE OF BOOST LOGGER UNLESS THERE'S A BETTER ALTERNATIVE
2. Arrangement of Functions : Find the kernels used just above the corresponding method

TODO:
1. Testing methods for both kinds of DFS (CPU and GPU)
2. Can avoid memcpy opertion of device_CostMatrix_ptr between constructor and vogel call
*/

#include "ss_model_parallel.h"

/*
Constructor - Maintaining consistancy across algorithm modes
*/
ssModel_parallel::ssModel_parallel(ProblemInstance *problem, flowInformation *flows)
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
    std::cout << "A ss_model_parallel object was successfully created" << std::endl; 
}

/*
Destructor - Low Prioirity Issue (Handle Later)
*/
ssModel_parallel::~ssModel_parallel()
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
void ssModel_parallel::generate_initial_BFS()
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


/* Perform pivot using a appropriate opterations as selected */
void ssModel_parallel::perform_pivot(bool &result, int iteration) 
{
    // Find a negative reduced cost and pivot along >>
    if (PIVOTING_STRATEGY == "parallel_bfs")
    { 
        int num_pivots = 0;
        SS_METHOD::perform_a_parallel_pivot(pivot, timer, graph, d_costs_ptr, result,
            data->numSupplies, data->numDemands, iteration, num_pivots);
    }
    else
    {
        std::cout<<"Invalid selection of pivoting strategy for ss Model"<<std::endl;
        exit(-1);
    }    
}


void ssModel_parallel::execute() 
{
    // SIMPLEX ALGORITHM >>
    std::cout<<"------------- PARAMS L1 -------------\nBFS: "<<BFS_METHOD<<"\nPIVOTING STRATEGY: "<<PIVOTING_STRATEGY;
    std::cout<<"\n-------------------------------------"<<std::endl;

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
    SS_METHOD::pivotMalloc(pivot, data->numSupplies, data->numDemands, PIVOTING_STRATEGY);
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

    while ((!result) && iteration_counter < MAX_ITERATIONS) {

        // std::cout<<"Iteration :"<<iteration_counter<<std::endl;
        make_adjacency_list(graph, data->numSupplies, data->numDemands);
        // view_tree();
        
        // 2.1 
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

    SS_METHOD::pivotFree(pivot, PIVOTING_STRATEGY);
    std::cout<<"\tSuccessfully de-allocated Resources for PIVOT ..."<<std::endl;

    std::cout<<"\tProcessing Solution ..."<<std::endl;
    cudaMemcpy(graph.d_flowMtx_ptr, graph.h_flowMtx_ptr, (graph.V-1)*sizeof(float), cudaMemcpyHostToDevice);
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

void ssModel_parallel::create_flows()
{
    memcpy(optimal_flows, feasible_flows, (data->active_flows)*sizeof(flowInformation));
}


/*
For Debugging purposes view - adjMatrix >> 
*/
void ssModel_parallel::view_tree() 
{

    std::cout<<"Viewing tree - adjMatrix & flowMatrix \n *******************************"<<std::endl;

    // Print adjMatrix & flowMatrix >>
    int * h_adjMtx;
    float * h_flowMtx;
    int _V = data->numSupplies+data->numDemands;
    
    h_adjMtx = (int *) malloc(((_V*(_V+1))/2)*sizeof(int));
    gpuErrchk(cudaMemcpy(h_adjMtx, graph.d_adjMtx_ptr, ((_V*(_V+1))/2)*sizeof(int), cudaMemcpyDeviceToHost));

    h_flowMtx = (float *) malloc((_V-1)*sizeof(float));
    gpuErrchk(cudaMemcpy(h_flowMtx, graph.d_flowMtx_ptr, (_V-1)*sizeof(float), cudaMemcpyDeviceToHost));

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