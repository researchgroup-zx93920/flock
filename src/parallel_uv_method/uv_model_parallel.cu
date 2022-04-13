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
    These are special kernels - all are classified in DUAL_solver.h module
    */

    if (CALCULATE_DUAL=="tree") 
    {
        /* 
        Initialize u and v and then solve them through 
        Breadth first search using the adj matrix provided
        */
        find_dual_using_tree(u_vars_ptr, v_vars_ptr, d_adjMtx_ptr, d_costs_ptr, U_vars, V_vars, data->numSupplies, data->numDemands);
    }
    else if (CALCULATE_DUAL=="sparse_linear_solver") {
        /* 
        Solve a system of linear equations
        1. Create a sparse matrix A and Vector B
            - Invoke a kernel to fill A_csr, B on device  
            - Set the default allocation
        2. Solve the sparse system A_csr * x = b // cuSparse solver
        */
        find_dual_using_sparse_solver(u_vars_ptr, v_vars_ptr, d_costs_ptr, d_adjMtx_ptr,
            d_csr_values, d_csr_columns, d_csr_offsets, d_x, d_b, nnz,
            data->numSupplies, data->numDemands);
    }

    else if (CALCULATE_DUAL=="dense_linear_solver") {
        /* 
        Solve a system of linear equations
        1. Create a dense matrix A and Vector B
            - Initialize a zero dense matrix A
            - Invoke a kernel to fill A, B on device  
            - Set the default allocation
        2. Solve the dense system Ax = b  // cuBlas solver
        */
        // YET TO BE IMPLEMENTED >> 
        find_dual_using_dense_solver(u_vars_ptr, v_vars_ptr, d_costs_ptr, d_adjMtx_ptr,
            d_A, d_x, d_b, data->numSupplies, data->numDemands);
    }

    else {
    
        std::cout<<"Invalid method of dual calculation!"<<std::endl;
        std::exit(-1); 

    }
}


/*
Pretty generic method to compute reduced 
costs provided a cost-matrix and u_vars, v_vars and cost Matrix on device
*/
void uvModel_parallel::get_reduced_costs() 
{
    dim3 __dimBlock(blockSize, blockSize, 1); // Refine this based on device query
    dim3 __dimGrid(ceil(1.0*data->numDemands/blockSize), ceil(1.0*data->numSupplies/blockSize), 1);
    computeReducedCosts<<< __dimGrid, __dimBlock >>>(u_vars_ptr, v_vars_ptr, d_costs_ptr, d_reducedCosts_ptr, 
        data->numSupplies, data->numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
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

    // Find the position of the most negative reduced cost >>
    int min_index = thrust::min_element(thrust::device, d_reducedCosts_ptr, 
            d_reducedCosts_ptr + (data->numSupplies*data->numDemands)) - d_reducedCosts_ptr;
    // have all the reduced costs in the d_reducedCosts_ptr on device
    float min_reduced_cost = 0;
    gpuErrchk(cudaMemcpy(&min_reduced_cost, d_reducedCosts_ptr + min_index, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout<<"Min Reduced Cost  = "<<min_reduced_cost<<std::endl;
    if (min_reduced_cost < 0 && std::abs(min_reduced_cost) > 10e-5)
    {
        // Found a negative reduced cost >>
        if (PIVOTING_STRATEGY == "sequencial") 
        {
            // pivot row and pivot col are declared private attributes
            pivot_row =  min_index/data->numDemands;
            pivot_col = min_index - (pivot_row*data->numDemands);
            std::cout<<"Sequencial Pivoting"<<std::endl;
        }

        else if (PIVOTING_STRATEGY == "parallel") 
        {
            std::cout<<" Parallel Pivoting!" <<std::endl;
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




void uvModel_parallel::execute() 
{
    // SIMPLEX ALGORITHM >>

    // **************************************
    // STEP 1: Finding BFS
    // **************************************
    generate_initial_BFS();

    /* **************************************
    STEP 2: Modified Distribution Method (u-v method) - parallel/hybrid (improve the BFS solution)
    **************************************
    LOOP THORUGH - 
        2.1 Use the current tree on device and solve u's and v's
        2.2 Compute Reduced costs
            2.3.1 If no - negative reduced costs - break the loop
            2.3.2 If there exist negative reduced costs -
                Perform a pivot operation (more details on the corresponding function) */
    
        
        // **************************************
        // INITIALIZE STEP 2 : Allocate relevant memory and transfer the necessary data to GPU Memory >>
        //            Perform Pre-Process Operation before pivoting
        // **************************************
        std::cout<<"SIMPLEX PASS 1 :: creating the necessary data structures on global memory"<<std::endl;

        // Follow DUAL_solver for the following
        initialize_device_DUAL(&u_vars_ptr, &v_vars_ptr, &U_vars, &V_vars, &d_csr_values, &d_csr_columns, &d_csr_offsets,
            &d_A, &d_b, &d_x, nnz, data->numSupplies, data->numDemands);
        std::cout<<"\tSuccessfully allocated Resources for DUAL ..."<<std::endl;

        // Container for reduced costs
        gpuErrchk(cudaMalloc((void **) &d_reducedCosts_ptr, sizeof(float)*data->numSupplies*data->numDemands));
        std::cout<<"\tSuccessfully allocated Resources for Reduced costs ..."<<std::endl;

        // Create tree structure on host device (for pivoting)
        create_IBF_tree_on_host_device(feasible_flows, 
            &d_adjMtx_ptr, &h_adjMtx_ptr, &d_flowMtx_ptr, &h_flowMtx_ptr, 
            data->numSupplies, data->numDemands);
        std::cout<<"\tGenerated initial tree (on host & device) ..."<<std::endl;
        // Follow PIVOTING_dfs for the following
        // initialize_pivoting();

        bool result = false;
        int iteration_counter = 0;
        
        // **************************************
        // LOOP STEP 2 : SIMPLEX PROCEDURE
        // **************************************
        std::cout<<"SIMPLEX PASS 2 :: find the dual -> reduced -> pivots -> repeat!"<<std::endl;
        result = false;
        
        auto start = std::chrono::high_resolution_clock::now();
        while ((!result) && iteration_counter < MAX_ITERATIONS) {

            std::cout<<"Iteration :"<<iteration_counter<<std::endl;

            // 2.1 
            solve_uv();

            // NOTE:: This method cannot be called before this step because of memory allocations above
            // u_vars_ptr and v_vars ptr were populated on device

            // 2.2 
            get_reduced_costs();
            // d_reducedCosts_ptr was populated on device
            
            // view_uvra();

            // 2.3
            perform_pivot(result);
            iteration_counter++;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double solution_time = duration.count();
        std::cout<<"\tSimplex completed in : "<<solution_time<<" millisecs. and "<<iteration_counter<<" iterations."<<std::endl;

        std::cout<<"SIMPLEX PASS 3 :: Clearing the device memory and transfering the necessary data on CPU"<<std::endl;
        
        // ****************************
        // TERMINATION
        // **************************** 
        terminate_DUAL(u_vars_ptr, v_vars_ptr, U_vars, V_vars, 
            d_csr_values, d_csr_columns, d_csr_offsets, d_A, d_b, d_x);
        // terminate_PIVOT();
        cudaFree(d_reducedCosts_ptr);


    // Recreate device flows using the current adjMatrix
    flowInformation default_flow;
    default_flow.qty = 0;

    flowInformation * d_flows_ptr;
    gpuErrchk(cudaMalloc((void **) &d_flows_ptr, sizeof(flowInformation)*(data->numSupplies*data->numDemands)));
    thrust::fill(thrust::device, d_flows_ptr, d_flows_ptr + (data->numSupplies*data->numDemands), default_flow);

    dim3 __blockDim(blockSize, blockSize, 1);
    int grid_size = ceil(1.0*(V)/blockSize); // VxV grid
    dim3 __gridDim(grid_size, grid_size, 1);
    retrieve_final_tree <<< __gridDim, __blockDim >>> (d_flows_ptr, d_adjMtx_ptr, d_flowMtx_ptr, 
        data->numSupplies, data->numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy the (flows > 0) back on the host >>
    auto flow_end = thrust::remove_if(thrust::device,
        d_flows_ptr, d_flows_ptr + (data->numSupplies*data->numDemands), is_zero());
    int flow_count = flow_end - d_flows_ptr;
    std::cout<<"\tFound "<<flow_count<<" active flows in the final result"<<std::endl;
    data->active_flows = flow_count;
    gpuErrchk(cudaMemcpy(feasible_flows, d_flows_ptr, (data->active_flows)*sizeof(flowInformation), cudaMemcpyDeviceToHost));

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

    // Print U >>
    float * u_vector;
    u_vector = (float *) malloc(data->numSupplies*sizeof(float));
    gpuErrchk(cudaMemcpy(u_vector, u_vars_ptr, data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < data->numSupplies; i++) {
        std::cout << "U[" << i << "] = " << u_vector[i] << std::endl;
    }

    std::cout<<"*****************************"<<std::endl;

    // Print V >>
    float * v_vector;
    v_vector = (float *) malloc(data->numDemands*sizeof(float));
    gpuErrchk(cudaMemcpy(v_vector, v_vars_ptr, data->numDemands*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < data->numDemands; i++) {
        std::cout << "V[" << i << "] = " << v_vector[i] << std::endl;
    }

    std::cout<<"*****************************"<<std::endl;

    // Print reduced costs
    float * h_reduced_costs;
    h_reduced_costs = (float *) malloc(data->numDemands*data->numSupplies*sizeof(float));
    gpuErrchk(cudaMemcpy(h_reduced_costs, d_reducedCosts_ptr, data->numDemands*data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < data->numDemands*data->numSupplies; i++) {
        std::cout << "ReducedCosts[" << i << "] = " << h_reduced_costs[i] << std::endl;
    }

    std::cout<<"*****************************"<<std::endl;

    // Print adjMatrix & flowMatrix >>
    int * h_adjMtx;
    float * h_flowMtx;
    int _V = data->numSupplies+data->numDemands;
    
    h_adjMtx = (int *) malloc(((_V*(_V+1))/2)*sizeof(int));
    gpuErrchk(cudaMemcpy(h_adjMtx, d_adjMtx_ptr, ((_V*(_V+1))/2)*sizeof(int), cudaMemcpyDeviceToHost));

    h_flowMtx = (float *) malloc((_V-1)*sizeof(float));
    gpuErrchk(cudaMemcpy(h_flowMtx, d_flowMtx_ptr, (_V-1)*sizeof(float), cudaMemcpyDeviceToHost));

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