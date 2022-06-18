reduced_costs = (float *) malloc(sizeof(float)*data->numSupplies*data->numDemands);
    
    
    while (!result) {

        pivot_row = -1; // row with least reduced cost
        pivot_col = -1; // col with least reduced cost

        // solve_uv();
        

        // std::cout<<"Pivot Row = "<<pivot_row<<std::endl;
        // std::cout<<"Pivot col = "<<pivot_col<<std::endl;

        if (pivot_row >= 0 && pivot_col >= 0){
           perform_pivot(pivot_row, pivot_col); 
        }
        else
        {   
            std::cout<<"Reached Optimality!"<<std::endl;
            double objval = 0.0;
            for (int i=0; i< (data->numSupplies+data->numDemands-1); i++){
                int _row = feasible_flows[i].source;
                int _col = feasible_flows[i].destination;
                int _key = _row*data->numDemands + _col;
                objval += feasible_flows[i].qty*data->costs[_key];
            }
            data->totalFlowCost = objval;
            std::cout<<"Objective Value = "<<objval<<std::endl;
            result = true;
        }
        iteration_counter++;
        // std::cout<<"Iteration : "<<iteration_counter<<" completed!"<<std::endl;
    }

    __global__ void execute_parallel_pivot(Variable * u_vars, Variable * v_vars, int * h_adjMtx, 
        int * visited_ptr, MatrixCell * device_costMatrix, 
        int * supplies, int * demands,
        int matrixSupplies, int matrixDemands) {
    
            int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
            int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
            int V = matrixSupplies+matrixDemands;
    
            if (row_indx < matrixSupplies && col_indx < matrixDemands) {
                // r =  C_ij - (u_i + v_j);
                float r = device_costMatrix[row_indx*matrixDemands+col_indx].cost - u_vars[row_indx].value - v_vars[col_indx].value;
                if (r < 0) {
                    h_adjMtx[row_indx*V + (col_indx+matrixSupplies)] = 1;
                    h_adjMtx[(col_indx+matrixSupplies)*V + row_indx] = 1;
                }
                __syncthreads();
                if (r < 0) {
                    pathEdge * loop = (pathEdge *) malloc(sizeof(pathEdge));
                    loop->index = row_indx;
                    loop->next = NULL;
                    // where magic happens .. >>
                    bool discovered_cycle = micro_DFS(visited_ptr, h_adjMtx, V, row_indx, -1, loop);
                    
                    if (discovered_cycle) {
    
                        // loop - iterate and find the minimum value and supporting indexes >>
                        // Skip the last one
                        bool break_iterating_loop = true;
                        int min_flow = INT_MAX, min_index = -1, sym_min_index = -1, _iterN = 0;
                        pathEdge * _to = loop->next, * _from = _to->next;
                        while (break_iterating_loop) {
                            if (h_adjMtx[_from->index*V + _to->index] <= min_flow) {
                                min_flow = h_adjMtx[_from->index*V + _to->index]; // symmetrix
                                min_index = _from->index*V + _to->index;
                                sym_min_index = _to->index*V + _from->index;
                            }
                            _from = _from->next;
                            _to = _to->next;
                            break_iterating_loop = _to->next!=NULL;
                        }
    
                        // 
                        // break_iterating_loop = true;
                        // _iterN = 0;
                        // pathEdge _to = loop->next, _from = _to->next;
                        // while (break_iterating_loop) {
                        //     if (_iterN%2==0){
                        //         if (h_adjMtx[_from*V + _to] <= min_flow) {
                        //             min_flow = h_adjMtx[_this_loop->index];
                        //             min_index = _from*V + _to;
                        //         }
                        //     _from = _from->next;
                        //     _to = _to->next;
                        //     break_iterating_loop = to->next!=NULL;
                        // }
                        // this_loop is at the entering arc - this_loop index, loop index
    
                    }
                }
            }
        }
    
    
    void uvModel_parallel::perform_pivoting_parallel(int * visited_ptr, int * adjMtx_ptr) {
    
        if (CALCULATE_DUAL=="tree") {
    
            // Start U-V vectors
            // 2 ways for u-v vectors - either delete and recreate
            // or map the function to assign attr = 0
            thrust::device_vector<Variable> U_vars(data->numSupplies);
            u_vars_ptr = thrust::raw_pointer_cast(U_vars.data());
            thrust::device_vector<Variable> V_vars(data->numDemands);
            v_vars_ptr = thrust::raw_pointer_cast(V_vars.data());
            thrust::device_vector<flowInformation> device_flows(feasible_flows, 
                feasible_flows + (data->numSupplies+data->numDemands-1));
            device_flows_ptr = thrust::raw_pointer_cast(device_flows.data());
            
            // thrust::device_vector<MatrixCell> device_costMatrix(costMatrix, 
            //     costMatrix + (data->numSupplies*data->numDemands));
            // device_costMatrix_ptr = thrust::raw_pointer_cast(device_costMatrix.data());
    
            // Make any one as 0
            Variable default_assign; 
            default_assign.value = 0.0;
            default_assign.assigned = true;
            U_vars[0] = default_assign;
    
            std::cout<<"\tTESTING CURRENT BFS: Solving the UV System"<<std::endl;
            
            // Start solving the system of eqn's (complementary slackness conditions)
            // Solve the system in at most m+n-1 iterations        
            // assigned cells -> m + n, and m+n-1 linearly independent equations to solve
            // For each of the m + n - 1 assignements in the assignment tree ->>
            // C_ij = u_i + v_j
    
            dim3 dimGrid(ceil(1.0*(data->numSupplies+data->numDemands-1)/blockSize),1,1);
            dim3 dimBlock(blockSize,1,1);
    
            // Solve the system of linear equations in the following kernel >>
            for (int i=0; i < (data->numSupplies+data->numDemands-1); i++) {
                assign_next <<< dimGrid, dimBlock >>> (device_flows_ptr, device_costMatrix_ptr, u_vars_ptr, 
                    v_vars_ptr, data->numSupplies, data->numDemands);
            }
    
            std::cout<<"\tComputing pivots ..."<<std::endl;
            int _block_size = 8;
            dim3 dimGrid2(ceil(1.0*data->numDemands/_block_size), ceil(1.0*data->numSupplies/_block_size), 1);
            dim3 dimBlock2(_block_size, _block_size, 1); // Refine this based on device query
    
            // thrust::device_vector<float> device_reducedCosts(data->numSupplies*data->numDemands);
            // device_reducedCosts_ptr = thrust::raw_pointer_cast(device_reducedCosts.data());
            // computeReducedCosts<<< dimGrid2, dimBlock2 >>>(u_vars_ptr, v_vars_ptr, device_costMatrix_ptr, device_reducedCosts_ptr, data->numSupplies, data->numDemands);
            execute_parallel_pivot<<< dimGrid2, dimBlock2 >>>(u_vars_ptr, v_vars_ptr, adjMtx_ptr, 
                visited_ptr, device_costMatrix_ptr, 
                data->supplies, data->demands,
                data->numSupplies, data->numDemands);
            cudaDeviceSynchronize();
            std::cout<<"\tPivoting - complete!"<<std::endl;
    
            // cudaMemcpy(reduced_costs, device_reducedCosts_ptr, data->numDemands*data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost);
    
            // for (size_t i = 0; i < device_reducedCosts.size(); i++) 
            // {
            // std::cout << "device_reducedCosts[" << i << "] = " << device_reducedCosts[i] << std::endl;
            // }
            
            // thrust::device_ptr<float> _start = device_reducedCosts.data();
            // thrust::device_ptr<float> minimum = thrust::min_element(_start, (_start+data->numSupplies*data->numDemands));
            // int min_indx = thrust::distance(_start, minimum);
            // if (device_reducedCosts[min_indx] < 0) { 
            //     pivot_row = min_indx/data->numDemands;
            //     pivot_col = min_indx - pivot_row*data->numDemands;
            // }
        }
    
        else{
            std::cout<<"Invalid method of dual calculation!"<<std::endl;
            std::exit(-1);
        }
    }
    
    
    
    void uvModel_parallel::execute2(){
        
        // **************************************
        // Finding Basic Feasible Solution >>
        // **************************************
    
        generate_initial_BFS();
    
        // **************************************
        // Modified Distribution Method (u-v method) - parallel (improve the BFS solution)
        // **************************************
        
        // Use feasible flows and make a graph in global memory >> Make a adjacency matrix 
        int V = data->numSupplies+data->numDemands;
        thrust::host_vector<int> h_adjMtx(V*V, 0);
        // std::vector<int> h_adjMtx(V*V, 0), visited_node(V, 0);
    
        for (int i = 0; i < V-1; i++) {
            flowInformation _edge = feasible_flows[i];
            h_adjMtx[_edge.source*V + (_edge.destination+data->numSupplies)] = _edge.qty;
            h_adjMtx[(_edge.destination+data->numSupplies)*V + _edge.source] = _edge.qty;
        }
        
        // int src = 1, dst = 0;
    
        // h_adjMtx[src*V + (dst+data->numSupplies)] = 1;
        // h_adjMtx[(dst+data->numSupplies)*V + src] = 1;
        // pathEdge * loop = (pathEdge *)malloc(sizeof(pathEdge));
        // loop->index = src;
        // loop->next = NULL;
    
            // Test micro-DFS
        // micro_DFS(visited_node, h_adjMtx, V, src, -1, loop);
        // std::cout<<"Micro DFS completed!"<<std::endl;
    
        // while(loop->next!=NULL) {
        //     std::cout<<"Cycle Entries - "<<loop->index<<std::endl;
        //     loop = loop->next;
        // }
        // thrust::device_vector<int> d_adjMtx = h_adjMtx;
        int * adjMtx_ptr = thrust::raw_pointer_cast(h_adjMtx.data());
        bool result = true;
        int iteration_counter = 0;
    
    
        while (result) {
            thrust::host_vector<int> visited_node(V, 0);
            int * visited_ptr = thrust::raw_pointer_cast(visited_node.data());
            perform_pivoting_parallel(visited_ptr, adjMtx_ptr);
            iteration_counter++;
            std::cout<<"Performing parallel pivot - round : "<<iteration_counter<<std::endl;
            result = false;
        }
        
    
    }

    void uvModel_parallel::perform_pivot() 
{
    // there's a pivot_row and a pivot_col
    // pivot_row = 0;
    // pivot_col = 4;

    // Do some preprocess >>
    // Now create a tree using the flows available:
    // initialize edges 
    std::vector<Edge> edges;
    for (int i=0; i < data->numDemands+data->numSupplies-1; i++) {
        flowInformation f = feasible_flows[i];
        Edge e = {.left = f.source, .right = f.destination+data->numSupplies};
        // in the vertex list of bipartite graph ..
        // the right hand side nodes start after an offset 
        edges.push_back(e);
    }
        
    // std::cout<<"edges :"<<edges.size()<<std::endl;

    // Add the incoming arc -
    Edge entering = {.left=pivot_row, .right=data->numSupplies+pivot_col};
    edges.push_back(entering);

    // total number of nodes in the graph (0 to 11)
    int n = data->numSupplies + data->numDemands;
    // build a graph from the given edges
    // since it was a tree and we have one additional edge
    // there exist exactly one cycle in the graph >>
    Graph graph(edges, n);
 
    // to keep track of whether a vertex is discovered or not >> some book-keeping stuff
    std::vector<bool> discovered(n);
    std::vector<int> alternating_path;

    // Perform DFS traversal from the vertex that is pivot-row >> to form a loop
    // std::cout<<"\tIMPROVING CURRENT BFS: Finding a loop in the assigment tree!"<<std::endl;
    DFS(graph, pivot_row, discovered, alternating_path, -1);
    alternating_path.push_back(pivot_row);

    // Dubugging stuff ::
    // if (DFS(graph, pivot_row, discovered, alternating_path, -1)) {
    //     std::cout << "The graph contains a cycle"<<std::endl;
    // }
    // else {
    //     std::cout << "The graph doesn't contain any cycle"<<std::endl;
    // }
    // std::cout<<"Initial Flows :: "<<std::endl;
    // for (int i=0; i< matrixDemands+matrixSupplies-1; i++){
    //     std::cout<<flows[i]<<std::endl;
    // }
    // for (int i=0; i < alternating_path.size(); i++) {
    //     std::cout<<"Alternating Path - "<<alternating_path[i]<<std::endl;   
    // }
    // exit(0);

    // std::cout<<"\tIMPROVING CURRENT BFS: Calculating pivot flow!"<<std::endl;
    int pivot_qty = INT_MAX;
    int pivot_idx = -1;
    int temp_qty, temp_idx; 
    // Find the minimum flow that will be decreased and it's index >>
    for (int i=1; i < alternating_path.size()-1; i++) {
        if (i%2 == 1){
            temp_idx = flow_indexes[{alternating_path[i+1], alternating_path[i]-data->numSupplies}];
            temp_qty = feasible_flows[temp_idx].qty;
            if (temp_qty < pivot_qty) {
                pivot_qty = temp_qty;
                pivot_idx = temp_idx;
            }    
        }   
    }
        
    // std::cout<<"\tPivot Qty :: "<<pivot_qty<<std::endl;
    // UPDATE FLOWS >>
    // leave the first edge in the alternating path because it is the entering arc 
    // update it after updating others - the reason is because it is not a part of flow/flow indexes
    // we'll handle it like pros'
    // start with decrease
    // std::cout<<"\tIMPROVING CURRENT BFS: Updating flows!"<<std::endl;
    int j = -1;
    for (int i=1; i<alternating_path.size()-1; i++) {
        if (i%2 == 1){
            temp_idx = flow_indexes[{alternating_path[i+1], alternating_path[i]-data->numSupplies}];
            // std::cout<<"Pivot - Flow : row "<<i<<" = "<<alternating_path[i+1]<<" , col = "<<alternating_path[i]-matrixSupplies<<std::endl;
        }
        else {
            temp_idx = flow_indexes[{alternating_path[i], alternating_path[i+1]-data->numSupplies}];
            // std::cout<<"Pivot - Flow : row "<<i<<" = "<<alternating_path[i]<<" , col = "<<alternating_path[i+1]-matrixSupplies<<std::endl;
        }
        feasible_flows[temp_idx].qty += j*pivot_qty;
        j *= -1; // j will alternate between -1 and +1 
    }

    // Update information on entering arc
    flow_indexes.erase(std::make_pair(feasible_flows[pivot_idx].source, feasible_flows[pivot_idx].destination));
    // std::cout<<"\tResidual Qty :: "<<flows[pivot_idx].qty<<std::endl;
    feasible_flows[pivot_idx].qty = pivot_qty;
    feasible_flows[pivot_idx].source = pivot_row;
    feasible_flows[pivot_idx].destination = pivot_col;
    flow_indexes.insert(std::make_pair(std::make_pair(pivot_row, pivot_col),pivot_idx));

}