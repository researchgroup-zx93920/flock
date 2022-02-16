#include "uv_model_parallel.h"

/*
Kernel to convert float cost matrix to the MatrixCell objects
*/
__global__ void createCostMatrix(MatrixCell *d_costMtx, float *d_costs, int n_supplies, int n_demands)
{

    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;

    if (s < n_supplies && d < n_demands)
    {
        int id = s * n_demands + d;
        MatrixCell _c = {.row = s, .col = d, .cost = d_costs[id]};
        d_costMtx[id] = _c;
    }
}

__global__ void computeReducedCosts(Variable * u_vars, Variable * v_vars, MatrixCell * device_costMatrix, float * device_reducedCosts_ptr, 
    int matrixSupplies, int matrixDemands) {

        int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

        if (row_indx < matrixSupplies && col_indx < matrixDemands) {
            // r =  C_ij - (u_i + v_j);
            float r = device_costMatrix[row_indx*matrixDemands+col_indx].cost - u_vars[row_indx].value - v_vars[col_indx].value;
            device_reducedCosts_ptr[row_indx*matrixDemands+col_indx] = r;
        }
}

/*
Perform DFS on the graph and returns true if any back-edge is found in the graph
At a thread level this may be implemented using a stack
*/
__host__ bool DFS(Graph const &graph, int v, std::vector<bool> &discovered, std::vector<int> &loop, int parent)
{
    // mark the current node as discovered
    discovered[v] = true;
    // std::cout<<"Parent : "<<parent<<" | V : "<<v<<std::endl;
 
    // do for every edge (v, w)
    for (int w: graph.adjList[v])
    {
        // if `w` is not discovered
        if (!discovered[w])
        {
            if (DFS(graph, w, discovered, loop, v)) {
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
    return false;
}

uvModel_parallel::uvModel_parallel(ProblemInstance *problem, flowInformation *flows)
{

    std::cout << std::endl <<"Initializing UV Model (parallel)" << std::endl;
    data = problem;
    optimal_flows = flows;

    // Allocate memory based on the problem size
    feasible_flows = (flowInformation *) malloc(sizeof(flowInformation) * ((data->numSupplies) + (data->numDemands) - 1));
    costMatrix = (MatrixCell *) malloc(sizeof(MatrixCell) * (data->numSupplies) * (data->numDemands));
    
    float * d_costs;
    cudaMalloc((void **) &d_costs, sizeof(float) * (data->numSupplies) * (data->numDemands));
    cudaMalloc((void **) &device_costMatrix_ptr, sizeof(MatrixCell) * (data->numSupplies) * (data->numDemands));
    cudaMemcpy(d_costs, data->costs, sizeof(float) * (data->numSupplies) * (data->numDemands), cudaMemcpyHostToDevice);

    dim3 _dimBlock(32,32,1);
    dim3 _dimGrid(ceil(1.0*data->numSupplies/32), ceil(1.0*data->numDemands/32), 1);
    createCostMatrix<<<_dimGrid, _dimBlock>>>(device_costMatrix_ptr, d_costs, data->numSupplies, data->numDemands);
    cudaDeviceSynchronize();

    cudaMemcpy(costMatrix, device_costMatrix_ptr, sizeof(MatrixCell) * (data->numSupplies) * (data->numDemands), cudaMemcpyDeviceToHost);
    cudaFree(d_costs);
    
    // Creating cost Matrix Cells objects -

    // for (int i = 0; i < data->numSupplies; i++)
    // {
    //     for (int j = 0; j < data->numDemands; j++)
    //     {
    //         int _key = i * data->numDemands + j;
    //         std::cout<<costMatrix[_key]<<std::endl;
    //     }
    // }

    std::cout << "An uv_model_parallel object was successfully created" << std::endl;
}

uvModel_parallel::~uvModel_parallel()
{
    // On thrust layer - replace by thrust eqv. of free
    // free(costMatrix);
    // free(feasible_flows);
}

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

    // The array feasible flows holds the initial basic feasible solution
}

/*
Given a u and v vector on device - computes the reduced costs for all the potential flows
*/
void uvModel_parallel::solve_uv()
{
    // Void :: solve_uv method doest that implicitly 
}

void uvModel_parallel::get_reduced_costs(int &pivot_row, int &pivot_col){
    /* 
    There could be multiple ways to solve the dual costs for a given problem
    Packaged method derive u's and v's and load them onto the U_vars and V_vars attributes

    1. Use a tree method for dual (trickel down approach)
    2. Use a off-the-shelf solver for linear equation 
    */
    // std::cout<<"\tTESTING CURRENT BFS: Determining Dual Costs"<<std::endl;

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

        // std::cout<<"TESTING CURRENT BFS: Solving the UV System"<<std::endl;
        
        // Start solving the system of eqn's (complementary slackness conditions)
        // Solve the system in at most m+n-1 iterations        
        // assigned cells -> m + n, and m+n-1 linearly independent equations to solve
        // For each of the m + n - 1 assignements in the assignment tree ->>
        // C_ij = u_i + v_j

        dim3 dimGrid(ceil(1.0*(data->numSupplies+data->numDemands-1)/blockSize),1,1);
        dim3 dimBlock(blockSize,1,1);

        // Solve the system of linear equations in the following kernel >>
        for (int i=0; i < (data->numSupplies+data->numDemands-1);i++) {
            assign_next <<< dimGrid, dimBlock >>> (device_flows_ptr, device_costMatrix_ptr, u_vars_ptr, v_vars_ptr, data->numSupplies, data->numDemands);
        }

        // std::cout<<"Computing Reduced Costs ..."<<std::endl;
        int _block_size = 16;
        dim3 dimGrid2(ceil(1.0*data->numDemands/_block_size), ceil(1.0*data->numSupplies/_block_size), 1);
        dim3 dimBlock2(_block_size, _block_size, 1); // Refine this based on device query

        thrust::device_vector<float> device_reducedCosts(data->numSupplies*data->numDemands);
        device_reducedCosts_ptr = thrust::raw_pointer_cast(device_reducedCosts.data());
        computeReducedCosts<<< dimGrid2, dimBlock2 >>>(u_vars_ptr, v_vars_ptr, device_costMatrix_ptr, device_reducedCosts_ptr, data->numSupplies, data->numDemands);
        cudaDeviceSynchronize();
        // std::cout<<"Computing Reduced Costs - complete!"<<std::endl;

        // cudaMemcpy(reduced_costs, device_reducedCosts_ptr, data->numDemands*data->numSupplies*sizeof(float), cudaMemcpyDeviceToHost);

        // for (size_t i = 0; i < device_reducedCosts.size(); i++) 
        // {
        // std::cout << "device_reducedCosts[" << i << "] = " << device_reducedCosts[i] << std::endl;
        // }
        
        thrust::device_ptr<float> _start = device_reducedCosts.data();
        thrust::device_ptr<float> minimum = thrust::min_element(_start, (_start+data->numSupplies*data->numDemands));
        int min_indx = thrust::distance(_start, minimum);
        if (device_reducedCosts[min_indx] < 0) { 
            pivot_row = min_indx/data->numDemands;
            pivot_col = min_indx - pivot_row*data->numDemands;
        }
    }
    else{
        std::cout<<"Invalid method of dual calculation!"<<std::endl;
        std::exit(-1);
    }
}

void uvModel_parallel::perform_pivot(int pivot_row, int pivot_col) 
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

void uvModel_parallel::execute(){
    
    // **************************************
    // Finding BFS >>
    // **************************************

    generate_initial_BFS();

    // **************************************
    // Modified Distribution Method (u-v method) - parallel (improve the BFS solution)
    // **************************************
    // std::cout<<"Improving Initial Basic Feasible Solution .. "<<std::endl;

    reduced_costs = (float *) malloc(sizeof(float)*data->numSupplies*data->numDemands);
    bool result = false;
    int iteration_counter = 1;
    auto start = std::chrono::high_resolution_clock::now();
    while (!result) {

        pivot_row = -1; // row with least reduced cost
        pivot_col = -1; // col with least reduced cost

        // solve_uv();
        get_reduced_costs(pivot_row, pivot_col);

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
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	double solution_time = duration.count();
    std::cout<<"Simplex completed in : "<<solution_time<<" secs. and "<<iteration_counter<<" iterations."<<std::endl;
}

void uvModel_parallel::create_flows()
{
    memcpy(optimal_flows, feasible_flows, (data->numSupplies+data->numDemands-1)*sizeof(flowInformation));
}