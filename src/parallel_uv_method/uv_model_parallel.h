#include <iostream>
#include <chrono>

#include "../structs.h"
#include "./parallel_structs.h"
#include "IBFS_vogel.h"
#include "IBFS_nwc.h"
#include "DUAL_tree.h"

// PARAMETERS
#define DETAILED_LOGS 1

/*
0 : Silent model
1 : Verbose model
*/

#define BFS_METHOD "nwc"

/*
nwc : Northwest Corner - sequential implementation
vam : vogel's approximation - parallel regret implementation
*/

#define CALCULATE_DUAL "tree"

// END OF PARAMETERS 

/*
Algorithm alternative to solve transportation problem

Step 1: Generate initial BFS - M + N - 1 edges selected, 
Compute is performed on device while flow is created loaded on host
Step 2: Repeat :: 
    2.1: Solve uv and compute reduced costs, uv generated stay on the device for 
    reduced cost compute - facility functions are provided to retrieve the output
    2.2: Any reduced cost < 0 - repeat, exit otherwise
Step 3: Return M + N - 1 flows 
*/
__host__ class uvModel_parallel
{

public:

    ProblemInstance * data;
    flowInformation * optimal_flows;

    flowInformation * feasible_flows, * device_flows_ptr; // Feasible flow at any point in the algorithm flow
    std::map<std::pair<int,int>, int> flow_indexes; // The flows at given cell (i,j) is available at this index in flows
    MatrixCell * costMatrix;
    Variable * u_vars_ptr, * v_vars_ptr;
    MatrixCell * device_costMatrix_ptr;
    float * device_reducedCosts_ptr, * reduced_costs;
    int pivot_row, pivot_col;

    void execute();
    void get_dual_costs();
    void create_flows();

    uvModel_parallel(ProblemInstance * problem, flowInformation * flows);
    ~uvModel_parallel();

private:
    bool solved;
    void generate_initial_BFS();
    void solve_uv();
    void get_reduced_costs(int &pivot_row, int &pivot_col);
    void perform_pivot(int pivot_row, int pivot_col);
    void solve();
};

// INTERMEDIATE STRUCTS - ONLY CONCERNED WITH CUDA >> 


