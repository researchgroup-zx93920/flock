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
/*
FUTURE USE : switch back between tree and linear-equation methods
*/

#define PIVOTING_STRATEGY "default"
/*
FUTURE USE : switch back between parallel, racing and barriers
*/

// >>>>>>>>>> END OF PARAMETERS // 

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

struct is_zero
{
    __host__ __device__ bool operator()(const flowInformation &x)
    {
        return (x.qty == 0);
    }
};

__host__ class uvModel_parallel
{

public:

    ProblemInstance * data;
    flowInformation * optimal_flows;

    flowInformation * feasible_flows, * d_flows_ptr; 
    // Feasible flow at any point in the algorithm flow
    // Useful for mapping data structs on CPU - GPU interop
    std::map<std::pair<int,int>, int> flow_indexes; 
    // The flows at given cell (i,j) is available at this index in flows

    MatrixCell * costMatrix; // Useful for vogel's 
    MatrixCell * device_costMatrix_ptr;
    int * d_adjMtx_ptr;
    float * d_reducedCosts_ptr, * reduced_costs, * d_costs_ptr, * u_vars_ptr, * v_vars_ptr;
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
    void get_reduced_costs();
    void perform_pivot(bool &result);
    // void perform_pivoting_parallel(int * visited_ptr, int * adjMtx_ptr);
    void solve();

    // Developer Facility Methods >>
    void view_uvra();
};

// INTERMEDIATE STRUCTS - ONLY CONCERNED WITH CUDA >> 


