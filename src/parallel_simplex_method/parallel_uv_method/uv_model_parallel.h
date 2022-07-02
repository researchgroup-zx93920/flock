#include "../parallel_structs.h"
#include "../parallel_kernels.h"
#include "../IBFS_vogel.h"
#include "../IBFS_nwc.h"
#include "DUAL_solver.h"
#include "PIVOT_uv.h"

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
class uvModel_parallel
{

public:

    ProblemInstance * data;
    flowInformation * optimal_flows;

    flowInformation * feasible_flows;  // Feasible flow at any point in the algorithm flow
                                       // Useful for mapping data structs on CPU - GPU interop

    void execute();
    void create_flows();

    // Model Statistics >>
    double uv_time, reduced_cost_time, pivot_time, cycle_discovery_time, resolve_time, adjustment_time;

    double objVal;
    int totalIterations;
    double totalSolveTime;

    uvModel_parallel(ProblemInstance * problem, flowInformation * flows);
    ~uvModel_parallel();

private:

    // Internal Flagging and constants
    bool solved;
    Graph graph;

    // ###############################
    // PREPROCESS and POSTPROCESS
    // ###############################

    // Containers for data exchange between functions 
    // - reduced cost of edges on the device
    // - cost of flow through an edge
    // Useful for initial basic feasible solution (IBFS), Dual and inbetween simplex
    // Having row column store with reducedcosts allows for reordering during DFS kernels    
    MatrixCell * costMatrix, * device_costMatrix_ptr; 
    float * d_costs_ptr, * d_reducedCosts_ptr;

    // TEMPORARY or DEBUGGING TOOLS
    float * h_reduced_costs;

    // ###############################
    // DUAL and REDUCED COSTS (test for optimality)
    // ###############################
    DualHandler dual;
        
    // ###############################
    // SIMPLEX PIVOT | Sequencial and Parallel pivoting strategy
    // ###############################
    PivotHandler pivot;
    PivotTimer timer;

    // ###############################
    // CLASS METHODS | Names are self explanatory - doc strings are available on the definition
    // ###############################
    void generate_initial_BFS();
    void solve_uv();
    void get_reduced_costs();
    void perform_pivot(bool &result, int iteration);
    void solve();

    // Developer Facility Methods >>
    void view_uv();
    void view_reduced_costs();
    void count_negative_reduced_costs();
    void view_tree();

};

