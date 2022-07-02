#include "../parallel_structs.h"
#include "../parallel_kernels.h"
#include "../IBFS_vogel.h"
#include "../IBFS_nwc.h"
#include "PIVOT_ss.h"

/*
Algorithm alternative to solve transportation problem

Step 1: Generate initial BFS - M + N - 1 edges selected, 
Compute is performed on device while flow is created loaded on host
Step 2: Repeat :: 
    2.1: Find all the possible non-conflicting recirculations with negative costs
    2.2: Any reduced cost < 0 - repeat, exit otherwise
Step 3: Return M + N - 1 flows 
*/
class ssModel_parallel
{

public:

    ProblemInstance * data;
    flowInformation * optimal_flows;

    flowInformation * feasible_flows;  // Feasible flow at any point in the algorithm flow
                                       // Useful for mapping data structs on CPU - GPU interop

    void execute();
    void create_flows();

    // Model Statistics >>
    double pivot_time, cycle_discovery_time, resolve_time, adjustment_time;

    double objVal;
    int totalIterations;
    double totalSolveTime;

    ssModel_parallel(ProblemInstance * problem, flowInformation * flows);
    ~ssModel_parallel();

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
    // Useful for initial basic feasible solution (IBFS), Dual and interchange between simplex flow  
    MatrixCell * costMatrix, * device_costMatrix_ptr; 
    float * d_costs_ptr;
        
    // ###############################
    // SIMPLEX PIVOT | Sequencial and Parallel pivoting strategy
    // ###############################
    PivotHandler pivot;
    PivotTimer timer;

    // ###############################
    // CLASS METHODS | Names are self explanatory - doc strings are available on the definition
    // ###############################
    void generate_initial_BFS();
    void perform_pivot(bool &result, int iteration);
    void solve();

    // Developer Facility Methods for potential analysis, debugging etc. >>
    void view_uv();
    void view_tree();

};