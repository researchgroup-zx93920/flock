#include "../parallel_structs.h"
#include "../parallel_kernels.h"
#include "../IBFS_vogel.h"
#include "../IBFS_nwc.h"
#include "../parallel_ss_method/PIVOT_ss.h"
#include "../parallel_uv_method/DUAL_solver.h"
#include "../parallel_uv_method/PIVOT_uv.h"

/* Starts on parallel pivot and then  */
class switchModel_parallel
{

public:

    ProblemInstance * data;
    flowInformation * optimal_flows;

    flowInformation * feasible_flows;  // Feasible flow at any point in the algorithm flow
                                       // Useful for mapping data structs on CPU - GPU interop

    void execute();
    void create_flows();

    double deviceCommunicationTime; // Not implemented
    // Model Statistics >>
    double uv_time, reduced_cost_time,pivot_time, cycle_discovery_time, resolve_time, adjustment_time;

    double objVal;
    int totalIterations;
    double totalSolveTime;

    switchModel_parallel(ProblemInstance * problem, flowInformation * flows);
    ~switchModel_parallel();

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
    MatrixCell * costMatrix, * device_costMatrix_ptr, * d_reducedCosts_ptr; 
    float * d_costs_ptr;
        
    // ###############################
    // SIMPLEX PIVOT | Sequencial and Parallel pivoting strategy
    // ###############################
    PivotHandler pivot;
    PivotTimer timer;

    DualHandler dual;

    // ###############################
    // CLASS METHODS | Names are self explanatory - doc strings are available on the definition
    // ###############################
    void generate_initial_BFS();
    void perform_pivot(bool &result, int iteration, int &mode);
    void solve();
    void solve_uv();
    void get_reduced_costs();

    // Developer Facility Methods >>
    void view_uv();
    void view_tree();

};



