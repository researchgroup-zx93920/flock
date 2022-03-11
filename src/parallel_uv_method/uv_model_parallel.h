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

#define BFS_METHOD "vam"
/*
nwc : Northwest Corner - sequential implementation
vam : vogel's approximation - parallel regret implementation
*/

#define CALCULATE_DUAL "tree"
/*
FUTURE USE : switch back between tree and linear-equation methods
*/

#define PIVOTING_STRATEGY "sequencial"
/*
sequencial : perform pivoting one at a time based on dantzig's rule
parallel : perform parallel pivoting
*/

#define MAX_ITERATIONS 10000

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

    flowInformation * feasible_flows;  
    // Feasible flow at any point in the algorithm flow
    // Useful for mapping data structs on CPU - GPU interop
    std::map<std::pair<int,int>, int> flow_indexes; 
    // The flows at given cell (i,j) is available at this index in flows

    void execute();
    void get_dual_costs();
    void create_flows();

    uvModel_parallel(ProblemInstance * problem, flowInformation * flows);
    ~uvModel_parallel();

private:

    bool solved;
    int V;

    flowInformation * d_flows_ptr; // pointer to flow objects on device
    MatrixCell * costMatrix; // Useful for vogel's 
    MatrixCell * device_costMatrix_ptr;

    int * d_adjMtx_ptr, * h_adjMtx_ptr; 
    float * d_flowMtx_ptr, * h_flowMtx_ptr;
    /*
        Adjacency matrix is a upper triangular matrix to store the tree inforamtion
        !! DOC PENDING !!
    */

    // Pointers to adjacency matrix of feasible flow tree in device and host respectively
    // Pointer to flowMtx, adjMtx only represents adjaceency, flow matrix contains flow values
    // Such a separation has been created to resolve degeneracies - 
    // sometimes zero flows would make life better :D
    
    float * d_reducedCosts_ptr, * d_costs_ptr, * u_vars_ptr, * v_vars_ptr;
    // Pointers to vectors relevant to pivoting
    //  - reduced cost of edges on the device
    //  - cost of flow through an edge
    //  - dual costs towards supply constraints
    //  - dual costs towards demand constraints

    // Temporary >> 
    float * h_reduced_costs;

    int pivot_row, pivot_col;
    // Useful for the case of sequencial pivoting
    
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

/*
Perform DFS on the graph and returns true if any back-edge is found in the graph
At a thread level this may be inefficient given the data structures used, thus 
another method exists for the thread specific DFS

__host__ bool modern_DFS(Graph const &graph, int v, std::vector<bool> &discovered, std::vector<int> &loop, int parent)
{

    discovered[v] = true; // mark the current node as discovered
 
    // do for every edge (v, w)
    for (int w: graph.adjList[v])
    {
        // if `w` is not discovered
        if (!discovered[w])
        {
            if (modern_DFS(graph, w, discovered, loop, v)) {
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
    discovered[v] = false; // Reset
    return false;
}
*/

/* 
Perform parallel DFS at the thread level using adjacency matrix - this one makes a stack called loop 
that stores the traversal \n
Note that:  This is a device function, expected to be light on memory, for well connected graphs, 
this will be helped thorugh cache, but that needs to checked through experiments.

__device__ bool micro_DFS(int * visited, int * adjMtx, pathEdge * loop, int V, int i, int parent) {
    
    atomicAdd(&visited[i], 1); // perform atomic add on visited - i - so that other threads aren't exploring this
    // visited[i] = 1; // >> sequential testing/debugging

    // For every neighbor of i 
    for(int j=0; j<V; j++) {
        if (adjMtx[i*V + j]>0) {

            // If it is not visited by anybody else =>
            if(visited[j]==0) {
                // Check if there's a forward edge 
                pathEdge * _loop = (pathEdge *) malloc(sizeof(pathEdge));
                _loop->index = j;
                _loop->next = NULL;
                if (micro_DFS(visited, adjMtx, V, j, i, _loop)) {
                    // loop_push
                    loop->next = _loop;
                    return true;
                }
                else {
                    free(_loop);
                    // Jumps to return false; 
                }
            }
            // We found a backward edge
            else if (j != parent) {
                // loop_push
                pathEdge * _loop = (pathEdge *) malloc(sizeof(pathEdge));
                _loop->index = j;
                _loop->next = NULL;
                loop->next = _loop;
                return true;
            }
        }
    }
    
    atomicSub(&visited[i], 1); // Reset visited flag to let other threads explore
    // visited[i] = 0; // >> sequential testing/debugging
    return false;
}
*/


