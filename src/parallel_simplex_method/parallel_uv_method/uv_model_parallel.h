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

    double deviceCommunicationTime; // Not implemented
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
    MatrixCell * costMatrix, * device_costMatrix_ptr, * d_reducedCosts_ptr; 
    float * d_costs_ptr;
        
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

// ARCHIVES | were used at some point >> 

/*
Perform DFS on the graph and returns true if any back-edge is found in the graph
At a thread level this may be inefficient given the data structures used, thus 
another method exists for the thread specific DFS

struct rowNodes {
    std::vector<int> child;
    bool covered;
};

struct colNodes {
    std::vector<int> parent;
    bool covered;
};

struct Edge {
    int left, right;
};

// A class to represent a graph object
class Graph
{
public:
 
    // a vector of vectors to represent an adjacency list
    std::vector<std::vector<int>> adjList;
 
    // Graph Constructor
    Graph(std::vector<Edge> const &edges, int n)
    {
        // resize the vector to hold `n` elements of type `vector<int>`
        adjList.resize(n);
 
        // add edges to the undirected graph
        for (auto &edge: edges)
        {
            adjList[edge.left].push_back(edge.right);
            adjList[edge.right].push_back(edge.left);
        }
    }
};


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


