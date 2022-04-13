#include <iostream>
#include <chrono>

#include "parallel_structs.h"
#include "parallel_kernels.h"
#include "IBFS_vogel.h"
#include "IBFS_nwc.h"
#include "DUAL_solver.h"
// #include "PIVOT_dfs.h"

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
    std::map<std::pair<int,int>, int> flow_indexes; 
    // The flows at given cell (i,j) is available at this index in flows

    void execute();
    void create_flows();

    double deviceCommunicationTime;

    uvModel_parallel(ProblemInstance * problem, flowInformation * flows);
    ~uvModel_parallel();

private:

    // Internal Flagging and constants
    bool solved;
    int V;

    // Containers for data exchange between functions 
        MatrixCell * costMatrix; // Useful for initial basic feasible solution (IBFS)
    
        // Adjacency matrix and flow matrix together represent the feasible tree
        // Such a separation has been created to resolve degeneracies - 
        // sometimes zero flows would make life better :D

        // For a simplex iteration, current feasible tree is maintained on both host (h_) and device (d_)
        int * d_adjMtx_ptr, * h_adjMtx_ptr;  
        float * d_flowMtx_ptr, * h_flowMtx_ptr;
        /*
        IMPORTANT | adjMtx and flowMtx has been designed with special consideration

        Adjacency matrix is a upper triangular matrix to store the tree inforamtion
        Let's say you're looking at edge (i,j)
        Now you want to find flow on i,j in the feasible tree
        Then, note this: 
        1. (i,j) is a feasible edge if adjMtx(i,j) > 0
        2. Further, flow on (i,j) = flow[adjMtx(i,j)-1], 
        In words: entries in  adjMtx point to the position of flow values stored in flowMtx
        */
    
    // ###############################
    // PREPROCESS and POSTPROCESS
    // ###############################
        MatrixCell * device_costMatrix_ptr; 

    // ###############################
    // DUAL and REDUCED COSTS (test for optimality)
    // ###############################
    // Pointers to vectors relevant to optimality tests
    //  - reduced cost of edges on the device
    //  - cost of flow through an edge
    //  - dual costs towards supply constraints
    //  - dual costs towards demand constraints
        float * d_reducedCosts_ptr, * d_costs_ptr, * u_vars_ptr, * v_vars_ptr;
    
        // DUAL :: Solving system of equations 
        float u_0 = 0, u_0_coef = 1;
        int * d_csr_offsets, * d_csr_columns;
        float * d_csr_values, * d_x, * d_A, * d_b;
        int64_t nnz;

        // DUAL :: Solving using a breadth first traversal on Tree
        Variable * U_vars, * V_vars;

        // DUAL :: Temporary
        float * h_reduced_costs;

    // ###############################
    // SIMPLEX PIVOT | Sequencial and Parallel pivoting strategy
    // ###############################

        // Useful for the sequencial strategy >>
        int pivot_row, pivot_col;
        
        // Useful for the parallel strategy >>
        int * backtracker, * depth, * loop_min_from, * loop_min_to, * loop_min_id;
        float * loop_minimum;
        stackNode * stack;
        vertex_conflicts * v_conflicts;
        bool * visited;

        vertex_conflicts _vtx_conflict_default;
    

    // ###############################
    // CLASS METHODS | Names are self explanatory - doc strings are available on the definition
    // ###############################
    void generate_initial_BFS();
    void solve_uv();
    void get_reduced_costs();
    void perform_pivot(bool &result);
    void solve();

    // Developer Facility Methods >>
    void view_uv();
    void view_reduced_costs();
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


