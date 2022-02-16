#include "uv_method.h"

std::ostream& operator << (std::ostream& o, const Variable& x) {
    o << x.value;
    return o;
}

// Data structure to store a graph edge
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
 
// Perform DFS on the graph and returns true if any back-edge is found in the graph
bool DFS(Graph const &graph, int v, std::vector<bool> &discovered, std::vector<int> &loop, int parent)
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

__global__ void assign_next(flowInformation * flows, MatrixCell * device_costMatrix, Variable * u_vars, 
    Variable * v_vars, int matrixSupplies, int matrixDemands) {
    
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    // __shared__ >> Can solve the equations locally before taking them to global

    if (indx < matrixSupplies + matrixDemands - 1) {
        flowInformation eqn = flows[indx];
        if (u_vars[eqn.row].assigned && !v_vars[eqn.col].assigned) {
            // In this case >> v_j = c_ij - u_i
            Variable var;
            var.assigned = true;
            var.value = device_costMatrix[eqn.row*matrixDemands+eqn.col].cost - u_vars[eqn.row].value;
            v_vars[eqn.col] = var;
        }
        else if (!u_vars[eqn.row].assigned && v_vars[eqn.col].assigned) {
            // In this case >> u_j = c_ij - v_j
            Variable var;
            var.assigned = true;
            var.value = device_costMatrix[eqn.row*matrixDemands+eqn.col].cost -  v_vars[eqn.col].value;
            u_vars[eqn.row] = var;
        }
    }
}

/* Updates flows and flow indexes towards optimal assignment
*/
__host__ bool test_and_improve(MatrixCell * costMatrix, flowInformation * flows, 
    std::map<std::pair<int,int>, int> &flow_indexes, float * reduced_costs,
    int matrixSupplies, int matrixDemands){
        
        

        // Questions ::
        // 1. Diagonal zero - corner case for U-V method (Resolution: Recall discussion with Hem, degenerate case)
        
        cudaMemcpy(reduced_costs, device_reducedCosts_ptr, matrixDemands*matrixSupplies*sizeof(float), cudaMemcpyDeviceToHost);
        
        // Pivoting - sequencial >>
        int pivot_row = 0;
        int pivot_col = 0;
        float most_negative = 0.0;
        // Look for most negative reduced cost >> 
        // In parallel - just look for a negative reduced cost 
        // IDEA: do a reduce max in thrust - skip memcpy operation 
        for (int i = 0; i < matrixSupplies*matrixDemands; i++) {
            if (reduced_costs[i] < 0 && reduced_costs[i] < most_negative) {
                most_negative = reduced_costs[i];
                pivot_col = i%matrixDemands;
                pivot_row = (i - pivot_col)/matrixDemands;
            }
        }
        
        if (most_negative == 0){
            std::cout<<"\tTESTING CURRENT BFS: No Negative reduced costs found!"<<std::endl;
            return true;
        }

        std::cout<<"\tTESTING CURRENT BFS: Negative reduced costs found - pivoting required!"<<std::endl;
        // Checkpoint >>
        // there's a pivot_row and a pivot_col
        // pivot_row = 0;
        // pivot_col = 4;

        // Do some preprocess >>
        // Now create a tree using the flows available:
        // initialize edges 
        std::vector<Edge> edges;
        for (int i=0; i< matrixDemands+matrixSupplies-1; i++) {
            flowInformation f = flows[i];
            Edge e = {.left = f.row, .right = f.col+matrixSupplies};
            // in the vertex list of bipartite graph ..
            // the right hand side nodes start after an offset 
            edges.push_back(e);
        }
        
        // std::cout<<"edges :"<<edges.size()<<std::endl;

        // Add the incoming arc -
        Edge entering = {.left=pivot_row, .right=matrixSupplies+pivot_col};
        edges.push_back(entering);

        // total number of nodes in the graph (0 to 11)
        int n = matrixSupplies + matrixDemands;
        // build a graph from the given edges
        // since it was a tree and we have one additional edge
        // there exist exactly one cycle in the graph >>
        Graph graph(edges, n);
 
        // to keep track of whether a vertex is discovered or not >> some book-keeping stuff
        std::vector<bool> discovered(n);
        std::vector<int> alternating_path;

        // Perform DFS traversal from the vertex that is pivot-row >> to form a loop
        std::cout<<"\tIMPROVING CURRENT BFS: Finding a loop in the assigment tree!"<<std::endl;
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

        std::cout<<"\tIMPROVING CURRENT BFS: Calculating pivot flow!"<<std::endl;
        int pivot_qty = INT_MAX;
        int pivot_idx = -1;
        int temp_qty, temp_idx; 
        // Find the minimum flow that will be decreased and it's index >>
        for (int i=1; i < alternating_path.size()-1; i++) {
            if (i%2 == 1){
                temp_idx = flow_indexes[{alternating_path[i+1], alternating_path[i]-matrixSupplies}];
                temp_qty = flows[temp_idx].qty;
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
        std::cout<<"\tIMPROVING CURRENT BFS: Updating flows!"<<std::endl;
        int j = -1;
        for (int i=1; i<alternating_path.size()-1; i++) {
            if (i%2 == 1){
                temp_idx = flow_indexes[{alternating_path[i+1], alternating_path[i]-matrixSupplies}];
                // std::cout<<"Pivot - Flow : row "<<i<<" = "<<alternating_path[i+1]<<" , col = "<<alternating_path[i]-matrixSupplies<<std::endl;
            }
            else {
                temp_idx = flow_indexes[{alternating_path[i], alternating_path[i+1]-matrixSupplies}];
                // std::cout<<"Pivot - Flow : row "<<i<<" = "<<alternating_path[i]<<" , col = "<<alternating_path[i+1]-matrixSupplies<<std::endl;
            }
            flows[temp_idx].qty += j*pivot_qty;
            j *= -1; // j will alternate between -1 and +1 
        }

        // Update information on entering arc
        flow_indexes.erase(std::make_pair(flows[pivot_idx].row, flows[pivot_idx].col));
        // std::cout<<"\tResidual Qty :: "<<flows[pivot_idx].qty<<std::endl;
        flows[pivot_idx].qty = pivot_qty;
        flows[pivot_idx].row = pivot_row;
        flows[pivot_idx].col = pivot_col;
        flow_indexes.insert(std::make_pair(std::make_pair(pivot_row, pivot_col),pivot_idx));
        std::cout<<"\tIMPROVING CURRENT BFS: Pivoting complete!"<<std::endl;
        return false;
    }