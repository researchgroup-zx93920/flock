/*
PARALLEL KERNELS are classified as simple and special

(^_^) All simple kernels are here 

Some of the kernels in parallel simplex are specialized and slightly complicated,
they are stored in a separate module for cleanliness, follow the
usage in uv_model_parallel.cu (aka parent) file. 

FYI Directly reviewing this file wouldn't make sense. If you see a kernel in parent
You'll either find it here or there's a comment that would take you to the 
appropriate place
*/

#include <iostream>

#include "parallel_structs.h"

#ifndef KERNELS
#define KERNELS

// ##################################################
// PREPROCESS and POSTPROCESS  >>
// ##################################################

/*
Kernel to convert float cost matrix to the MatrixCell objects
*/
__global__ void createCostMatrix(MatrixCell *d_costMtx, float * d_costs_ptr, const int n_supplies, const int n_demands)
{

    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;

    if (s < n_supplies && d < n_demands)
    {
        int id = s * n_demands + d;
        MatrixCell _c = {.row = s, .col = d, .cost = d_costs_ptr[id]};
        d_costMtx[id] = _c;
    }
}

/*
Generate a tree on the global memory using the initial set of feasible flows
*/
__global__ void create_initial_tree(flowInformation * d_flows_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr,
    int numSupplies, int numDemands)
{
    
    int V = numSupplies+numDemands;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    if (gid < V - 1) {
    
        flowInformation _this_flow = d_flows_ptr[gid];
        int row = _this_flow.source;
        int column =  _this_flow.destination;
        int idx = TREE_LOOKUP(row, numSupplies+column, V); // Index in adjacency matrix
        float _qty = 1.0*_this_flow.qty;
        if (_qty==0){
            // Handling degeneracy - Flow purturbation
            _qty=epsilon;
        }
        d_flowMtx_ptr[gid] = _qty;
        d_adjMtx_ptr[idx] = gid+1;
    }
}

/*
Reverse operation of generating a tree from the feasible flows - unordered allocation
*/
__global__ void retrieve_final_tree(flowInformation * d_flows_ptr, int * d_adjMtx_ptr, float * d_flowMtx_ptr,
        const int numSupplies, const int numDemands) 
{

    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies+numDemands;
    
    // Upper triangle scope of adj matrix
    if (col_indx < V && col_indx >= numSupplies && row_indx < numSupplies) {
        
        // Check if this is a flow edge - 
        int gid = TREE_LOOKUP(row_indx, col_indx, V);
        int flow_id = d_adjMtx_ptr[gid];
        if (flow_id > 0) {

            flowInformation _this_flow;
            _this_flow.qty = round(d_flowMtx_ptr[flow_id - 1]);
            _this_flow.source = row_indx;
            _this_flow.destination = col_indx - numSupplies;
            d_flows_ptr[flow_id - 1] = _this_flow;

        }
    }
}


/* 
Transfer flows on device and prepare an adjacency and flow matrix using the flows from IBFS
In case of sequencial pivoting - one would need a copy of adjMatrix on the host to traverse the graph
*/
__host__ void create_IBF_tree_on_host_device(Graph &graph, flowInformation * feasible_flows, 
    const int numSupplies, const int numDemands) {

    int V = numSupplies+numDemands;
    int _utm_entries = (V*(V+1))/2; // Number of entries in upper triangular matrix 

    gpuErrchk(cudaMalloc((void **) &graph.d_adjMtx_ptr, sizeof(int)*_utm_entries)); 
    thrust::fill(thrust::device, graph.d_adjMtx_ptr, (graph.d_adjMtx_ptr) + _utm_entries, 0);

    gpuErrchk(cudaMalloc((void **) &graph.d_flowMtx_ptr, sizeof(float)*(V-1)));
    thrust::fill(thrust::device, graph.d_flowMtx_ptr, (graph.d_flowMtx_ptr) + (V-1), 0);

    // Make a replica of feasible flows on device
    flowInformation * d_flows_ptr;
    gpuErrchk(cudaMalloc((void **) &d_flows_ptr, sizeof(flowInformation)*(V-1)));
    gpuErrchk(cudaMemcpy(d_flows_ptr, feasible_flows, sizeof(flowInformation)*(V-1), cudaMemcpyHostToDevice));

    // Small kernel to parallely create a tree using the flows
    create_initial_tree <<< ceil(1.0*(V-1)/blockSize), blockSize >>> (d_flows_ptr, graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Now device_flows are useless; 
    // All information about graph is now contained within d_adjMatrix, d_flowMatrix on device =>
    gpuErrchk(cudaFree(d_flows_ptr));
    
    // Make a copy on host >>
    graph.h_adjMtx_ptr = (int *) malloc(sizeof(int)*(_utm_entries));
    gpuErrchk(cudaMemcpy(graph.h_adjMtx_ptr, graph.d_adjMtx_ptr, sizeof(int)*(_utm_entries), cudaMemcpyDeviceToHost));
    graph.h_flowMtx_ptr = (float *) malloc(sizeof(float)*(V-1));
    gpuErrchk(cudaMemcpy(graph.h_flowMtx_ptr, graph.d_flowMtx_ptr, sizeof(float)*(V-1), cudaMemcpyDeviceToHost));

    // Iterations would also work with a adjacency list
    gpuErrchk(cudaMalloc((void **) &graph.d_vertex_start, sizeof(int)*(V)));
    gpuErrchk(cudaMalloc((void **) &graph.d_vertex_degree, sizeof(int)*(V+1)));
    gpuErrchk(cudaMalloc((void **) &graph.d_adjVertices, sizeof(int)*2*(V-1)));
    graph.h_vertex_start = (int *) malloc(sizeof(int)*V);
    graph.h_vertex_degree = (int *) malloc(sizeof(int)*V);
    graph.h_adjVertices = (int *) malloc(sizeof(int)*2*(V-1));

}

/*
Given a feasible tree on device, load a feasible solution to transportation problem on the host
*/
__host__ void retrieve_solution_on_current_tree(flowInformation * feasible_flows, Graph &graph,
    int &active_flows, const int numSupplies, const int numDemands)
{
    
    // Recreate device flows using the current adjMatrix
    flowInformation default_flow;
    default_flow.qty = 0;

    flowInformation * d_flows_ptr;
    gpuErrchk(cudaMalloc((void **) &d_flows_ptr, sizeof(flowInformation)*(numSupplies*numDemands)));
    thrust::fill(thrust::device, d_flows_ptr, d_flows_ptr + (numSupplies*numDemands), default_flow);

    dim3 __blockDim(blockSize, blockSize, 1);
    int grid_size = ceil(1.0*(numSupplies+numDemands)/blockSize); // VxV threads
    dim3 __gridDim(grid_size, grid_size, 1);
    retrieve_final_tree <<< __gridDim, __blockDim >>> (d_flows_ptr, graph.d_adjMtx_ptr, graph.d_flowMtx_ptr, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy the (flows > 0) back on the host >>
    auto flow_end = thrust::remove_if(thrust::device,
        d_flows_ptr, d_flows_ptr + (numSupplies*numDemands), is_zero());
    int flow_count = flow_end - d_flows_ptr;
    // Update active flows in result 
    active_flows = flow_count;
    gpuErrchk(cudaMemcpy(feasible_flows, d_flows_ptr, (flow_count)*sizeof(flowInformation), cudaMemcpyDeviceToHost));

}

/* Clear up the memory occupied by graph on host on device */
__host__ void close_solver(Graph &graph)
{

    gpuErrchk(cudaFree(graph.d_vertex_start));
    gpuErrchk(cudaFree(graph.d_vertex_degree));
    gpuErrchk(cudaFree(graph.d_adjVertices));
    gpuErrchk(cudaFree(graph.d_adjMtx_ptr));
    gpuErrchk(cudaFree(graph.d_flowMtx_ptr));
    
    free(graph.h_vertex_start);
    free(graph.h_vertex_degree);
    free(graph.h_adjVertices);
    free(graph.h_adjMtx_ptr);
    free(graph.h_flowMtx_ptr);

}

__global__ void determine_length(int * length, int * d_adjMtx_ptr, const int V) {

    __shared__ int L;
    int location;
    
    if (threadIdx.x == 0) {
        L = 0;
        length[0] = 0;
    }
    __syncthreads();

    if (blockIdx.x < V) {
            int j = threadIdx.x;
            while (j < V) {
                int idx = TREE_LOOKUP(blockIdx.x, j, V);
                if (d_adjMtx_ptr[idx] > 0) {
                    location = atomicAdd(&L, 1);
                }
                j = j + blockDim.x;
            }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {

        length[blockIdx.x+1] = L;
    }
    
}

__global__ void fill_Ea(int * start, int * Ea, int * d_adjMtx_ptr, const int V, const int numSupplies) {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int offset = start[i];
        int L = 0;
        if (i < V) {
                for (int j=0; j<V; j++) {
                        int idx = TREE_LOOKUP(i, j, V);
                        if (d_adjMtx_ptr[idx] > 0) {
                                Ea[offset + L] = j;
                                L++;
                        }
                }
        }
}


__global__ void fill_Ea2(int * start, int * Ea, int * d_adjMtx_ptr, const int V, const int numSupplies) {
    
    // int i = blockIdx.x*blockDim.x + threadIdx.x;
    int offset = start[blockIdx.x];
    __shared__ int L;
    if (threadIdx.x == 0) {
        L = 0;
    }
    __syncthreads();

    if (blockIdx.x < V) {
            int j = threadIdx.x;
            while (j < V) {
                int idx = TREE_LOOKUP(blockIdx.x, j, V);
                if (d_adjMtx_ptr[idx] > 0) {
                    int location = atomicAdd(&L, 1);
                    Ea[offset + location] = j;
                }
                j = j + blockDim.x;
            }
    }
}

/*
DEBUG UTILITY : VIEW ADJACENCY LIST STRCTURE 
*/
__host__ void __debug_view_adjList(int * start, int * length, int * Ea, const int V) 
{        
        int * h_length = (int *) malloc(sizeof(int)*V);
        int * h_start = (int *) malloc(sizeof(int)*V);
        int * h_Ea = (int *) malloc(sizeof(int)*2*(V-1));

        cudaMemcpy(h_length, length, sizeof(int)*V, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_start, start, sizeof(int)*V, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ea, Ea, sizeof(int)*2*(V-1), cudaMemcpyDeviceToHost);

        std::cout<<"Str = [ ";
        for (int i =0; i < V; i++){
                std::cout<<h_start[i]<<", ";
        }
        std::cout<<"]"<<std::endl;
        std::cout<<"Len = [ ";
        for (int i =0; i < V; i++){
                std::cout<<h_length[i]<<", ";
        }
        std::cout<<"]"<<std::endl;
        std::cout<<"Ea = [ ";
        for (int i =0; i < 2*(V-1); i++){
                std::cout<<h_Ea[i]<<", ";
        }
        std::cout<<"]"<<std::endl;
        
        free(h_length);
        free(h_Ea);
        free(h_start);
        // *************** END OF DEBUG UTILITY ***************

}

__host__ void make_adjacency_list(Graph &graph, const int numSupplies, const int numDemands) {

        // Kernel Dimensions >> 
        dim3 __blockDim(64, 1, 1); 
        dim3 __gridDim(graph.V, 1, 1);

        determine_length <<< __gridDim, __blockDim >>> (graph.d_vertex_degree, graph.d_adjMtx_ptr, graph.V);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        thrust::inclusive_scan(thrust::device, graph.d_vertex_degree, graph.d_vertex_degree + graph.V, graph.d_vertex_start);
        
        // fill_Ea <<< __gridDim, __blockDim >>> (graph.d_vertex_start, graph.d_adjVertices, graph.d_adjMtx_ptr, graph.V, numSupplies);
        fill_Ea2 <<< __gridDim, __blockDim >>> (graph.d_vertex_start, graph.d_adjVertices, graph.d_adjMtx_ptr, graph.V, numSupplies);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Entries in Matrix >>
        // int _utm_entries = (V*(V+1))/2; // Number of entries in upper triangular matrix
        // auto result_end = thrust::copy_if(thrust::device, d_adjMtx_ptr, d_adjMtx_ptr + _utm_entries, 
        //                     Ea, is_nonzero_entry()); // --> need col indices of non-zeros

        // DEBUG :: 
        // __debug_view_adjList(graph.d_vertex_start, &graph.d_vertex_degree[1], graph.d_adjVertices, graph.V);
        // exit(0);
}


// ##################################################
// SOLVING DUAL >>
// ##################################################

/*
APPROACH 1 :
Kernels concerned with solving the UV System using a BFS Traversal Approach
*/

__global__ void copy_row_shadow_prices(Variable * U_vars, float * u_vars_ptr, const int numSupplies) 
{    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < numSupplies) {
        u_vars_ptr[gid] = U_vars[gid].value;
    }
}

__global__ void copy_col_shadow_prices(Variable * V_vars, float * v_vars_ptr, const int numDemands) 
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < numDemands) {
        v_vars_ptr[gid] = V_vars[gid].value;
    }
}

__global__ void initialize_U_vars(Variable * U_vars, const int numSupplies) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    Variable default_var;
    if (gid < numSupplies) {
        U_vars[gid] = default_var;
    }
}

__global__ void initialize_V_vars(Variable * V_vars, const int numDemands) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    Variable default_var;
    if (gid < numDemands) {
        V_vars[gid] = default_var;
    }
}

/*
Breadth First Traversal on UV
*/
__global__ void assign_next(int * d_adjMtx_ptr, float * d_costs_ptr, 
    Variable *u_vars, Variable *v_vars, const int numSupplies, const int numDemands) {
    
    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies + numDemands;

    // Within the scope of the adj matrix
    if (row_indx < numSupplies && col_indx < numDemands) {
        // Check if these are adjacent - (checks in upper triangular matrix, because row < adj-col-index)
        int indx = TREE_LOOKUP(row_indx, col_indx + numSupplies, V); // Adjusted destination vertex ID
        if (d_adjMtx_ptr[indx] > 0) {

            Variable u_i = u_vars[row_indx];
            Variable v_j = v_vars[col_indx];
            
            // Check if any of the u or v has not been assigned and adjacent is assigned - then assign it
            if (u_vars[row_indx].assigned && (!v_vars[col_indx].assigned)) {
                // In this case >> v_j = c_ij - u_i
                Variable var;
                var = d_costs_ptr[row_indx*numDemands+col_indx] - u_vars[row_indx].value;
                // var.assigned = true;
                v_vars[col_indx] = var;
            }
            else if ((!u_vars[row_indx].assigned) && v_vars[col_indx].assigned) {
                // In this case >> u_j = c_ij - v_j
                Variable var;
                var = d_costs_ptr[row_indx*numDemands+col_indx] -  v_vars[col_indx].value;
                // var.assigned = true;
                u_vars[row_indx] = var;
            }
        }
    }
}

// Credits: https://github.com/siddharths2710/cuda_bfs/blob/master/cuda_bfs/kernel.cu
__global__ void CUDA_BFS_KERNEL(int * start, int * length, int *Ea, bool * Fa, bool * Xa, 
        float * variables, float * d_costs_ptr, bool * done, const int numSupplies, const int numDemands, const int V)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id > V)
		*done = false;

	if (Fa[id] == true && Xa[id] == false)
	{
		// printf("%d ", id); //This printf gives the order of vertices in BFS	
		Fa[id] = false;
		Xa[id] = true;
		__syncthreads(); 
		int k = 0;
		int start_ptr = start[id];
		int end_ptr = start_ptr + length[id];
		for (int i = start_ptr; i < end_ptr; i++) 
		{
			int nid = Ea[i];
			if (Xa[nid] == false)
			{       
                int row_indx = min(nid, id);
                int col_indx = max(nid, id) - numSupplies;
				variables[nid] = d_costs_ptr[row_indx*numDemands+col_indx] - variables[id];
				Fa[nid] = true;
				*done = false;
			}
		}
	}
}

/*
APPROACH 2:
Kernels concerned with solving the UV System using a using a matrix solver
*/

// Custom Fill kernel for csr row pointers
__global__ void fill_csr_offset (int * d_csr_offsets, const int length) {
        
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < length) {
                if (idx == 0) {
                        d_csr_offsets[idx] = 0;
                }
                else {
                        d_csr_offsets[idx] = 2*idx - 1; 
                }
        }
}

/*
Create a dense linear system in parallel by looking at current feasible tree 
*/
__global__ void initialize_dense_u_v_system(float * d_A, float * d_b, int * d_adjMtx_ptr, 
    float * d_costs_ptr, const int numSupplies, const int numDemands) {
        
    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies + numDemands;

    if (row_indx < numSupplies && col_indx < numDemands) {
        int indx = TREE_LOOKUP(row_indx, col_indx + numSupplies, V); // Adjusted destination vertex ID
        int flow_indx = d_adjMtx_ptr[indx];
        if (flow_indx > 0) {
            // This is a flow - flow_indx = row_number, u = row_number, v = col_number
            d_A[flow_indx * V + row_indx] = 1;
            d_A[flow_indx * V + numSupplies + col_indx] = 1;
            d_b[flow_indx] = d_costs_ptr[row_indx*numDemands + col_indx];
        }
    }
}

/*
Create a sparse linear system in parallel by looking at current feasible tree 
*/
__global__ void initialize_sparse_u_v_system(int * d_csr_columns, float * d_b, int * d_adjMtx_ptr, 
    float * d_costs_ptr, const int numSupplies, const int numDemands) {
        
    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int V = numSupplies + numDemands;

    if (row_indx < numSupplies && col_indx < numDemands) {
        int indx = TREE_LOOKUP(row_indx, col_indx + numSupplies, V); // Adjusted destination vertex ID
        int flow_indx = d_adjMtx_ptr[indx];
        if (flow_indx > 0) {
            // This is a flow - flow_indx = row_number, u = row_number, v = col_number
            d_csr_columns[2*flow_indx-1] = row_indx;
            d_csr_columns[2*flow_indx] = numSupplies + col_indx;
            d_b[flow_indx] = d_costs_ptr[row_indx*numDemands + col_indx];
        }
    }
}

/*
Load the solution of system to the appropriate place
*/
__global__ void retrieve_uv_solution(float * d_x, float * u_vars_ptr, float * v_vars_ptr, const int numSupplies, const int numDemands) 
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int V = numSupplies + numDemands;
    if (gid < V) {
        if (gid < numSupplies) {
            u_vars_ptr[gid] = d_x[gid];
        } 
        else {
            v_vars_ptr[gid - numSupplies] = d_x[gid];
        }
    }
}

// ##################################################
// COMPUTING REDUCED COSTS >>
// ##################################################


/*
Kernel to compute Reduced Costs in the transportation table
*/
__global__ void computeReducedCosts(float * u_vars_ptr, float * v_vars_ptr, float * d_costs_ptr, float * d_reducedCosts_ptr, 
    const int numSupplies, const int numDemands)
{
        int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

        if (row_indx < numSupplies && col_indx < numDemands) {
            // r =  C_ij - (u_i + v_j);
            float r = d_costs_ptr[row_indx*numDemands+col_indx] - u_vars_ptr[row_indx] - v_vars_ptr[col_indx];
            d_reducedCosts_ptr[row_indx*numDemands+col_indx] = r;
        }
}


/* Optimized function for the above kernel to compute reduced costs */
__global__ void computeReducedCosts(float * u_vars_ptr, float * v_vars_ptr, float * d_costs_ptr, MatrixCell * d_reducedCosts_ptr, 
    const int numSupplies, const int numDemands)
{

        __shared__ float U[reducedcostBlock];
        __shared__ float V[reducedcostBlock];
        
        int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

        if (row_indx < numSupplies && col_indx < numDemands) {
            // r =  C_ij - (u_i + v_j);
            U[threadIdx.y] = u_vars_ptr[row_indx];
            V[threadIdx.x] = v_vars_ptr[col_indx];
        }
        
        __syncthreads();
        
        if (row_indx < numSupplies && col_indx < numDemands) {

            float r = d_costs_ptr[row_indx*numDemands+col_indx] - U[threadIdx.y] - V[threadIdx.x];
            MatrixCell _m = {.row = row_indx, .col = col_indx, .cost = r};
            d_reducedCosts_ptr[row_indx*numDemands+col_indx] = _m;
        
        }
}


/*
 Naive CUDA kernel implementation of Floyd Wharshall
 check if path from vertex x -> y will be shorter using a via. vertex k 
 for all vertices in graph:
    check if (x -> k -> y) < (x -> k)

*/
__global__ void _naive_floyd_warshall_kernel(const int k, const int V, const int numSupplies, const int numDemands, int * d_adjMtx, int * path) {
    
    int col_indx = blockDim.x * blockIdx.x + threadIdx.x;
    int row_indx = blockDim.y * blockIdx.y + threadIdx.y;

    // Compute shortest path from all [supplies to demands] or [demands to supplies]
    // (row_indx < numSupplies && col_indx >= numSupplies && col_indx < V) || (row_indx >= numSupplies && row_indx < V && col_indx < numSupplies)
    if (row_indx < V && col_indx < V) {
        int indexYX = row_indx * V + col_indx;
        int indexKX = k * V + col_indx;
        int indexYK = row_indx*V + k;

        int newPath = d_adjMtx[indexYK] + d_adjMtx[indexKX];
        int oldPath = d_adjMtx[indexYX];
        if (oldPath > newPath) {
            d_adjMtx[indexYX] = newPath;
            path[indexYX] = path[indexKX];
        }
    }
}


/*
Recursively explore pathMtx and store all the discovered cycles in the expanded form 
Can spped this up using a 3D grid? 
*/
__global__ void expand_all_cycles(int * d_pivot_cycles, int * d_adjMtx_transform, 
    int * d_pathMtx, MatrixCell * d_reducedCosts_ptr,  int * d_numNegativeCosts, 
    const int diameter, const int numSupplies, const int numDemands) 
{

    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx < *d_numNegativeCosts) {

        MatrixCell pivot_cell = d_reducedCosts_ptr[idx];
        int col_indx = pivot_cell.col;  // demand point
        int row_indx = pivot_cell.row;  // supply point 
        int offset_1 = diameter*idx; // offset for cycle store
        int V = numSupplies + numDemands;
        // Entering Edge is from supply - i to demand - j we discover a cycle by finding a shortest path from [j] -> [i]
        int offset_path = row_indx*V + (col_indx + numSupplies); // offset for pathMtx
        int offset_distance = row_indx*numDemands + col_indx;

        int depth = d_adjMtx_transform[offset_distance] + 1;
        int current_vtx = col_indx + numSupplies; // backtrack from - j
        int target_vtx = row_indx; // reach - i 
        d_pivot_cycles[offset_1] = target_vtx;
        
        int d = 1;

        while (d < depth) {
            d_pivot_cycles[offset_1 + d] = current_vtx;
            current_vtx = d_pathMtx[target_vtx*V + current_vtx];
            d++;
        }

        d_pivot_cycles[offset_1+depth] = target_vtx;

    }
}

__global__ void derive_cells_on_paths(int * d_pivot_cycles, int * d_adjMtx_transform, 
    int * d_pathMtx, MatrixCell * d_reducedCosts_ptr,  int * d_numNegativeCosts, 
    const int diameter, const int numSupplies, const int numDemands) 
{

    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx < *d_numNegativeCosts) {

        MatrixCell pivot_cell = d_reducedCosts_ptr[idx];
        int col_indx = pivot_cell.col;  // demand point
        int row_indx = pivot_cell.row;  // supply point 
        int offset_1 = diameter*idx; // offset for cycle store
        int V = numSupplies + numDemands;
        // Entering Edge is from supply - i to demand - j we discover a cycle by finding a shortest path from [j] -> [i]
        int offset_path = row_indx*V + (col_indx + numSupplies); // offset for pathMtx
        int offset_distance = row_indx*numDemands + col_indx;

        int depth = d_adjMtx_transform[offset_distance];
        int current_vtx = col_indx + numSupplies; // backtrack from - j
        int target_vtx = row_indx; // reach - i 
        int next_vtx;

        for (int d = 0; d < depth; d++) {
            next_vtx = d_pathMtx[target_vtx*V + current_vtx];
            d_pivot_cycles[offset_1 + d] = (1 - d%2)*(next_vtx*numDemands + (current_vtx-numSupplies)) +
                                        (d%2)*(current_vtx*numDemands + (next_vtx-numSupplies));
            current_vtx = next_vtx;
        }
    }
}


/*
Check if a cycle is still feasible if any of the edges of the cycle located at min_indx has been used
If it is infeasible then set the reduced cost of this cell as non-negative to deactivate pivot here
*/
__global__ void check_pivot_feasibility(int * d_adjMtx_transform, int * d_pivot_cycles, 
    MatrixCell * d_reducedCosts_ptr, const int min_r_index, const int diameter, const int numSupplies, const int numDemands) {

    // Nomenclature : 
    // this_cycle - cycle located on this thread
    // earlier_cycle - cycle located on the min_r_index 
    // Context : Edges in earlier cycle will be used by pivot (because min_reduced cost)
    //      Then if there are any common edges between this_cycle and earlier_cycle
    //      then this_cycle conflicts with earlier_cycle and it has to die
    //      Set reduced cost = non_negative for this cycle

    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;  // demand point
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;  // supply point 
    int offset_1 = diameter*(row_indx*numDemands + col_indx); // offset for this_cycle store
    int V = numSupplies + numDemands;
    int offset_2 = row_indx*V + (col_indx+numSupplies); // offset for path and adjMtx 
    
    // Load the earlier cycle and it's depth - enough memory is available through diameter
    extern __shared__ int earlier_cycle[];

    int _pivot_row =  min_r_index/numDemands;
    int _pivot_col = min_r_index - (_pivot_row*numDemands);
    int earlier_cycle_depth = d_adjMtx_transform[_pivot_row*V + (_pivot_col+numSupplies)] + 1; // depth of earlier_cycle
    // Set reduced cost of earlier_cycle nonNegative >>
    MatrixCell _nonNegative = {.row = _pivot_row, .col = _pivot_col, .cost = epsilon};
    d_reducedCosts_ptr[_pivot_row*numDemands + _pivot_col] = _nonNegative;
    // all threads in block load the same value

    // Load the earlier cycle in parallel 
    int _stride = blockDim.x*blockDim.y; 
    int _local_index = threadIdx.y*blockDim.x + threadIdx.x;
    int offset_3 = diameter*(_pivot_row*numDemands + _pivot_col); // offset for earlier_cycle_store
    while (_local_index < earlier_cycle_depth + 1) {
        earlier_cycle[_local_index] = d_pivot_cycles[offset_3 + _local_index];
        _local_index = _local_index + _stride;
    }
    __syncthreads();

    // Now earlier cycle is available in shared memory - traverse this_cycle and check for common edges
    if (row_indx < numSupplies && col_indx < numDemands) {

        int vtx_i; // i^th vertex in this_cycle 
        int vtx_j; // j^th vertex in earlier_cycle
        int this_cycle_depth = d_adjMtx_transform[offset_2] + 1;
        int vtx_i1, vtx_j1; // i+1^th and j+1^th vertices in corresponding columns
        int edge_i, edge_j; // edge id of edges (i,i+1) and (j,j+1) - edgeid is the index of edge in supply x demand matrix

        for (int i = 1; i < this_cycle_depth + 1; i++) {
            
            vtx_i = d_pivot_cycles[offset_1+i] - numSupplies*(i%2);
            vtx_i1 = d_pivot_cycles[offset_1+i+1] - numSupplies*((i+1)%2);
            edge_i = (vtx_i*numDemands + vtx_i1)*((i+1)%2) + (vtx_i1*numDemands + vtx_i)*(i%2);

            for (int j = 1; j < earlier_cycle_depth + 1; j++) {
                
                vtx_j = earlier_cycle[j] - numSupplies*(j%2);
                vtx_j1 = earlier_cycle[j+1] - numSupplies*((j+1)%2);
                edge_j = (vtx_j*numDemands + vtx_j1)*((j+1)%2) + (vtx_j1*numDemands + vtx_j)*(j%2);
                // Note that we're looking up undirected edges, so we compare unique identifiers of both
                
                // Whenever the cycles intersect
                if (edge_i == edge_j) {
                    // set reduced cost of this_cycles as non-negative
                    MatrixCell _nonNegative = {.row = row_indx, .col = col_indx, .cost = epsilon};
                    d_reducedCosts_ptr[row_indx*numDemands + col_indx] = _nonNegative;
                    return;
                }
            }
        }
    }
}


/*
Check if a cycle is still feasible if any of the edges of the cycle located at min_indx has been used
If it is infeasible then set the reduced cost of this cell as non-negative to deactivate pivot here
*/
__global__ void check_pivot_feasibility(MatrixCell * d_reducedCosts_ptr, const int min_indx,
                const int earlier_from, const int earlier_to, 
                int * d_adjMtx_transform, int * d_pivot_cycles,
                const int diameter, const int numSupplies, const int numDemands, 
                const int stride, const int maxGridDim) {

    // Nomenclature : 
    // this_cycle - cycle located on this blockIdx.x
    // earlier_cycle - cycle located on the min_index 
    // Context : Edges in earlier cycle will be used by pivot (because min_reduced cost)
    //      Then if there are any common edges between this_cycle and earlier_cycle
    //      then this_cycle conflicts with earlier_cycle and it has to die
    //      Set reduced cost = non_negative for this cycle
    
    __shared__ int this_cycle[blockSize];
    __shared__ int earlier_cycle[blockSize];
    __shared__ bool conflict; // = false; 
    __shared__ MatrixCell this_cycleE;
    __shared__ int this_cycle_depth ,earlier_cycle_depth, num_tiles_earlier, y_offset;
    
    // This cycle is available at offset - blockIdx.x*diameter  
    // Earlier cycle is available at offset - min_indx*diameter
    if (threadIdx.x ==0 && threadIdx.y == 0) {
        y_offset = maxGridDim*stride;
        this_cycleE = d_reducedCosts_ptr[y_offset + blockIdx.y];
        this_cycle_depth = d_adjMtx_transform[this_cycleE.row*numDemands + this_cycleE.col];
        earlier_cycle_depth = d_adjMtx_transform[earlier_from*numDemands + earlier_to]; 
        num_tiles_earlier = ceil(1.0*earlier_cycle_depth/blockDim.x);
    }
    
    __syncthreads();
    
    // First loaded this cycle in shared memory - 
    if (this_cycleE.cost < 0) {
        
        // One time load per block
        int idx = blockDim.x*blockIdx.x + threadIdx.x;

        if (threadIdx.y == 0 && idx < this_cycle_depth) {
            int offset = (y_offset + blockIdx.y)*diameter + idx;
            this_cycle[threadIdx.x] = d_pivot_cycles[offset];
        }

        __syncthreads();

        // Tiling for earlier cycle >>
        // This cycle static
        for (int tile_no=0; tile_no < num_tiles_earlier; tile_no++) {

            int load_from = min_indx*diameter + tile_no*blockDim.x;
            int load_upto = min_indx*diameter + earlier_cycle_depth;

            if (threadIdx.y == 0 && load_from + threadIdx.x < load_upto) {
                earlier_cycle[threadIdx.x] = d_pivot_cycles[load_from + threadIdx.x];
            }

            __syncthreads();

            // Comparison within block >> 
            if (idx < this_cycle_depth && load_from + threadIdx.y < load_upto) {

                if (this_cycle[threadIdx.x] == earlier_cycle[threadIdx.y]) {
                    d_reducedCosts_ptr[y_offset + blockIdx.y].cost = epsilon; // atomic not required - race is healthy! 
                }
            
            }
        }
    }
}



/* 
Compute Oppotunity costs and delta -
Logic : For each edge retreive cost and flow - 
        Track their sum and minimum as you traverse along
        Store the final value in appropriate array
*/
__global__ void compute_opportunity_cost_and_delta(int * d_adjMtx_ptr, float * d_flowMtx_ptr, float * d_costs_ptr, 
    int * d_adjMtx_transform, int * d_pivot_cycles, float * d_opportunity_costs, 
    const int diameter, const int numSupplies, const int numDemands) {

    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;  // demand point
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;  // supply point 
    int offset_1 = diameter*(row_indx*numDemands + col_indx); // offset for cycle store
    int V = numSupplies + numDemands;
    int offset_2 = row_indx*V + (col_indx+numSupplies); // offset for adjMtx_transformed

    if (row_indx < numSupplies && col_indx < numDemands) {


        int this_cycle_depth = d_adjMtx_transform[offset_2] + 1;

        int id_graph, id_costs, _from, _to, vtx_i, vtx_i1;
        float _flow, min_flow = INT_MAX, opportunity_cost = 0;
        
        for (int i = 0; i < this_cycle_depth; i++) {

            vtx_i = d_pivot_cycles[offset_1+i];
            vtx_i1 = d_pivot_cycles[offset_1+i+1];
            
            // ########### PART - 1 | Finding the opportunity costs >>
            // Add evens and substract odds
            _from = vtx_i - numSupplies*(i%2);
            _to = vtx_i1 - numSupplies*((i+1)%2);
            id_costs = (_from*numDemands + _to)*((i+1)%2) + (_to*numDemands + _from)*(i%2);
            opportunity_cost = opportunity_cost + ((int) pow(-1, (int)i%2))*d_costs_ptr[id_costs];

            // ########### PART - 2 | Finding the minimum flow >>
            // Traverse the loop find the minimum flow that could be increased
            // on the incoming edge - (Look for minimum of flows on odd indexed edges)
            if (i%2==1) 
            {
                id_graph = d_adjMtx_ptr[TREE_LOOKUP(vtx_i, vtx_i1, V)] - 1;
                _flow = d_flowMtx_ptr[id_graph];
                
                if (_flow < min_flow) 
                {
                    min_flow = _flow;
                }
            }
        }

        // Load the values the in books for next kernel
        d_opportunity_costs[row_indx*numDemands + col_indx] = opportunity_cost*min_flow;
    }
}


/* 
Compute Oppotunity costs and delta -
Logic : For each edge retreive costs - 
        Track their sum as you traverse along
        Store the final value in appropriate array
*/
__global__ void compute_opportunity_cost(int * d_adjMtx_ptr, float * d_flowMtx_ptr, float * d_costs_ptr, 
    int * d_adjMtx_transform, int * d_pivot_cycles, float * d_opportunity_costs, 
    const int diameter, const int numSupplies, const int numDemands) {

    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;  // demand point
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;  // supply point 
    int offset_1 = diameter*(row_indx*numDemands + col_indx); // offset for cycle store
    int V = numSupplies + numDemands;
    int offset_2 = row_indx*V + (col_indx+numSupplies); // offset for adjMtx_transformed

    if (row_indx < numSupplies && col_indx < numDemands) {

        int id_costs, _from = -1, _to = -1;
        int this_cycle_depth = d_adjMtx_transform[offset_2] + 1;
        float opportunity_cost = 0.0f;
        
        for (int i = 0; i < this_cycle_depth; i++) {

            // Finding the opprotunity costs >>
            // Traverse the loop find the minimum flow that could be increased
            // on the incoming edge - (Look for minimum of flows on odd indexed edges)
            _from = d_pivot_cycles[offset_1+i] - numSupplies*(i%2);
            _to = d_pivot_cycles[offset_1+i+1] - numSupplies*((i+1)%2);
            id_costs = (_from*numDemands + _to)*((i+1)%2) + (_to*numDemands + _from)*(i%2);
            opportunity_cost = opportunity_cost + ((int) pow(-1, (int)i%2))*d_costs_ptr[id_costs];
        }

        // Load the values the in books for next kernel
        d_opportunity_costs[row_indx*numDemands + col_indx] = opportunity_cost;  
    }
}

/*
Check if a cycle is still feasible if any of the edges of the cycle located at min_indx has been used
If it is infeasible then set the reduced cost of this cell as non-negative to deactivate pivot here
*/
__global__ void check_pivot_feasibility_dfs(int * depth, int * backtracker, 
    MatrixCell * d_reducedCosts_ptr, const int min_r_index, 
    const int numSupplies, const int numDemands, const int num_threads_launching) {

    // Nomenclature : 
    // this_cycle - cycle located on this thread
    // earlier_cycle - cycle located on the min_r_index 
    // Context : Edges in earlier cycle will be used by pivot (because min_reduced cost)
    //      Then if there are any common edges between this_cycle and earlier_cycle
    //      then this_cycle conflicts with earlier_cycle and it has to die
    //      Set reduced cost = non_negative for this cycle
    int V = numSupplies + numDemands;
    int indx = blockIdx.x*blockDim.x + threadIdx.x; 
    
    // Load the earlier cycle and it's depth - enough memory is available through diameter
    extern __shared__ int earlier_cycle[];

    int earlier_cycle_depth = depth[min_r_index]; // depth of earlier_cycle
    // Set reduced cost of earlier_cycle nonNegative >>
    d_reducedCosts_ptr[min_r_index].cost = epsilon;
    // all threads in block load the same value

    // Load the earlier cycle in parallel 
    int _stride = blockDim.x; 
    int _local_index = threadIdx.x;
    int offset_3 = V*(min_r_index); // offset for earlier_cycle_store
    while (_local_index < earlier_cycle_depth) {
        earlier_cycle[_local_index] = backtracker[offset_3 + _local_index];
        _local_index = _local_index + _stride;
    }
    __syncthreads();

    // Now earlier cycle is available in shared memory - traverse this_cycle and check for common edges
    if (indx < num_threads_launching) {

        int vtx_i; // i^th vertex in this_cycle 
        int vtx_j; // j^th vertex in earlier_cycle
        int this_cycle_depth = depth[indx];
        int vtx_i1, vtx_j1; // i+1^th and j+1^th vertices in corresponding columns
        int edge_i, edge_j; // edge id of edges (i,i+1) and (j,j+1) - edgeid is the index of edge in supply x demand matrix

        for (int i = 1; i < this_cycle_depth; i++) {
            
            vtx_i = backtracker[V*indx+i] - numSupplies*(i%2);
            vtx_i1 = backtracker[V*indx+i+1] - numSupplies*((i+1)%2);
            edge_i = (vtx_i*numDemands + vtx_i1)*((i+1)%2) + (vtx_i1*numDemands + vtx_i)*(i%2);

            for (int j = 1; j < earlier_cycle_depth; j++) {
                
                vtx_j = earlier_cycle[j] - numSupplies*(j%2);
                vtx_j1 = earlier_cycle[j+1] - numSupplies*((j+1)%2);
                edge_j = (vtx_j*numDemands + vtx_j1)*((j+1)%2) + (vtx_j1*numDemands + vtx_j)*(j%2);
                // Note that we're looking up undirected edges, so we compare unique identifiers of both
                
                // Whenever the cycles intersect
                if (edge_i == edge_j) {
                    // set reduced cost of this_cycles as non-negative
                    d_reducedCosts_ptr[indx].cost = epsilon;
                    return;
                }
            }
        }
    }
}


/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Dependent phase 1
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
__global__ void _blocked_fw_dependent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph, int* const pred) {
    __shared__ int cacheGraph[blockSize][blockSize];
    __shared__ int cachePred[blockSize][blockSize];

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = blockSize * blockId + idy;
    const int v2 = blockSize * blockId + idx;

    int newPred;
    int newPath;

    const int cellId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) {
        cacheGraph[idy][idx] = graph[cellId];
        cachePred[idy][idx] = pred[cellId];
        newPred = cachePred[idy][idx];
    } else {
        cacheGraph[idy][idx] = INT16_MAX;
        cachePred[idy][idx] = -1;
    }

    // Synchronize to make sure the all value are loaded in block
    __syncthreads();

    #pragma unroll
    for (int u = 0; u < blockSize; ++u) {
        newPath = cacheGraph[idy][u] + cacheGraph[u][idx];

        // Synchronize before calculate new value
        __syncthreads();
        if (newPath < cacheGraph[idy][idx]) {
            cacheGraph[idy][idx] = newPath;
            newPred = cachePred[u][idx];
        }

        // Synchronize to make sure that all value are current
        __syncthreads();
        cachePred[idy][idx] = newPred;
    }

    if (v1 < nvertex && v2 < nvertex) {
        graph[cellId] = cacheGraph[idy][idx];
        pred[cellId] = cachePred[idy][idx];
    }
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Partial dependent phase 2
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
__global__ void _blocked_fw_partial_dependent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph, int* const pred) {
    if (blockIdx.x == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    int v1 = blockSize * blockId + idy;
    int v2 = blockSize * blockId + idx;

    __shared__ int cacheGraphBase[blockSize][blockSize];
    __shared__ int cachePredBase[blockSize][blockSize];

    // Load base block for graph and predecessors
    int cellId = v1 * pitch + v2;

    if (v1 < nvertex && v2 < nvertex) {
        cacheGraphBase[idy][idx] = graph[cellId];
        cachePredBase[idy][idx] = pred[cellId];
    } else {
        cacheGraphBase[idy][idx] = INT16_MAX;
        cachePredBase[idy][idx] = -1;
    }

    // Load i-aligned singly dependent blocks
    if (blockIdx.y == 0) {
        v2 = blockSize * blockIdx.x + idx;
    } else {
   // Load j-aligned singly dependent blocks
        v1 = blockSize * blockIdx.x + idy;
    }

    __shared__ int cacheGraph[blockSize][blockSize];
    __shared__ int cachePred[blockSize][blockSize];

    // Load current block for graph and predecessors
    int currentPath;
    int currentPred;

    cellId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) {
        currentPath = graph[cellId];
        currentPred = pred[cellId];
    } else {
        currentPath = INT16_MAX;
        currentPred = -1;
    }
    cacheGraph[idy][idx] = currentPath;
    cachePred[idy][idx] = currentPred;

    // Synchronize to make sure the all value are saved in cache
    __syncthreads();

    int newPath;
    // Compute i-aligned singly dependent blocks
    if (blockIdx.y == 0) {
        #pragma unroll
        for (int u = 0; u < blockSize; ++u) {
            newPath = cacheGraphBase[idy][u] + cacheGraph[u][idx];

            if (newPath < currentPath) {
                currentPath = newPath;
                currentPred = cachePred[u][idx];
            }
            // Synchronize to make sure that all threads compare new value with old
            __syncthreads();

           // Update new values
            cacheGraph[idy][idx] = currentPath;
            cachePred[idy][idx] = currentPred;

           // Synchronize to make sure that all threads update cache
            __syncthreads();
        }
    } else {
    // Compute j-aligned singly dependent blocks
        #pragma unroll
        for (int u = 0; u < blockSize; ++u) {
            newPath = cacheGraph[idy][u] + cacheGraphBase[u][idx];

            if (newPath < currentPath) {
                currentPath = newPath;
                currentPred = cachePredBase[u][idx];
            }

            // Synchronize to make sure that all threads compare new value with old
            __syncthreads();

           // Update new values
            cacheGraph[idy][idx] = currentPath;
            cachePred[idy][idx] = currentPred;

           // Synchronize to make sure that all threads update cache
            __syncthreads();
        }
    }

    if (v1 < nvertex && v2 < nvertex) {
        graph[cellId] = currentPath;
        pred[cellId] = currentPred;
    }
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Independent phase 3
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
__global__ void _blocked_fw_independent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph, int* const pred) {
    if (blockIdx.x == blockId || blockIdx.y == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = blockDim.y * blockIdx.y + idy;
    const int v2 = blockDim.x * blockIdx.x + idx;

    __shared__ int cacheGraphBaseRow[blockSize][blockSize];
    __shared__ int cacheGraphBaseCol[blockSize][blockSize];
    __shared__ int cachePredBaseRow[blockSize][blockSize];

    int v1Row = blockSize * blockId + idy;
    int v2Col = blockSize * blockId + idx;

    // Load data for block
    int cellId;
    if (v1Row < nvertex && v2 < nvertex) {
        cellId = v1Row * pitch + v2;

        cacheGraphBaseRow[idy][idx] = graph[cellId];
        cachePredBaseRow[idy][idx] = pred[cellId];
    }
    else {
        cacheGraphBaseRow[idy][idx] = INT16_MAX;
        cachePredBaseRow[idy][idx] = -1;
    }

    if (v1  < nvertex && v2Col < nvertex) {
        cellId = v1 * pitch + v2Col;
        cacheGraphBaseCol[idy][idx] = graph[cellId];
    }
    else {
        cacheGraphBaseCol[idy][idx] = INT16_MAX;
    }

    // Synchronize to make sure the all value are loaded in virtual block
   __syncthreads();

   int currentPath;
   int currentPred;
   int newPath;

   // Compute data for block
   if (v1  < nvertex && v2 < nvertex) {
       cellId = v1 * pitch + v2;
       currentPath = graph[cellId];
       currentPred = pred[cellId];

        #pragma unroll
       for (int u = 0; u < blockSize; ++u) {
           newPath = cacheGraphBaseCol[idy][u] + cacheGraphBaseRow[u][idx];
           if (currentPath > newPath) {
               currentPath = newPath;
               currentPred = cachePredBaseRow[u][idx];
           }
       }
       graph[cellId] = currentPath;
       pred[cellId] = currentPred;
   }
}

/* 
Kernel to determine a feasible triadic closure in feasible tree
 If there exsit edges such that x - z and z - y then, this kernel sets pathMtx[x][y] = z
*/
__global__ void analyse_t_closures(int k, int * d_pathMtx, int * d_adjMtx_transform, const int V) {

    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;
    int tz = blockIdx.z*blockDim.z + threadIdx.z;

    bool in_domain = tx < V && ty < V && tz < V;

    // if in-domain and x and y are distinct
    if (in_domain && !(tx == ty)) {
        
        bool xTOz = d_pathMtx[tx*V + tz] >= 0;
        bool zTOy = d_pathMtx[tz*V + ty] >= 0;
        bool not_xTOy = d_pathMtx[tx*V + ty] == -1;

        // No direct path between x - y and x - z, z - y exist 
        // reach x - y via z and read y - x via z
        // Triadic closure exist
        if (not_xTOy && xTOz && zTOy) {
            // Update path - (x -> y via (z)) >>
            d_pathMtx[tx*V + ty] = tz;
            d_adjMtx_transform[tx*V + ty] = k+2;

            // Update path - (y -> x via (z)) >>
            d_pathMtx[ty*V + tx] = tz;
            d_adjMtx_transform[tx*V + ty] = k+2;
        }
    }
}


/* Set initial values in adj Matrix and path matrix */
__global__ void initialize_parallel_pivot(vertexPin * empty_frontier, 
    int * d_vertex_start, int * d_vertex_degree, int * d_adjVertices,
    float * d_costs_ptr, const int numSupplies, const int numDemands) {
    
    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;

    if (row_indx < numSupplies) {
        int degree = d_vertex_degree[row_indx];
        if (col_indx < degree) {
            // Parallely load the neighbourhood of idx 
            int start = d_vertex_start[row_indx];
            int this_vertex = d_adjVertices[start + col_indx];
            float r_cost = 0; //-1.0f*d_costs_ptr[row_indx*numDemands + col_indx];
            vertexPin _pin = {.from = row_indx, .via = this_vertex, .to = this_vertex, .skip = row_indx, .recirculation = r_cost};
            // Load the pin in the frontier
            empty_frontier[start + col_indx] = _pin;
        }
    }    
}




/* This is BFS Kernel with localized block atomics */
__global__ void update_distance_path_and_create_next_frontier(
                                        int * pathMtx,
                                        int * d_adjMtx_transform,
                                        int * d_vertex_start,
                                        int * d_vertex_length,
                                        int * d_adjVertices,
                                        vertexPin * currLevelNodes,
                                        vertexPin * nextLevelNodes,
                                        int * numCurrLevelNodes,
                                        int * numNextLevelNodes,
                                        const int V,
                                        const int iteration_number) 
{

    // Update the pathMatrix and distance
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    vertexPin this_vertex;
    if (idx < *numCurrLevelNodes) {

        // Path for this vertex is updated and neighbors of this_vertex are queued >>
        this_vertex = currLevelNodes[idx];

        int path_idx = this_vertex.to*V + this_vertex.from;
        pathMtx[path_idx] = this_vertex.via;
        if (iteration_number%2==0) {
            d_adjMtx_transform[path_idx] = iteration_number+1;
        }
    }

    // Now next frontier is prepared >> 

    // Initialize shared memory queue
    __shared__ vertexPin bqNodes[BQ_CAPACITY];
    __shared__ int block_location, gq_location, bq_size;

    if (threadIdx.x == 0) {
        block_location = 0;
        gq_location    = 0;
        bq_size        = 0;
    }
    __syncthreads();

    // For all nodes in the curent level -> get all neighbors of the node
    // Except the skip add rest to the block queue
    // If full, add it to the global queue
    // vertexPin this_vertex = currLevelNodes[idx];
    int current = this_vertex.to;
    int current_skip = this_vertex.skip;
    // Can try threads in y for this
    if (idx < *numCurrLevelNodes) {

        for (int j = d_vertex_start[current]; j < d_vertex_start[current] + d_vertex_length[current]; j++) {
      
            int neighbor = d_adjVertices[j];

            // set the neighbor to 1 and check if it was not already visited -
            if (!(neighbor==current_skip)) {

                int location = atomicAdd(&block_location, 1);
                this_vertex.skip = current;
                this_vertex.to = neighbor;
                
                if (location < BQ_CAPACITY) {
                    bqNodes[location] = this_vertex;
                }
                
                else {
                    location = atomicAdd(numNextLevelNodes, 1);
                    nextLevelNodes[location] = this_vertex;
                }
            }
        }
    }

    __syncthreads();
    // Calculate space for block queue to go into global queue
    if (threadIdx.x == 0) {
        // no race condition on bq_size here
        bq_size     = (BQ_CAPACITY < block_location) ? BQ_CAPACITY : block_location;
        gq_location = atomicAdd(numNextLevelNodes, bq_size);
    }
    __syncthreads();

    // Store block queue in global queue
    for (unsigned int i = threadIdx.x; i < bq_size; i += blockDim.x) {
        nextLevelNodes[gq_location + i] = bqNodes[i];
    }
}



__global__ void update_distance_path_and_create_next_frontier_block_per_vertex(
                                        int * pathMtx,
                                        int * d_adjMtx_transform,
                                        int * d_vertex_start,
                                        int * d_vertex_length,
                                        int * d_adjVertices,
                                        vertexPin * currLevelNodes,
                                        vertexPin * nextLevelNodes,
                                        int * numCurrLevelNodes,
                                        int * numNextLevelNodes,
                                        float * d_costs_ptr,
                                        float * opportunity_cost,
                                        const int numSupplies,
                                        const int numDemands,
                                        const int iteration_number) 
{
 
    // Initialize shared memory queue
    __shared__ int write_offset, read_offset, degree, skip_location;
    __shared__ float update_recirculation;
    vertexPin this_vertex = currLevelNodes[blockIdx.x];
    int V = numSupplies + numDemands;

    if (threadIdx.x == 0) {

        read_offset = d_vertex_start[this_vertex.to];
        degree = d_vertex_length[this_vertex.to];
        write_offset = atomicAdd(numNextLevelNodes, degree-1);
        skip_location = V;

        // Update the pathMatrix and distance
        // Path for this vertex is updated and neighbors of this_vertex are queued >>
        int path_idx = this_vertex.from*V + this_vertex.to;
        int i = (int) iteration_number%2;
        pathMtx[path_idx] = this_vertex.skip;

        // Update reduced cost accumulator >>
        unsigned int cost_indx = (1 - i)*(this_vertex.skip*numDemands + (this_vertex.to - numSupplies)) 
            + (i)*(this_vertex.to*numDemands + (this_vertex.skip - numSupplies));
        update_recirculation = this_vertex.recirculation + ((int) pow(-1, i+1))*d_costs_ptr[cost_indx];

        if (i==0) { // No divergence here - all threads in iteration execute the same
            
            int dist_indx = this_vertex.from*numDemands + (this_vertex.to - numSupplies);
            d_adjMtx_transform[dist_indx] = iteration_number+1;
            opportunity_cost[dist_indx] = opportunity_cost[dist_indx] + update_recirculation;
        
        }
    }

    __syncthreads();

    // For all nodes in the curent level -> get all neighbors of the node
    // Except the skip add rest to the block queue
    // If full, add it to the global queue
    this_vertex.recirculation =  update_recirculation;
    int current = this_vertex.to;
    int current_skip = this_vertex.skip;
    
    for (int j = threadIdx.x; j < degree; j += blockDim.x) {

        int neighbor = d_adjVertices[read_offset + j];

        // check the position of skip -
        if (neighbor==current_skip) {
            skip_location = j;
        }

        __syncthreads();

        // check the position of skip -
        if (j < skip_location) {

            this_vertex.skip = current;
            this_vertex.to = neighbor;
            nextLevelNodes[write_offset+j] = this_vertex;
        
        }

        if (j > skip_location) {

            this_vertex.skip = current;
            this_vertex.to = neighbor;
            nextLevelNodes[write_offset+j-1] = this_vertex;

        }

    }

}

__global__ void collectNegativeReducedCosts(MatrixCell * d_reducedCosts_ptr, int * numNegativeCosts,
    float * opportunity_costs, const int numSupplies, const int numDemands) 
{
    
    int row_indx = threadIdx.y + blockIdx.y*blockDim.y;
    int col_indx = threadIdx.x + blockIdx.x*blockDim.x;

    if (row_indx < numSupplies && col_indx < numDemands) {

        int gid = row_indx*numDemands + col_indx;
        float r_cost = opportunity_costs[gid];
        if (r_cost < 0 && abs(r_cost) > epsilon2) {
            int position = atomicAdd(numNegativeCosts, 1);
            MatrixCell cost_obj = {.row = row_indx, .col = col_indx, .cost = r_cost};
             d_reducedCosts_ptr[position] = cost_obj;
        }
    }
}

#endif