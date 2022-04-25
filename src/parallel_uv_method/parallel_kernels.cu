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
__global__ void createCostMatrix(MatrixCell *d_costMtx, float * d_costs_ptr, int n_supplies, int n_demands)
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
        int numSupplies, int numDemands) 
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
__host__ void create_IBF_tree_on_host_device(flowInformation * feasible_flows,
    int ** d_adjMtx_ptr, int ** h_adjMtx_ptr, float ** d_flowMtx_ptr, float ** h_flowMtx_ptr, 
    int numSupplies, int numDemands) 
{
    int V = numSupplies+numDemands;
    int _utm_entries = (V*(V+1))/2; // Number of entries in upper triangular matrix 

    gpuErrchk(cudaMalloc((void **) d_adjMtx_ptr, sizeof(int)*_utm_entries)); 
    thrust::fill(thrust::device, *d_adjMtx_ptr, (*d_adjMtx_ptr) + _utm_entries, 0);

    gpuErrchk(cudaMalloc((void **) d_flowMtx_ptr, sizeof(float)*(V-1)));
    thrust::fill(thrust::device, *d_flowMtx_ptr, (*d_flowMtx_ptr) + (V-1), 0);

    // Make a replica of feasible flows on device
    flowInformation * d_flows_ptr;
    gpuErrchk(cudaMalloc((void **) &d_flows_ptr, sizeof(flowInformation)*(V-1)));
    gpuErrchk(cudaMemcpy(d_flows_ptr, feasible_flows, sizeof(flowInformation)*(V-1), cudaMemcpyHostToDevice));

    // Small kernel to parallely create a tree using the flows
    create_initial_tree <<< ceil(1.0*(V-1)/blockSize), blockSize >>> (d_flows_ptr, *d_adjMtx_ptr, *d_flowMtx_ptr, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Now device_flows are useless; 
    // All information about graph is now contained within d_adjMatrix, d_flowMatrix on device =>
    gpuErrchk(cudaFree(d_flows_ptr));
    
    // Make a copy on host >>
    *h_adjMtx_ptr = (int *) malloc(sizeof(int)*(_utm_entries));
    gpuErrchk(cudaMemcpy(*h_adjMtx_ptr, *d_adjMtx_ptr, sizeof(int)*(_utm_entries), cudaMemcpyDeviceToHost));
    *h_flowMtx_ptr = (float *) malloc(sizeof(float)*(V-1));
    gpuErrchk(cudaMemcpy(*h_flowMtx_ptr, *d_flowMtx_ptr, sizeof(float)*(V-1), cudaMemcpyDeviceToHost));
}

/*
Given a feasible tree on device, load a feasible solution to transportation problem on the host
*/
__host__ void retrieve_solution_on_current_tree(flowInformation * feasible_flows, int * d_adjMtx_ptr, float * d_flowMtx_ptr, 
    int &active_flows, int numSupplies, int numDemands)
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
    retrieve_final_tree <<< __gridDim, __blockDim >>> (d_flows_ptr, d_adjMtx_ptr, d_flowMtx_ptr, numSupplies, numDemands);
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


// ##################################################
// SOLVING DUAL >>
// ##################################################

/*
APPROACH 1 :
Kernels concerned with solving the UV System using a BFS Traversal Approach
*/

__global__ void copy_row_shadow_prices(Variable * U_vars, float * u_vars_ptr, int numSupplies) 
{    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < numSupplies) {
        u_vars_ptr[gid] = U_vars[gid].value;
    }
}

__global__ void copy_col_shadow_prices(Variable * V_vars, float * v_vars_ptr, int numDemands) 
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < numDemands) {
        v_vars_ptr[gid] = V_vars[gid].value;
    }
}

__global__ void initialize_U_vars(Variable * U_vars, int numSupplies) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    Variable default_var;
    if (gid < numSupplies) {
        U_vars[gid] = default_var;
    }
}

__global__ void initialize_V_vars(Variable * V_vars, int numDemands) {
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
    Variable *u_vars, Variable *v_vars, int numSupplies, int numDemands) {
    
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
        float * variables, float * d_costs_ptr, bool * done, int numSupplies, int numDemands, int V)
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

__global__ void determine_length(int * length, int * d_adjMtx_ptr, int V) {
        int L = 0;
        int i = blockIdx.x *blockDim.x + threadIdx.x;
        // No data re-use (this is a straight fwd kernel)
        if (i < V) 
        {    
                for (int j=0; j<V; j++) {
                        int idx = TREE_LOOKUP(i, j, V);
                        if (d_adjMtx_ptr[idx] > 0) {
                                L++;
                        }
                }
                length[i+1] = L;
                length[0] = 0;
        }
}

__global__ void fill_Ea(int * start, int * Ea, int * d_adjMtx_ptr, int V, int numSupplies) {
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


/*
APPROACH 2:
Kernels concerned with solving the UV System using a using a matrix solver
*/

// Custom Fill kernel for csr row pointers
__global__ void fill_csr_offset (int * d_csr_offsets, int length) {
        
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
    float * d_costs_ptr, int numSupplies, int numDemands) {
        
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
    float * d_costs_ptr, int numSupplies, int numDemands) {
        
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
__global__ void retrieve_uv_solution(float * d_x, float * u_vars_ptr, float * v_vars_ptr, int numSupplies, int numDemands) 
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
    int numSupplies, int numDemands)
{

        int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

        if (row_indx < numSupplies && col_indx < numDemands) {
            // r =  C_ij - (u_i + v_j);
            float r = d_costs_ptr[row_indx*numDemands+col_indx] - u_vars_ptr[row_indx] - v_vars_ptr[col_indx];
            d_reducedCosts_ptr[row_indx*numDemands+col_indx] = r;
        }
}

#endif