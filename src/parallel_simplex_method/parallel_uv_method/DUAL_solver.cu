#include "DUAL_solver.h"


__host__ void dualMalloc(DualHandler &dual, int numSupplies, int numDemands) {
    
    int V = numSupplies + numDemands;
    // Create and Initialize u and v variables 
    // To be allocated regardless 
    gpuErrchk(cudaMalloc((void **) &dual.u_vars_ptr, sizeof(float)*numSupplies));
    gpuErrchk(cudaMalloc((void **) &dual.v_vars_ptr, sizeof(float)*numDemands));

    if (CALCULATE_DUAL=="device_bfs") {

        //  empty u and v equations using the Variable Data Type >>
        gpuErrchk(cudaMalloc((void **) &dual.Fa, sizeof(bool)*V));
        gpuErrchk(cudaMalloc((void **) &dual.Xa, sizeof(bool)*V));
        gpuErrchk(cudaMalloc((void **) &dual.variables, sizeof(float)*V));

    }

    else if (CALCULATE_DUAL=="host_bfs") {

        //  empty u and v equations using the Variable Data Type >>
        dual.h_visited = (bool *) malloc(sizeof(bool)*V);
        dual.h_variables = (float *) malloc(sizeof(float)*V);
    }

    else if (CALCULATE_DUAL=="device_sparse_linear_solver") {

        int U_0 = 0;
        float U_0_value = 0.0;

        // Allocate memory to store the sparse linear system
        dual.nnz = 2*V - 1;

        // Values are coefs of u and v, which are always one only position and b-vector changes with iterations, So
        gpuErrchk(cudaMalloc((void **) &dual.d_csr_values,  dual.nnz * sizeof(float)));
        thrust::fill(thrust::device, dual.d_csr_values, (dual.d_csr_values) + dual.nnz, 1.0);

        // U_0 is always set to zero - meaning first element is always 0,0 in csr
        gpuErrchk(cudaMalloc((void **) &dual.d_csr_columns, dual.nnz * sizeof(int)));
        gpuErrchk(cudaMemcpy(dual.d_csr_columns, &U_0, sizeof(int), cudaMemcpyHostToDevice));

        // The row pointers also remain constant {0,1,3,5, ... , 2V-1}, Custom Filler kernel below
        gpuErrchk(cudaMalloc((void**) &dual.d_csr_offsets, (V + 1) * sizeof(int)));
        fill_csr_offset <<< ceil(1.0*(V+1)/blockSize), blockSize >>> (dual.d_csr_offsets, V+1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMalloc((void **) &dual.d_b, sizeof(float)*V));
        gpuErrchk(cudaMemcpy(dual.d_b, &U_0_value, sizeof(float), cudaMemcpyHostToDevice));
        
        // d_x is only allocated here - it is to be populated by API's
        gpuErrchk(cudaMalloc((void **) &dual.d_x, V * sizeof(float)));
    }

    else if (CALCULATE_DUAL=="device_dense_linear_solver") {

        // Allocate memory to store the dense linear system
        gpuErrchk(cudaMalloc((void **) &dual.d_A, sizeof(float)*V*V));
        gpuErrchk(cudaMalloc((void **) &dual.d_b, sizeof(float)*V));
        gpuErrchk(cudaMalloc((void **) &dual.d_x, V * sizeof(float)));
    }
}

__host__ void dualFree(DualHandler &dual) {
     
        gpuErrchk(cudaFree(dual.u_vars_ptr));
        gpuErrchk(cudaFree(dual.v_vars_ptr));
        
        if (CALCULATE_DUAL=="device_bfs") {

                gpuErrchk(cudaFree(dual.Fa));
                gpuErrchk(cudaFree(dual.Xa));
                gpuErrchk(cudaFree(dual.variables));
        }

        else if (CALCULATE_DUAL=="host_bfs") {
                
                free(dual.h_visited);
                free(dual.h_variables);
        }
        
        else if (CALCULATE_DUAL=="device_sparse_linear_solver") {

        gpuErrchk(cudaFree(dual.d_csr_values));
        gpuErrchk(cudaFree(dual.d_csr_columns));
        gpuErrchk(cudaFree(dual.d_csr_offsets));
        gpuErrchk(cudaFree(dual.d_b));
        gpuErrchk(cudaFree(dual.d_x));
        
        }
        
        else if (CALCULATE_DUAL=="device_dense_linear_solver") {

        gpuErrchk(cudaFree(dual.d_A));
        gpuErrchk(cudaFree(dual.d_b));
        gpuErrchk(cudaFree(dual.d_x));

        }
}


__host__ void find_dual_using_sparse_solver(DualHandler &dual, Graph &graph, float * d_costs_ptr, int numSupplies, int numDemands)
{

        // Nice thing is that csr values and offsets remain static over the iterations
        // So a bunch of things are assigned here statically during the dualMalloc phase
        dim3 __blockDim(blockSize, blockSize, 1); 
        dim3 __gridDim(ceil(1.0*numDemands/blockSize), ceil(1.0*numSupplies/blockSize), 1);
        initialize_sparse_u_v_system <<< __gridDim, __blockDim >>> (dual.d_csr_columns, dual.d_b, graph.d_adjMtx_ptr, d_costs_ptr, 
                numSupplies, numDemands);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        /* *********************
        DEBUG UTILITY :: Print the csr matrix for u-v system
         ************************/
        // float * h_csr_values = (float *) malloc(sizeof(float)*nnz);
        // int * h_csr_columns = (int *) malloc(sizeof(int)*nnz);
        // int * h_csr_offsets = (int *) malloc(sizeof(int)*(V+1));
        // gpuErrchk(cudaMemcpy(h_csr_values, d_csr_values, sizeof(float)*nnz, cudaMemcpyDeviceToHost));
        // gpuErrchk(cudaMemcpy(h_csr_columns, d_csr_columns, sizeof(int)*nnz, cudaMemcpyDeviceToHost));
        // gpuErrchk(cudaMemcpy(h_csr_offsets, d_csr_offsets, sizeof(int)*(V+1), cudaMemcpyDeviceToHost));
        // std::cout<<"CSR Values = [";
        // for (int i =0; i< nnz; i++){
        //         std::cout<<h_csr_values[i]<<", ";
        // }
        // std::cout<<"]"<<std::endl;
        // std::cout<<"CSR Columns = [";
        // for (int i =0; i< nnz; i++){
        //         std::cout<<h_csr_columns[i]<<", ";
        // }
        // std::cout<<"]"<<std::endl;
        // std::cout<<"CSR Offsets = [";
        // for (int i =0; i < V+1; i++){
        //         std::cout<<h_csr_offsets[i]<<", ";
        // }
        // std::cout<<"]"<<std::endl;
        // exit(0);
        /* ********** END OF UTILITY ************* */

        // Core >>		
        cusolverSpHandle_t solver_handle;
	CUSOLVER_CHECK(cusolverSpCreate(&solver_handle));

        cusparseMatDescr_t descrA;
        CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
	CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)); 

        int singularity;
        // Cholesky : 1 : symrcm, 2 : symamd, or 3 : csrmetisnd
        int reorder = 0;

        if (SPARSE_SOLVER=="qr") {
                
                CUSOLVER_CHECK(cusolverSpScsrlsvqr(solver_handle, graph.V, dual.nnz, descrA, 
                                        dual.d_csr_values, dual.d_csr_offsets, dual.d_csr_columns, dual.d_b, 
                                        10e-9, reorder, dual.d_x, &singularity));
        
        }

	else if (SPARSE_SOLVER=="chol") {

                CUSOLVER_CHECK(cusolverSpScsrlsvchol(solver_handle, graph.V, dual.nnz, descrA,
                     dual.d_csr_values, dual.d_csr_offsets, dual.d_csr_columns, dual.d_b,
                     10e-9, reorder, dual.d_x, &singularity));
        }

        else {
        
                std::cout<<" Invalid sparse solver!"<<std::endl;
                exit(0);
        }
        
        // Clean up ! 
        CUSOLVER_CHECK(cusolverSpDestroy(solver_handle));
        CUSPARSE_CHECK(cusparseDestroyMatDescr(descrA));

        if (singularity == -1) {

                dim3 __blockDim2(blockSize, 1, 1);
                dim3 __gridDim2(ceil(1.0*graph.V/blockSize), 1, 1);
                retrieve_uv_solution <<< __gridDim2, __blockDim2 >>> (dual.d_x, dual.u_vars_ptr, dual.v_vars_ptr, numSupplies, numDemands);
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
        }
        
        else {
        
                std::cout<<" ========== !! Unexpected ERROR :: Matrix A is singular !!"<<std::endl;
                std::cout<<" ========== Return singularity = "<<singularity<<std::endl;
                exit(0);
                // float * h_x = (float *) malloc(sizeof(float)*V);
                // cudaMemcpy(h_x, d_x, sizeof(float)*V, cudaMemcpyDeviceToHost);
                // for (int i=0; i<V; i++) {
                //     std::cout<< "X [" <<i<<"] = "<<h_x[i]<<std::endl;
                // }
        }
        // Make tree 
}


__host__ void find_dual_using_dense_solver(DualHandler &dual, Graph &graph, float * d_costs_ptr, 
        int numSupplies, int numDemands)
{

        thrust::fill(thrust::device, dual.d_A, dual.d_A + (graph.V * graph.V), 0.0f);
        thrust::fill(thrust::device, dual.d_b, dual.d_b + (graph.V), 0.0f);

        // Nice thing is that csr values and offsets remain static over the iterations
        dim3 __blockDim(blockSize, blockSize, 1); 
        dim3 __gridDim(ceil(1.0*numDemands/blockSize), ceil(1.0*numSupplies/blockSize), 1);
        initialize_dense_u_v_system <<< __gridDim, __blockDim >>> (dual.d_A, dual.d_b, 
                graph.d_adjMtx_ptr, d_costs_ptr, 
                numSupplies, numDemands);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
}


__host__ void find_dual_using_device_bfs(DualHandler &dual, Graph &graph, float * d_costs_ptr, int numSupplies, int numDemands) {

        bool f0 = true;
        // Initialize BFS >>
	thrust::fill(thrust::device, dual.Fa, dual.Fa + graph.V, false);
        thrust::fill(thrust::device, dual.Xa, dual.Xa + graph.V, false);
        thrust::fill(thrust::device, dual.variables, dual.variables + graph.V, 0.0);
        gpuErrchk(cudaMemcpy(&dual.Fa[0], &f0, sizeof(bool), cudaMemcpyHostToDevice));

        // >>> Running BFS
        // std::cout<<"Running BFS"<<std::endl;
        bool done;
	bool * d_done;
	gpuErrchk(cudaMalloc((void**) &d_done, sizeof(bool)));
	int count = 0;
        dim3 __blockDim(blockSize, 1, 1); 
        dim3 __gridDim(ceil(1.0*graph.V/blockSize), 1, 1);

	do {
		count++;
		done = true;
		gpuErrchk(cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice));
		CUDA_BFS_KERNEL <<<__gridDim, __blockDim >>>(graph.d_vertex_start, &graph.d_vertex_degree[1], graph.d_adjVertices, 
                                        dual.Fa, dual.Xa, dual.variables, d_costs_ptr, 
                                        d_done, numSupplies, numDemands, graph.V);
		gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk(cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost));

	} while (!done && count < (numSupplies+numDemands-1));

        // std::cout<<"BFS Complete!"<<std::endl;
	gpuErrchk(cudaMemcpy(dual.u_vars_ptr, &dual.variables[0], sizeof(float)*numSupplies, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(dual.v_vars_ptr, &dual.variables[numSupplies], sizeof(float)*numDemands, cudaMemcpyDeviceToDevice));
}

__host__ void find_dual_using_host_bfs(DualHandler &dual,  Graph &graph, float * h_costs_ptr, 
        int numSupplies, int numDemands) {

        // Copy Adjacency list on host >> assuming Tranformation already occured at the start of pivoting 
        gpuErrchk(cudaMemcpy(graph.h_vertex_degree, &graph.d_vertex_degree[1], sizeof(int)*graph.V, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(graph.h_vertex_start, graph.d_vertex_start, sizeof(int)*graph.V, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(graph.h_adjVertices, graph.d_adjVertices, sizeof(int)*2*(graph.V-1), cudaMemcpyDeviceToHost));

        thrust::fill(thrust::host, dual.h_visited, dual.h_visited + graph.V, false);
        thrust::fill(thrust::host, dual.h_variables, dual.h_variables + graph.V, 0.0f);

        // Initialize >>
        std::queue<int> assigned_parents;
        dual.h_visited[0] = true;
        // The value of u0 is already zero as initialized
        assigned_parents.push(0);

        // Perform a BFS on Host (trickle down) >> 
        int parent, child, row, col;
        
        int * h_start = graph.h_vertex_start;
        int * h_length = graph.h_vertex_degree;
        int * h_Ea = graph.h_adjVertices;


        while (!assigned_parents.empty()) {
                parent = assigned_parents.front();
                for (int i = h_start[parent]; i < h_start[parent] + h_length[parent]; i++) {
                        child = h_Ea[i];
                        if (!dual.h_visited[child]) {
                                dual.h_visited[child] = true;
                                row = min(parent, child);
                                col = max(parent, child) - numSupplies;
                                dual.h_variables[child] = h_costs_ptr[row*numDemands + col] - dual.h_variables[parent];
                                assigned_parents.push(child);
                        }
                }
                assigned_parents.pop();
        }

        // Transfer back to GPU >> 
        gpuErrchk(cudaMemcpy(dual.u_vars_ptr, &dual.h_variables[0], sizeof(int)*numSupplies, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dual.v_vars_ptr, &dual.h_variables[numSupplies], sizeof(int)*numDemands, cudaMemcpyHostToDevice));

}