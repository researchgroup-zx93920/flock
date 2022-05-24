#include "DUAL_solver.h"


__host__ void initialize_device_DUAL(float ** u_vars_ptr, float ** v_vars_ptr, 
        bool ** Fa, bool ** Xa, float ** variables,
        float ** d_csr_values, int ** d_csr_columns, int ** d_csr_offsets,
        float ** d_A, float ** d_b, float ** d_x, int64_t &nnz, 
        bool ** h_visited, float ** h_variables,
        int numSupplies, int numDemands) {
    
    int V = numSupplies + numDemands;
    // Create and Initialize u and v variables 
    // To be allocated regardless 
    gpuErrchk(cudaMalloc((void **) u_vars_ptr, sizeof(float)*numSupplies));
    gpuErrchk(cudaMalloc((void **) v_vars_ptr, sizeof(float)*numDemands));

    if (CALCULATE_DUAL=="device_bfs") {

        //  empty u and v equations using the Variable Data Type >>
        gpuErrchk(cudaMalloc((void**) Fa, sizeof(bool)*V));
        gpuErrchk(cudaMalloc((void**) Xa, sizeof(bool)*V));
        gpuErrchk(cudaMalloc((void**) variables, sizeof(float)*V));

    }

    else if (CALCULATE_DUAL=="host_bfs") {

        //  empty u and v equations using the Variable Data Type >>
        * h_visited = (bool *) malloc(sizeof(bool)*V);
        * h_variables = (float *) malloc(sizeof(float)*V);
    }

    else if (CALCULATE_DUAL=="device_sparse_linear_solver") {

        int U_0 = 0;
        float U_0_value = 0.0;

        // Allocate memory to store the sparse linear system
        nnz = 2*V - 1;

        // Values are coefs of u and v, which are always one only position and b-vector changes with iterations, So
        gpuErrchk(cudaMalloc((void**) d_csr_values,  nnz * sizeof(float)));
        thrust::fill(thrust::device, *d_csr_values, (*d_csr_values) + nnz, 1.0);

        // U_0 is always set to zero - meaning first element is always 0,0 in csr
        gpuErrchk(cudaMalloc((void**) d_csr_columns, nnz * sizeof(int)));
        gpuErrchk(cudaMemcpy(*d_csr_columns, &U_0, sizeof(int), cudaMemcpyHostToDevice));

        // The row pointers also remain constant {0,1,3,5, ... , 2V-1}, Custom Filler kernel below
        gpuErrchk(cudaMalloc((void**) d_csr_offsets, (V + 1) * sizeof(int)));
        fill_csr_offset <<< ceil(1.0*(V+1)/blockSize), blockSize >>> (*d_csr_offsets, V+1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMalloc((void **) d_b, sizeof(float)*V));
        gpuErrchk(cudaMemcpy(*d_b, &U_0_value, sizeof(float), cudaMemcpyHostToDevice));
        
        // d_x is only allocated here - it is to be populated by API's
        gpuErrchk(cudaMalloc((void **) d_x, V * sizeof(float)));
    }

    else if (CALCULATE_DUAL=="device_dense_linear_solver") {

        // Allocate memory to store the dense linear system
        gpuErrchk(cudaMalloc((void **) d_A, sizeof(float)*V*V));
        gpuErrchk(cudaMalloc((void **) d_b, sizeof(float)*V));
        gpuErrchk(cudaMalloc((void **) d_x, V * sizeof(float)));

    }
}

__host__ void terminate_device_DUAL(float * u_vars_ptr, float * v_vars_ptr, 
        bool * Fa, bool * Xa, float * variables,
        float * d_csr_values, int * d_csr_columns, int * d_csr_offsets,
        float * d_A, float * d_b, float * d_x, 
        bool * h_visited, float * h_variables) {
     
        gpuErrchk(cudaFree(u_vars_ptr));
        gpuErrchk(cudaFree(v_vars_ptr));
        
        if (CALCULATE_DUAL=="device_bfs") {

                gpuErrchk(cudaFree(Fa));
                gpuErrchk(cudaFree(Xa));
                gpuErrchk(cudaFree(variables));
        }

        else if (CALCULATE_DUAL=="host_bfs") {
                
                free(h_visited);
                free(h_variables);
        }
        
        else if (CALCULATE_DUAL=="device_sparse_linear_solver") {

        gpuErrchk(cudaFree(d_csr_values));
        gpuErrchk(cudaFree(d_csr_columns));
        gpuErrchk(cudaFree(d_csr_offsets));
        gpuErrchk(cudaFree(d_b));
        gpuErrchk(cudaFree(d_x));
        
        }
        
        else if (CALCULATE_DUAL=="device_dense_linear_solver") {

        gpuErrchk(cudaFree(d_A));
        gpuErrchk(cudaFree(d_b));
        gpuErrchk(cudaFree(d_x));

        }
}


__host__ void find_dual_using_sparse_solver(float * u_vars_ptr, float * v_vars_ptr, 
        float * d_costs_ptr, int * d_adjMtx_ptr,
        float * d_csr_values, int * d_csr_columns, int * d_csr_offsets, float * d_x, float * d_b, 
        int64_t nnz, int numSupplies, int numDemands)
{
        int V = numSupplies + numDemands;

        // Nice thing is that csr values and offsets remain static over the iterations
        dim3 __blockDim(blockSize, blockSize, 1); 
        dim3 __gridDim(ceil(1.0*numDemands/blockSize), ceil(1.0*numSupplies/blockSize), 1);
        initialize_sparse_u_v_system <<< __gridDim, __blockDim >>> (d_csr_columns, d_b, d_adjMtx_ptr, d_costs_ptr, 
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
                
                CUSOLVER_CHECK(cusolverSpScsrlsvqr(solver_handle, V, nnz, descrA, 
                                        d_csr_values, d_csr_offsets, d_csr_columns, d_b, 
                                        10e-9, reorder, d_x, &singularity));
        
        }

	else if (SPARSE_SOLVER=="chol") {

                CUSOLVER_CHECK(cusolverSpScsrlsvchol(solver_handle, V, nnz, descrA,
                     d_csr_values, d_csr_offsets, d_csr_columns, d_b,
                     10e-9, reorder, d_x, &singularity));

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
                dim3 __gridDim2(ceil(1.0*V/blockSize), 1, 1);
                retrieve_uv_solution <<< __gridDim2, __blockDim2 >>> (d_x, u_vars_ptr, v_vars_ptr, numSupplies, numDemands);
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


__host__ void find_dual_using_dense_solver(float * u_vars_ptr, float * v_vars_ptr, 
        float * d_costs_ptr, int * d_adjMtx_ptr,
        float * d_A, float * d_x, float * d_b, 
        int numSupplies, int numDemands) 
{
        int V = numSupplies + numDemands;
        thrust::fill(thrust::device, d_A, d_A + (V * V), 0.0f);
        thrust::fill(thrust::device, d_b, d_b + (V), 0.0f);

        // Nice thing is that csr values and offsets remain static over the iterations
        dim3 __blockDim(blockSize, blockSize, 1); 
        dim3 __gridDim(ceil(1.0*numDemands/blockSize), ceil(1.0*numSupplies/blockSize), 1);
        initialize_dense_u_v_system <<< __gridDim, __blockDim >>> (d_A, d_b, d_adjMtx_ptr, d_costs_ptr, 
                numSupplies, numDemands);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

}

__host__ void find_dual_using_bfs(float * u_vars_ptr, float * v_vars_ptr, 
        int * length, int * start, int * Ea, bool * Fa, bool * Xa, float * variables,
        int * d_adjMtx_ptr, float * d_costs_ptr, int numSupplies, int numDemands) {

        int V = numSupplies + numDemands;

        bool f0 = true;
        // Initialize BFS >>
	thrust::fill(thrust::device, Fa, Fa + V, false);
        thrust::fill(thrust::device, Xa, Xa + V, false);
        thrust::fill(thrust::device, variables, variables + V, 0.0);
        gpuErrchk(cudaMemcpy(&Fa[0], &f0, sizeof(bool), cudaMemcpyHostToDevice));

        // >>> Running BFS
        // std::cout<<"Running BFS"<<std::endl;
        bool done;
	bool * d_done;
	gpuErrchk(cudaMalloc((void**) &d_done, sizeof(bool)));
	int count = 0;
        dim3 __blockDim(blockSize, 1, 1); 
        dim3 __gridDim(ceil(1.0*V/blockSize), 1, 1);

	do {
		count++;
		done = true;
		gpuErrchk(cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice));
		CUDA_BFS_KERNEL <<<__gridDim, __blockDim >>>(start, length, Ea, Fa, Xa, variables, d_costs_ptr, d_done, numSupplies, numDemands, V);
		gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk(cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost));

	} while (!done && count < (numSupplies+numDemands-1));

        // std::cout<<"BFS Complete!"<<std::endl;
	gpuErrchk(cudaMemcpy(u_vars_ptr, &variables[0], sizeof(float)*numSupplies, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(v_vars_ptr, &variables[numSupplies], sizeof(float)*numDemands, cudaMemcpyDeviceToDevice));
}

__host__ void find_dual_using_seq_bfs(float * u_vars_ptr, float * v_vars_ptr, 
        int * h_length, int * h_start, int * h_Ea, bool * h_visited, float * h_variables,
        int * d_adjMtx_ptr, float * h_costs_ptr, int numSupplies, int numDemands) {

        int V = numSupplies + numDemands;

        thrust::fill(thrust::host, h_visited, h_visited + V, false);
        thrust::fill(thrust::host, h_variables, h_variables + V, 0.0f);

        // Initialize >>
        std::queue<int> assigned_parents;
        h_visited[0] = true;
        // The value of u0 is already zero as initialized
        assigned_parents.push(0);

        // Perform a BFS on Host (trickle down) >> 
        int parent, child, row, col;

        while (!assigned_parents.empty()) {
                parent = assigned_parents.front();
                for (int i = h_start[parent]; i < h_start[parent] + h_length[parent]; i++) {
                        child = h_Ea[i];
                        if (!h_visited[child]) {
                                h_visited[child] = true;
                                row = min(parent, child);
                                col = max(parent, child) - numSupplies;
                                h_variables[child] = h_costs_ptr[row*numDemands + col] - h_variables[parent];
                                assigned_parents.push(child);
                        }
                }
                assigned_parents.pop();
        }

        // Transfer back to GPU >> 
        gpuErrchk(cudaMemcpy(u_vars_ptr, &h_variables[0], sizeof(int)*numSupplies, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(v_vars_ptr, &h_variables[numSupplies], sizeof(int)*numDemands, cudaMemcpyHostToDevice));

}