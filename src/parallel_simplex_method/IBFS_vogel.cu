#include "IBFS_vogel.h"

__global__ void computeDifferences(vogelDifference * minima, MatrixCell * row_book, MatrixCell * col_book,
                                            int n_rows, int n_cols, int prev_eliminated)
{

    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    float _diff;
    int current_second_min;
    vogelDifference min_indexes = minima[indx];
    if (indx < n_rows && prev_eliminated >= n_rows)
    {
        /* Then in the previous step a column was eliminated -
        Now, indx corresponds to row ID's in vect
        In this case row minima are updated and column minima are maintained
                1. RowMinima exist at indx < n_rows in diff
                2. Elimiated col indx = prev_eliminated - n_rows
                3. Now see if the corresponding diff for this row
                has either ileast-1 or ileast-2 equal to elimiated col indx

                In row_book index from [indx*n_cols] upto [(indx+1)*n_cols] is the sorted order of indx'th
                row costs

                4. If previous was true then increment ileast1 ad ileast2 accordingly, o/w Untouched
        */
        int rejectColIdx = prev_eliminated - n_rows;
        MatrixCell least1 = row_book[indx * (n_cols + 2) + min_indexes.ileast_1]; // Minimum
        MatrixCell least2 = row_book[indx * (n_cols +2) + min_indexes.ileast_2]; // Second Minimum
        
        if (least1.col == rejectColIdx)
        {
            // assign ileast2 to ileast1 and increment ileast2 to next uncovered
            min_indexes.ileast_1 = min_indexes.ileast_2;
            least1 = least2;
            // It is within range and uncovered
            while (min_indexes.ileast_2 < n_cols - 1)
            {
                // Loop will keep moving untill - covered = false is found for the corresponding column
                min_indexes.ileast_2 += 1;
                if (minima[n_rows + row_book[indx * (n_cols +2) + min_indexes.ileast_2].col].ileast_1 < n_rows) {
                    break;
                }
            }
            least2 = row_book[indx * (n_cols + 2) + min_indexes.ileast_2];
        }

        else if (least2.col == rejectColIdx) {
            // let ileast1 stay what it is and increment ileast2 to next uncovered
            while (min_indexes.ileast_2 < n_cols - 1)
            {
                // Loop will keep moving untill - covered = false is found for the corresponding column
                min_indexes.ileast_2 += 1;
                if (minima[n_rows + row_book[indx * (n_cols +2) + min_indexes.ileast_2].col].ileast_1 < n_rows) {
                    break;
                }
            }
            least2 = row_book[indx * (n_cols + 2) + min_indexes.ileast_2];
        }

        // Computing New Row differences
        _diff = least2.cost - least1.cost;
        min_indexes.idx = least1.col;
        min_indexes.diff = _diff;
        minima[indx] = min_indexes;
    }

    else if (indx >= n_rows && indx < n_cols + n_rows && prev_eliminated < n_rows) {
            
        /* Then a row was eliminated (The covered for this has already been set to true before invoke in O(1)) -
        Now, indx corresponds to column ID's in vect, only do this operation for uncovered columns
        In this case col minima are updated and row minima are maintained
                1. ColMinima exist at indx >= n_rows in diff
                2. Elimiated row indx = prev_eliminated (-> directly)
                3. Now see if the corresponding diff for indx'th col
                has either ileast-1 or ileast-2 equal to elimiated row indx

                In col_book index from [indx*n_rows] upto [(indx+1)*n_rows] is the sorted order of indx'th
                col costs

                4. If previous was true then increment ileast1 ad ileast2 accordingly, o/w Untouched
        */
        int rejectRowIdx = prev_eliminated;
        MatrixCell least1 = col_book[(indx - n_rows) * (n_rows+2) + min_indexes.ileast_1]; // Minimum
        MatrixCell least2 = col_book[(indx - n_rows) * (n_rows+2) + min_indexes.ileast_2]; // Second Minimum
                
        if (least1.row == rejectRowIdx)
        {
            // assign ileast2 to ileast1 and increment ileast2 to next uncovered
            min_indexes.ileast_1 = min_indexes.ileast_2;
            least1 = least2;
            while (min_indexes.ileast_2 < n_rows - 1)
            {
                // Loop will keep moving untill - covered = false is found for the corresponding row
                min_indexes.ileast_2 += 1;
                if (minima[col_book[(indx - n_rows) * (n_rows+2) + min_indexes.ileast_2].row].ileast_1 < n_cols) {
                    break;
                }
            }
            least2 = col_book[(indx - n_rows) * (n_rows+2) + min_indexes.ileast_2];
        }

        else if (least2.row == rejectRowIdx)
        {
            // let ileast1 stay what it is and increment ileast2 to next uncovered
            while (min_indexes.ileast_2 < n_rows - 1)
            {
                // Loop will keep moving untill - covered = false is found for the corresponding row
                min_indexes.ileast_2 += 1;
                if (minima[col_book[(indx - n_rows) * (n_rows+2) + min_indexes.ileast_2].row].ileast_1 < n_cols) {
                    break;
                }
            }
            least2 = col_book[(indx - n_rows) * (n_rows+2) + min_indexes.ileast_2];
        }

        // Computing Column differences
        _diff = least2.cost - least1.cost;
        min_indexes.idx  = least1.row;
        min_indexes.diff = _diff;
        minima[indx] = min_indexes;
    }
}


// Initialization Kernel >> Launched on (M*N) threads
__global__ void fill_deviceCostMatrix_row_book(MatrixCell * book, MatrixCell * d_costMatrix, int numSupplies, int numDemands, int width) {

    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

    if (row_indx < numSupplies && col_indx < numDemands) {
        int gid = row_indx*width + col_indx;
        book[gid] = d_costMatrix[row_indx*numDemands + col_indx];
    }
}


__global__ void fill_deviceCostMatrix_col_book(MatrixCell * book, MatrixCell * d_costMatrix, int numSupplies, int numDemands, int width) {

    int row_indx = blockIdx.y*blockDim.y + threadIdx.y;
    int col_indx = blockIdx.x*blockDim.x + threadIdx.x;

    if (row_indx < numSupplies && col_indx < numDemands) {
        int gid = col_indx*width + row_indx;
        book[gid] = d_costMatrix[row_indx*numDemands + col_indx];
    }
}


// Initialization Kernel >> Launched on (M) threads >>
__global__ void offset_rowbook(MatrixCell * row_book, int numSupplies, int numDemands) {

    // Inserting in the supply_row
        // Second Max goes to col index - [numDemands]
        // Max goes to the col index - [numDemands+1] 
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    if (indx < numSupplies) {
        int gid;
        
        MatrixCell max_cell_1 = {.row = indx, .col = numDemands, .cost = 1.0f*INT16_MAX};
        MatrixCell max_cell_2 = {.row = indx, .col = numDemands+1, .cost = 1.0f*INT16_MAX};

        // Inserting second Max
        gid = indx*(numDemands+2) + numDemands;
        row_book[gid] = max_cell_1;

        // Inserting Max
        gid = indx*(numDemands+2) + numDemands+1;
        row_book[gid] = max_cell_2;
    }
}

__global__ void override_row_sorts(MatrixCell * row_book, int numSupplies, int numDemands) {

    // Inserting in the supply_row
        // Second Max goes to col index - [numDemands]
        // Max goes to the col index - [numDemands+1] 
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    if (indx < numSupplies) {
        // Updating Max
        int gid = indx*(numDemands+2) + numDemands+1;
        row_book[gid].cost -= 1;
    }
}

__global__ void offset_colbook(MatrixCell * col_book, int numSupplies, int numDemands) {

    int indx = blockIdx.x*blockDim.x + threadIdx.x;

    // Inserting in the demand column
        // Second Max goes to row index - [numSupplies]
        // Max goes to the row index - [numSupplies+1] 
    if (indx < numDemands) {
        
        int gid;
        MatrixCell max_cell_1 = {.row = numSupplies, .col = indx, .cost = 1.0f*INT16_MAX};
        MatrixCell max_cell_2 = {.row = numSupplies+1, .col = indx, .cost = 1.0f*INT16_MAX};

        // Inserting second Max
        gid = indx*(numSupplies+2) + numSupplies;
        col_book[gid] = max_cell_1;

        // Inserting Max
        gid = indx*(numSupplies+2) + (numSupplies+1) ;
        col_book[gid] = max_cell_2;
    }
}

__global__ void override_col_sorts(MatrixCell * col_book, int numSupplies, int numDemands) {

    // Inserting in the supply_row
        // Second Max goes to col index - [numDemands]
        // Max goes to the col index - [numDemands+1] 
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    if (indx < numDemands) {
        // Updating Max
        int gid = indx*(numSupplies+2) + (numSupplies+1);
        col_book[gid].cost -= 1;
    }
}


/*
Doc: Pending -
Improvement Idea -
1. Would Reordering of rows and columns will improve performance?
        - First reorder based on cover
        - Reorder based on minimum in prev_eliminated
        - Also reorder demand supply accordingly >> Maintain the original indexes to assign flow
2. Can we discard Matrix Cell Data Structure after the sorting job is done in step 1?
*/
__host__ void find_vogel_bfs_parallel(int *supplies, int *demands, MatrixCell * costMatrix, flowInformation * feasible_flows, 
        int numSupplies, int numDemands)
{

    // Step 0 :
    std::cout << "FINDING BFS : Vogel Device Kernel - Step 0 : Setting up book-keeping structures" << std::endl;
    thrust::device_vector<MatrixCell> device_costMatrix(costMatrix, costMatrix + (numSupplies * numDemands));
    MatrixCell * device_costMatrix_ptr = thrust::raw_pointer_cast(device_costMatrix.data());

    // Book-keeping Structures on device >>

    // *********************************
    // Row Wise Bookeeping
    // *********************************

    MatrixCell * device_costMatrixRowBook;
    gpuErrchk(cudaMalloc((void **) & device_costMatrixRowBook, sizeof(MatrixCell)*(numSupplies)*(numDemands+2)));

    dim3 dB_1(blockSize, blockSize, 1);
    dim3 dG_1(ceil(1.0*numDemands/blockSize), ceil(1.0*numSupplies/blockSize), 1);        
    fill_deviceCostMatrix_row_book<<< dG_1, dB_1 >>>(
        device_costMatrixRowBook, device_costMatrix_ptr, numSupplies, numDemands, numDemands+2);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    dim3 dB_2(blockSize, 1, 1);
    dim3 dG_2(ceil(1.0*numSupplies/blockSize), 1, 1);
    offset_rowbook<<< dG_2, dB_2 >>>(
        device_costMatrixRowBook, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // auto rowgen = [=]  __device__ (MatrixCell x) {return x.row;};
    // -> Alternative way using such lambda exp (compile with --expt-extended-lambda -std=c++11)
    // same implement on column section

    rowgen op1;
    thrust::device_vector<int> device_rowSegments(numSupplies * (numDemands+2));
    thrust::transform(thrust::device, 
        device_costMatrixRowBook, device_costMatrixRowBook + numSupplies*(numDemands+2), // input_range 1
        device_rowSegments.begin(),                                       // output_range
        op1);                                                             // unary func

    // mytime = dtime_usec(0);
    thrust::stable_sort_by_key(thrust::device,
        device_costMatrixRowBook, device_costMatrixRowBook + numSupplies*(numDemands+2),
        device_rowSegments.begin(),
        compareCells());

    thrust::stable_sort_by_key(thrust::device, 
        device_rowSegments.begin(), device_rowSegments.end(),
        device_costMatrixRowBook);

    override_row_sorts<<< dG_2, dB_2 >>>(
        device_costMatrixRowBook, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Stable Sort Examdevice_costMatrixRowBook, device_costMatrixRowBook + numSupplies*(numDemands+2)ple >>

    // A = [maxtrixCell(1),2,30,2,1,8]
    // A_segments = [A[i].row, .....]

    // A_segments = [0,1,2,0,1,2]
    // --
    // sorted_A = [1,1,2,2,8,30]
    // A_segemnts = [0,1,0,1,1,0]
    // --
    // // sort A using A segments as key
    // sorted_A = [1,2,30, 1,2,8]
    // A_segemnts = [0,0,0, 1,1,1]

    // *********************************
    // Column Wise Bookeeping
    // *********************************

    MatrixCell * device_costMatrixColBook;
    gpuErrchk(cudaMalloc((void **) & device_costMatrixColBook, sizeof(MatrixCell)*(numSupplies+2)*(numDemands)));

    // dim3 dB_1(blockSize, blockSize, 1);
    // dim3 dG_1(ceil(1.0*numDemands/blockSize), ceil(1.0*numSupplies/blockSize), 1);        
    fill_deviceCostMatrix_col_book<<< dG_1, dB_1 >>>(
        device_costMatrixColBook, device_costMatrix_ptr, numSupplies, numDemands, numSupplies+2);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    dim3 dB_3(blockSize, 1, 1);
    dim3 dG_3(ceil(1.0*numDemands/blockSize), 1, 1);
    offset_colbook<<< dG_3, dB_3 >>>(
        device_costMatrixColBook, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    colgen op2;
    thrust::device_vector<int> device_colSegments((numSupplies+2) * numDemands);
    thrust::transform(thrust::device, 
        device_costMatrixColBook, device_costMatrixColBook + (numSupplies+2)*(numDemands), // input_range 1
        device_colSegments.begin(),                                       // output_range
        op2);                                                             // unary func

    // mytime = dtime_usec(0);
    thrust::stable_sort_by_key(thrust::device, 
        device_costMatrixColBook, device_costMatrixColBook + (numSupplies+2)*(numDemands),
        device_colSegments.begin(),
        compareCells());
    thrust::stable_sort_by_key(thrust::device, 
        device_colSegments.begin(), device_colSegments.end(),
        device_costMatrixColBook);

    override_col_sorts<<< dG_3, dB_3 >>>(
        device_costMatrixColBook, numSupplies, numDemands);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    /* ***************************************************
            DEBUG UTILITY
    *************************************************** */
    // MatrixCell * host_costMatrixColBook = (MatrixCell *) malloc(sizeof(MatrixCell)*(numSupplies+2)*(numDemands));
    // gpuErrchk(cudaMemcpy(host_costMatrixColBook, device_costMatrixColBook, sizeof(MatrixCell)*(numSupplies+2)*(numDemands), cudaMemcpyDeviceToHost));

    // std::cout<<"Column Book:"<<std::endl;
    // for (size_t i = 0; i < (numSupplies+2)*(numDemands); i++) {
    //         std::cout << "device_costMatrixColBook[" << i << "] = " << host_costMatrixColBook[i] << std::endl;
    // }

    // MatrixCell * host_costMatrixRowBook = (MatrixCell *) malloc(sizeof(MatrixCell)*(numSupplies)*(numDemands+2));
    // gpuErrchk(cudaMemcpy(host_costMatrixRowBook, device_costMatrixRowBook, sizeof(MatrixCell)*(numSupplies)*(numDemands+2), cudaMemcpyDeviceToHost));

    // std::cout<<"Row Book:"<<std::endl;
    // for (size_t i = 0; i < (numSupplies)*(numDemands+2); i++) {
    //         std::cout << "device_costMatrixRowBook[" << i << "] = " << host_costMatrixRowBook[i] << std::endl;
    // }

    // exit(0);

    // *********************************
    // END OF INITIALIZATION | Prepare for iterations >>
    // *********************************

    // Freeup Some Memory
    // device_rowSegments.clear();
    // device_colSegments.clear();
    std::cout << "FINDING BFS : Vogel Device Kernel - Step 1 : Preparing for assignment" << std::endl;

    /*
    Illustration:

    - Row-current Minima pointer [[0,1],[0,1],[0,1] . . .]
    - Col-current Minima pointer [[0,1],[0,1],[0,1] . . .]

    a of strcuts vogelDifference was created to accomodate the above need
    Further differences_vector concatenates both row differences and column differences vector
    so that that row/col reduction operations are simulataneous
    */

    // Add something to host for keeping residuals -
    thrust::host_vector<int> res_supplies(supplies, supplies + numSupplies);
    thrust::host_vector<int> res_demands(demands, demands + numDemands);
    thrust::device_vector<vogelDifference> differences_vector(numSupplies + numDemands);

    vogelDifference *vect = thrust::raw_pointer_cast(differences_vector.data());

    // Some more book-keeping -
    float _d = 1.0 * INT_MIN;
    int prev_eliminated = -1, flow_row, flow_col;
    MatrixCell host_assignmentCell;
    flowInformation _this_flow;

    // [ n_rows + n_cols - 1 ] ierations will generate the basic feasible solution
    // Untested assumption [0] Absence of degeneracy - Todo
    std::cout << "FINDING BFS : Vogel Device Kernel - Step 2 : Running Initial Assignment" << std::endl;
    int counter = 0;
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(ceil(1.0 * differences_vector.size() / blockSize), 1, 1);
    // numSupplies + numDemands - 1

    while (counter < (numSupplies + numDemands - 1))
    {
        /* Procedure:
        1. Find the max of differences - parallel op
        2. Jump to ileast_1 in the corresponding row/col diff using the book - seq. op
        3. Consume Demand and Supply - seq. op
        4. Eliminate the respective - seq. op
        5. update differences vect - parallel
        */
        
        // ************************
        // Map Step
        // ************************

        if (counter == 0) {

            // Initialize row differences
            computeDifferences<<< dimGrid, dimBlock >>>(vect, device_costMatrixRowBook, device_costMatrixColBook, numSupplies, numDemands, numSupplies+numDemands);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // Initialize column differences
            computeDifferences<<< dimGrid, dimBlock >>>(vect, device_costMatrixRowBook, device_costMatrixColBook, numSupplies, numDemands, -1);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

        }

        else {
        
            computeDifferences<<< dimGrid, dimBlock >>>(vect, device_costMatrixRowBook, device_costMatrixColBook, numSupplies, numDemands, prev_eliminated);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        
        }

        // ************************
        // Reduction Step
        // ************************

        thrust::device_vector<vogelDifference>::iterator _iter = thrust::max_element(differences_vector.begin(), differences_vector.end(), compareDiff());
        int max_index = _iter - differences_vector.begin();
        vogelDifference host_maxDiff = differences_vector[max_index];

        // Following are all constant time ops, everything is happening on host -
        if (max_index >= numSupplies)
        {
                // some column has max diff
                flow_col = max_index - numSupplies;
                flow_row = host_maxDiff.idx;
        }
        else
        {
                // some row has max diff
                flow_row = max_index;
                flow_col = host_maxDiff.idx;
        }

        // ************************
        // Allocation Step
        // ************************

        if (res_demands[flow_col] > res_supplies[flow_row])
        {       
                // consume available supply and update demand
                _this_flow = {.source = flow_row, .destination = flow_col, .qty = std::max(1.0f*res_supplies[flow_row], epsilon)};
                feasible_flows[counter] = _this_flow;
                res_demands[flow_col] -= res_supplies[flow_row];
                res_supplies[flow_row] = 0;
                prev_eliminated = flow_row;
                // Device is informed about the assignment
                vogelDifference default_diff = {.idx = -1, .ileast_1 = numDemands, .ileast_2 = numDemands+1, .diff = -1.0f}; 
                // assigned a default to eliminated row
                differences_vector[prev_eliminated] = default_diff;
        }
        else
        {
                // satisfy current demand and update supply
                _this_flow = {.source = flow_row, .destination = flow_col, .qty = std::max(1.0f*res_demands[flow_col], epsilon)};
                feasible_flows[counter] = _this_flow;
                res_supplies[flow_row] -= res_demands[flow_col];
                res_demands[flow_col] = 0;
                prev_eliminated = numSupplies + flow_col;
                // Device is informed about the assignment
                vogelDifference default_diff = {.idx = -1, .ileast_1 = numSupplies, .ileast_2 = numSupplies+1, .diff = -1.0f}; 
                // assigned a default to eliminated column
                differences_vector[prev_eliminated] = default_diff;
        }

        // Device adopts the assignment
        counter++;

        // *****************************
        // !! DEBUGGGING UTILITY !!
        // Print Current Minima vector and allocations at the end of each iteration
        // *****************************
        // std::cout<<"Flow Row : "<<flow_row<<std::endl;
        // std::cout<<"Flow Col : "<<flow_col<<std::endl;
        // std::cout<<"Counter : "<<counter<<std::endl;
        // std::cout<<"Qty Allocated : "<<_this_flow.qty<<std::endl;
        // for (size_t i = 0; i < differences_vector.size(); i++) {
        //         std::cout << "differences_vector[" << i << "] = " << differences_vector[i] << std::endl;
        // }

    }

    std::cout << "FINDING BFS : Vogel Device Kernel - END : Initial Assignment Complete" << std::endl;
}


void find_vogel_bfs_sequencial(int * supplies, int * demands, MatrixCell * costMatrix, 
        flowInformation * flows, int numSupplies, int numDemands) {
    
    std::cout<<"Vogel's Approximation sequencial BFS Method"<<std::endl;
    // Book-keeping stuff >>
    int coveredRows = 0 , coveredColumns = 0;
    int * residual_supply = (int *) malloc(numSupplies*sizeof(int));
    std::memcpy(residual_supply, supplies, numSupplies*sizeof(int));

    int *residual_demand = (int *) malloc(numDemands*sizeof(int));
    std::memcpy(residual_demand, demands, numDemands*sizeof(int));

    int * rowCovered = (int *) calloc(numSupplies, sizeof(int));
    int * colCovered = (int *) calloc(numDemands, sizeof(int));    
    int * differences = (int *) calloc(numSupplies + numDemands, sizeof(int));
    std::cout<<"\tSTEP 0 : Setting-up book-keeping structs"<<std::endl;

    std::cout<<"\tSTEP 1 : Running Vogel's Heuristic"<<std::endl;
    bool prev_row = true, prev_col = true; // Denotes if a row/col was eliminated in previous iteration

    while ((coveredRows + coveredColumns) < (numDemands+numSupplies-1)) {
        
        // std::cout<<"Iteration - "<<coveredColumns+coveredRows<<std::endl;
        float temp1, temp2, tempDiff;
        float costTemp;
        int i_tempDiff, i_minCost;
        // std::cout<<"prev_row = "<<prev_row<<std::endl;
        // std::cout<<"prev_col = "<<prev_col<<std::endl;

        // Re/Calculate row differences >> 
        if (prev_row) {
            for (int i=0; i< numSupplies; i++){
                if (rowCovered[i] == 0) {
                    temp1 = INT_MAX;
                    temp2 = INT_MAX;
                    for (int j=0; j< numDemands; j++) {
                        // Only look at columns not covered >> 
                        if (colCovered[j] == 0) {
                            float entry = costMatrix[i*numDemands + j].cost;
                            if (entry <= temp1) {
                                temp2 = temp1;
                                temp1 = entry;
                            }
                            else if (entry <= temp2) {
                                temp2 = entry;
                            }
                        }
                    }
                    differences[i] = temp2 - temp1;
                }
                else {
                    differences[i] = INT_MIN;
                }
            }
            prev_row = false;
        }
        

        // Re/Calculate col differences >> 
        if (prev_col) {
            for (int j=0; j< numDemands; j++){
                if (colCovered[j] == 0) {
                    temp1 = INT_MAX;
                    temp2 = INT_MAX;
                    // Only look at rows not covered >>
                    for (int i=0; i< numSupplies; i++) {
                        if (rowCovered[i] == 0) {
                            float entry = costMatrix[i*numDemands + j].cost;
                            if (entry <= temp1) {
                                temp2 = temp1;
                                temp1 = entry;
                            }
                            else if (entry <= temp2) {
                                temp2 = entry;
                            }
                        }
                    }
                    differences[numSupplies + j] = temp2 - temp1;
                }
                else {
                    differences[numSupplies + j] = INT_MIN;
                }
            }
            prev_col = false;
        }
        
        // Determine the maximum of differences - (Reduction)
        tempDiff = INT_MIN;
        i_tempDiff = -1;
        for (int i=0; i < numSupplies + numDemands; i++) {
            if (differences[i] > tempDiff) {
                // tie broken by first seen
                tempDiff = differences[i];
                i_tempDiff = i;
            }
        }
        
        int counter = coveredRows + coveredColumns;
        // Check if row or col difference and determine corresponding min cost
        // Now we have Basic row and col
        // Assign flow based on availability 
        if (i_tempDiff >= numSupplies) {
            // This is a col difference
            i_tempDiff -= numSupplies;
            // In this column index find the min cost
            costTemp = INT_MAX;
            for (int i=0; i<numSupplies; i++) {
                float entry = costMatrix[i*numDemands + i_tempDiff].cost;
                if (entry < costTemp && rowCovered[i] == 0) {
                    costTemp = entry;
                    i_minCost = i;
                }
            }

            // std::cout<<"Col: Index-1 "<<i_tempDiff<<std::endl;
            // std::cout<<"Col: Index-2 "<<i_minCost<<std::endl;

            // std::cout<<" Res-Sup "<<residual_supply[i_minCost]<<std::endl;
            // std::cout<<" Res-Demand "<<residual_demand[i_tempDiff]<<std::endl;
            
            // Min cost row is i_minCost
            if (residual_demand[i_tempDiff] > residual_supply[i_minCost]){
                
                flows[counter] = {.source = i_minCost, .destination = i_tempDiff,
                                .qty = 1.0f*residual_supply[i_minCost]};
                residual_demand[i_tempDiff] -= residual_supply[i_minCost];
                residual_supply[i_minCost] = 0;
                rowCovered[i_minCost] = 1;
                prev_row = true;
                coveredRows += 1;
            }
            else {
                flows[counter] = {.source = i_minCost, .destination = i_tempDiff, 
                                        .qty = 1.0f*residual_demand[i_tempDiff]};
                residual_supply[i_minCost] -= residual_demand[i_tempDiff];
                residual_demand[i_tempDiff] = 0;
                colCovered[i_tempDiff] = 1;
                prev_col = true;
                coveredColumns += 1;
            }
        }
        else {
            // Then this is a row difference
            // In this row find the min cost
            costTemp = INT_MAX;
            
            for (int j=0; j<numDemands; j++) {
                float entry = costMatrix[i_tempDiff*numDemands + j].cost;
                if (entry < costTemp && colCovered[j] == 0) {
                    costTemp = entry;
                    i_minCost = j;
                }
            }
            // minCost column is i_minCost
            // std::cout<<"Row: Index-1 "<<i_tempDiff<<std::endl;
            // std::cout<<"Row: Index-2 "<<i_minCost<<std::endl;

            // std::cout<<" Res-Sup "<<residual_supply[i_tempDiff]<<std::endl;
            // std::cout<<" Res-Demand "<<residual_demand[i_minCost]<<std::endl;

            if (residual_demand[i_minCost] > residual_supply[i_tempDiff]){
                flows[counter] = {.source = i_tempDiff, .destination = i_minCost, 
                                    .qty = 1.0f*residual_supply[i_tempDiff]};
                residual_demand[i_minCost] -= residual_supply[i_tempDiff];
                residual_supply[i_tempDiff] = 0;
                rowCovered[i_tempDiff] = 1;
                prev_row = true;
                coveredRows += 1;
            }
            else {
                flows[counter] = {.source = i_tempDiff, .destination = i_minCost,
                                    .qty = 1.0f*residual_demand[i_minCost]};
                residual_supply[i_tempDiff] -= residual_demand[i_minCost];
                residual_demand[i_minCost] = 0;
                colCovered[i_minCost] = 1;
                prev_col = true;
                coveredColumns += 1;
            }  
        }
    }
    std::cout<<"\tVogel's Heuristic Completed!"<<std::endl;
}