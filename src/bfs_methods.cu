#include "bfs_methods.h"

__host__ void find_nw_corner_bfs_seq(int * supplies, int * demands, MatrixCell * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands){

        std::cout<<"Running Northwest Corner Seq BFS Method"<<std::endl;

        // Step 1 :: Jumpt to NW corner >>
        int current_row_number = 0;
        int current_col_number = 0;
        int current_demand = demands[current_row_number];
        int current_supply = supplies[current_col_number];

        // Allocate flow equal to minimum of demand and supply and update the buffer accordingly >>
        while (current_row_number < matrixSupplies && current_col_number < matrixDemands) {

                if (current_demand >= current_supply) {
                        flows[current_row_number*matrixDemands + current_col_number] = current_supply;
                        current_demand = current_demand -  current_supply;
                        current_row_number++;
                        current_supply = supplies[current_row_number];
                }
                else {
                        flows[current_row_number*matrixDemands + current_col_number] = current_demand;
                        current_supply = current_supply -  current_demand;
                        current_col_number++;
                        current_demand = demands[current_col_number];
                }
        }
        std::cout<<"Feasible BFS Generated!"<<std::endl;
}


struct rowgen{
    __host__ __device__ int operator()(MatrixCell &x) const{
        return x.row;
    }
};

struct colgen{
    __host__ __device__ int operator()(MatrixCell &x) const{
        return x.col;
    }
};

struct compareCells {
  __host__ __device__  bool operator()(const MatrixCell i, const MatrixCell j) const{
        return (i.cost <= j.cost);
    }
};

struct compareDiff {
  __host__ __device__  bool operator()(const vogelDifference i, const vogelDifference j) const{
        return (i.diff <= j.diff);
    }
};

__global__ void initializeDifferencesVector(vogelDifference * minima, MatrixCell * row_book, MatrixCell * col_book, int n_rows, int n_cols) {
        
        int indx = blockIdx.x*blockDim.x + threadIdx.x;
        
        if (indx < n_rows+n_cols) {
                float _diff;
                if (indx < n_rows) {
                        _diff = row_book[indx*n_cols + 1].cost - row_book[indx*n_cols].cost;
                }
                else { 
                        _diff = col_book[(indx-n_rows)*n_rows + 1].cost - col_book[(indx-n_rows)*n_rows].cost;
                }
                vogelDifference d = {.idx = indx, .ileast_1 = 0, .ileast_2 = 1, .diff = _diff};
                minima[indx] = d;
        }
}


__global__ void updateDifferences(
        vogelDifference * minima, // Row and Column Differences 
        MatrixCell *row_book, MatrixCell *col_book, // Row Wise and Column wise Look up (2n^2 Data Struct)
        bool * covered,                 // Flags if particular row/col was covered
        int n_rows, int n_cols, int prev_eliminated) {
        
        int indx = blockIdx.x*blockDim.x + threadIdx.x;

        // Note - Following if and else are simply the pivots of each other
        // Same kernel uses prev_elminated values and accordingly updates row/col differences
        if (indx < n_rows && prev_eliminated >= n_rows && !covered[indx]) {

                /* Then a column was eliminated (The covered for this has already been set to true before invoke in O(1)) -
                Now, indx corresponds to row ID's in vect, Only do this operator for uncovered rows
                In this case row minima are updated and column minima are maintained
                        1. RowMinima exist at indx < n_rows in diff
                        2. Elimiated col indx = prev_eliminated - n_rows
                        3. Now see if the corresponding diff for indx'th row 
                        has either ileast-1 or ileast-2 equal to elimiated col indx
                        
                        In row_book index from [indx*n_cols] upto [(indx+1)*n_cols] is the sorted order of indx'th 
                        row costs 
                        
                        4. If previous was true then increment ileast1 ad ileast2 accordingly, o/w Untouched
                */
                int rejectColIdx = prev_eliminated - n_rows;
                float _diff;
                MatrixCell least1 = row_book[indx*n_cols + minima[indx].ileast_1]; // Minimum
                MatrixCell least2 = row_book[indx*n_cols + minima[indx].ileast_2]; // Second Minimum
                if (least1.col == rejectColIdx){
                        // assign ileast2 to ileast1 and increment ileast2 to next uncovered
                        minima[indx].ileast_1 = minima[indx].ileast_2;
                        minima[indx].ileast_2 = min(minima[indx].ileast_2+1, n_cols-1);
                        while (covered[n_rows + row_book[indx*n_cols + minima[indx].ileast_2].col] && minima[indx].ileast_2 < n_cols-1) {
                                // Loop will keep moving untill - covered = false is found for the corresponding column
                                // Boundary condition need to be applied >> pending!
                               minima[indx].ileast_2 += 1;
                        }
                        _diff = row_book[indx*n_cols + minima[indx].ileast_2].cost - 
                                                row_book[indx*n_cols + minima[indx].ileast_1].cost;
                        minima[indx].diff = _diff;
                }
                else if (least2.col == rejectColIdx) {
                        // let ileast1 stay what it is and increment ileast2 to next uncovered
                        // Todo: TestBoundary Condition
                        minima[indx].ileast_2 = min(minima[indx].ileast_2+1, n_cols-1);
                        while (covered[n_rows + row_book[indx*n_cols + minima[indx].ileast_2].col] && minima[indx].ileast_2 < n_cols-1) {
                                // Loop will keep moving untill - covered = false is found for the corresponding column
                                // Boundary condition need to be applied >> pending!
                               minima[indx].ileast_2 += 1;
                        }
                        _diff = row_book[indx*n_cols + minima[indx].ileast_2].cost - 
                                                row_book[indx*n_cols + minima[indx].ileast_1].cost;
                        minima[indx].diff = _diff;
                }
        }

        else if (indx >= n_rows && indx < n_rows + n_cols && prev_eliminated < n_rows && !covered[indx]) {

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
                float _diff;
                MatrixCell least1 = col_book[(indx-n_rows)*n_rows + minima[indx].ileast_1]; // Minimum
                MatrixCell least2 = col_book[(indx-n_rows)*n_rows + minima[indx].ileast_2]; // Second Minimum
                if (least1.row == rejectRowIdx){
                        // assign ileast2 to ileast1 and increment ileast2 to next uncovered
                        minima[indx].ileast_1 = minima[indx].ileast_2;
                        minima[indx].ileast_2 = min(minima[indx].ileast_2+1, n_rows-1);
                        while (covered[col_book[(indx-n_rows)*n_rows + minima[indx].ileast_2].row] && minima[indx].ileast_2 < n_rows-1) {
                                // Loop will keep moving untill - covered = false is found for the corresponding row
                                // Boundary condition need to be applied >> pending!
                               minima[indx].ileast_2 += 1;
                        }
                        _diff = col_book[(indx-n_rows)*n_rows + minima[indx].ileast_2].cost - 
                                                col_book[(indx-n_rows)*n_rows + minima[indx].ileast_1].cost;
                        minima[indx].diff = _diff;
                }
                else if (least2.row == rejectRowIdx) {
                        // let ileast1 stay what it is and increment ileast2 to next uncovered
                        // Todo: TestBoundary Condition
                        minima[indx].ileast_2 = min(minima[indx].ileast_2+1, n_rows-1);
                        while (covered[col_book[(indx-n_rows)*n_rows + minima[indx].ileast_2].row] && minima[indx].ileast_2 < n_rows-1) {
                                // Loop will keep moving untill - covered = false is found for the corresponding row
                                // Boundary condition need to be applied >> pending!
                               minima[indx].ileast_2 += 1;
                        }
                        _diff = col_book[(indx-n_rows)*n_rows + minima[indx].ileast_2].cost - 
                                                col_book[(indx-n_rows)*n_rows + minima[indx].ileast_1].cost;
                        minima[indx].diff = _diff;
                }
        }

}

/*
Doc: Pending
*/
__host__ void find_vogel_bfs_parallel(int * supplies, int * demands, MatrixCell * costMatrix, 
        int * flows, int matrixSupplies, int matrixDemands) {
        
        // Step 0 :
        std::cout<<"FINDING BFS : Vogel Device Kernel - Step 0 : Setting up book-keeping structures"<<std::endl; 
        thrust::device_vector<MatrixCell> device_costMatrix(costMatrix, costMatrix + matrixSupplies*matrixDemands);

        // Book-keeping Structures on device >>
        
        // *********************************
        // Row Wise Bookeeping
        // *********************************

        thrust::device_vector<MatrixCell> device_costMatrixRowBook(matrixSupplies*matrixDemands);  
        thrust::copy(thrust::device, device_costMatrix.begin(),device_costMatrix.end(), device_costMatrixRowBook.begin());
        
        // auto rowgen = [=]  __device__ (MatrixCell x) {return x.row;};  
        // -> Alternative way using such lambda exp (compile with --expt-extended-lambda -std=c++11) 
        // same implement on column section
        
        rowgen op1;
        thrust::device_vector<int> device_rowSegments(matrixSupplies*matrixDemands);
        thrust::transform(
                device_costMatrixRowBook.begin(), device_costMatrixRowBook.end(),  // input_range 1
                device_rowSegments.begin(),                                        // output_range        
                op1);                                                               // unary func
        
        // mytime = dtime_usec(0);
        thrust::stable_sort_by_key(
                device_costMatrixRowBook.begin(), device_costMatrixRowBook.end(), 
                device_rowSegments.begin(),
                compareCells()
                );
        thrust::stable_sort_by_key(
                device_rowSegments.begin(), device_rowSegments.end(), 
                device_costMatrixRowBook.begin());
        
        cudaDeviceSynchronize();
        // mytime = dtime_usec(mytime);
        // std::cout << "vectorized sorting time for rows:" << mytime/(float)USECPSEC << "s" << std::endl;

        // *********************************
        // Column Wise Bookeeping
        // *********************************

        thrust::device_vector<MatrixCell> device_costMatrixColBook(matrixSupplies*matrixDemands);  
        thrust::copy(thrust::device, device_costMatrix.begin(),device_costMatrix.end(), device_costMatrixColBook.begin());
        
        colgen op2;
        thrust::device_vector<int> device_colSegments(matrixSupplies*matrixDemands);
        thrust::transform(
                device_costMatrixColBook.begin(), device_costMatrixColBook.end(),  // input_range 1
                device_colSegments.begin(),                                        // output_range        
                op2);                                                               // unary func
        
        // mytime = dtime_usec(0);
        thrust::stable_sort_by_key(
                device_costMatrixColBook.begin(), device_costMatrixColBook.end(), 
                device_colSegments.begin(),
                compareCells()
                );
        thrust::stable_sort_by_key(
                device_colSegments.begin(), device_colSegments.end(), 
                device_costMatrixColBook.begin());
        
        cudaDeviceSynchronize();
        
        // mytime = dtime_usec(mytime);
        // std::cout << "vectorized sorting time for cols:" << mytime/(float)USECPSEC << "s" << std::endl;

        // *********************************
        // Prepare for iterations >>
        // *********************************

        // Freeup Some Memory
        device_rowSegments.clear();
        device_colSegments.clear();
        std::cout<<"FINDING BFS : Vogel Device Kernel - Step 1 : Preparing for assignment"<<std::endl;

        /* 
        Illustration:

        - Row-current Minima pointer [[0,1],[0,1],[0,1] . . .] 
        - Col-current Minima pointer [[0,1],[0,1],[0,1] . . .]
        
        vogelDifference vector was created to accomodate the above need
        Further currentMinimaVect concatenates both row differences and column differences vector 
        so that that row/col reduction operations are simulataneous 
        */

        // Add something to host for keeping residuals - 
        thrust::host_vector<int> res_supplies(supplies, supplies+matrixSupplies);
        thrust::host_vector<int> res_demands(demands, demands+matrixDemands);

        thrust::device_vector<vogelDifference> currentMinimaVect(matrixSupplies+matrixDemands);
        thrust::device_vector<bool> rowColCovered(matrixSupplies+matrixDemands);
        thrust::fill(rowColCovered.begin(), rowColCovered.end(), false);

        vogelDifference * vect = thrust::raw_pointer_cast(currentMinimaVect.data());
        MatrixCell * row_book = thrust::raw_pointer_cast(device_costMatrixRowBook.data());
        MatrixCell * col_book = thrust::raw_pointer_cast(device_costMatrixColBook.data());
        bool * covered = thrust::raw_pointer_cast(rowColCovered.data());

        dim3 dimBlock(blockSize, 1, 1);
        dim3 dimGrid(ceil(1.0*currentMinimaVect.size()/blockSize),1,1);
        
        initializeDifferencesVector<<<dimGrid, dimBlock>>>(vect, row_book, col_book, matrixSupplies, matrixDemands);
        cudaDeviceSynchronize();

        // Some more book-keeping - 
        float _d = 1.0*INT_MIN;
        vogelDifference default_diff = {.idx = -1, .ileast_1 = -1, .ileast_2 = -1,  .diff = _d}; // assigned to eliminated item
        int prev_eliminated, flow_row, flow_col;
        MatrixCell host_assignmentCell;

        // n_rows + n_cols - 1 ierations will generate the basic feasible solution
        // Untested assumption - Absence of degeneracy - Todo
        std::cout<<"FINDING BFS : Vogel Device Kernel - Step 2 : Running Initial Assignment"<<std::endl; 
        int counter = 0;
        // matrixSupplies + matrixDemands - 1
        while (counter < matrixSupplies + matrixDemands - 1) 
        {
                /* Procedure:
                1. Find the max of differences - parallel op
                2. Jump to ileast_1 in the corresponding row/col diff using the book - seq. op
                3. Consume Demand and Supply - seq. op
                4. Eliminate the respective - seq. op
                */
                thrust::device_vector<vogelDifference>::iterator _iter = thrust::max_element(currentMinimaVect.begin(), currentMinimaVect.end(), compareDiff());
                vogelDifference host_maxDiff = currentMinimaVect[_iter -  currentMinimaVect.begin()];

                // Following are all constant time ops, everything is happening on host -
                if (host_maxDiff.idx >= matrixSupplies) {
                        // some column has max diff
                        flow_col = host_maxDiff.idx - matrixSupplies;
                        host_assignmentCell = device_costMatrixColBook[(host_maxDiff.idx - matrixSupplies)*matrixSupplies + host_maxDiff.ileast_1];
                        flow_row = host_assignmentCell.row;
                }
                else {
                        // some row has max diff
                        flow_row = host_maxDiff.idx;
                        host_assignmentCell = device_costMatrixRowBook[host_maxDiff.idx*matrixDemands + host_maxDiff.ileast_1];
                        flow_col = host_assignmentCell.col;
                }

                // std::cout<<"Flow Row = "<<flow_row<<std::endl;
                // std::cout<<"Flow Col = "<<flow_col<<std::endl;

                if (res_demands[flow_col] > res_supplies[flow_row]) {
                        // consume available supply and update demand
                        flows[flow_row*matrixDemands + flow_col] = res_supplies[flow_row];
                        res_demands[flow_col] -= res_supplies[flow_row];
                        res_supplies[flow_row] = 0;
                        prev_eliminated = flow_row;
                }
                
                else {
                        // satisfy current demand and update supply
                        flows[flow_row*matrixDemands + flow_col] = res_demands[flow_col];
                        res_supplies[flow_row] -= res_demands[flow_col];
                        res_demands[flow_col] = 0;
                        prev_eliminated = matrixSupplies + flow_col;
                }
                
                // Device is informed about the assignment
                rowColCovered[prev_eliminated] = true;
                currentMinimaVect[prev_eliminated] = default_diff;
                
                // Device adopts the assignment 
                updateDifferences<<<dimGrid, dimBlock>>>(vect, row_book, col_book, covered, matrixSupplies, matrixDemands, prev_eliminated);
                cudaDeviceSynchronize();
                counter++;
        }

        // for (size_t i = 0; i < currentMinimaVect.size(); i++) {
        //         std::cout << "currentMinimaVect[" << i << "] = " << currentMinimaVect[i] << std::endl;
        // }
        
        // // Testing difference update column elimination >>
        // std::cout<<"***************************"<<std::endl;
        // prev_eliminated = matrixSupplies;
        // rowColCovered[prev_eliminated] = true;
        // currentMinimaVect[prev_eliminated] = default_diff;

        // updateDifferences<<<dimGrid, dimBlock>>>(vect, row_book, col_book, covered, matrixSupplies, matrixDemands, prev_eliminated);
        // cudaDeviceSynchronize();

        // for (size_t i = 0; i < currentMinimaVect.size(); i++) {
        //         std::cout << "currentMinimaVect[" << i << "] = " << currentMinimaVect[i] << std::endl;
        // }

        // // Testing difference update row elimination >>
        // prev_eliminated = 0;
        // rowColCovered[prev_eliminated] = true;
        // currentMinimaVect[prev_eliminated] = default_diff;
        // std::cout<<"***************************"<<std::endl;

        // updateDifferences<<<dimGrid, dimBlock>>>(vect, row_book, col_book, covered, matrixSupplies, matrixDemands, prev_eliminated);
        // cudaDeviceSynchronize();

        // for (size_t i = 0; i < currentMinimaVect.size(); i++) {
        //         std::cout << "currentMinimaVect[" << i << "] = " << currentMinimaVect[i] << std::endl;
        // }


}