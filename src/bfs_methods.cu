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

struct vogelDifference {
        float diff;
        int idx, ileast_1, ileast_2;
        // idx stores itselves index in difference array
        // ileast_1 and ileast2 are indexes of min-2 values
        // least_1,least_2,
};

__global__ void initializeDifferencesVector(vogelDifference * minima, int size) {
        
        int indx = blockIdx.x*blockDim.x + threadIdx.x;
        
        if (indx < size) {
                vogelDifference d = {.idx = indx, .ileast_1 = 0, .ileast_2 = 1, .diff = 0.0};
                minima[indx] = d;
        }
}

__global__ updateDifferences(vogelDifferences *diff, MatrixCell *row_book, MatrixCell *col_book, int n_rows, int n_cols, int prev_eleminated) {
        
        int indx = blockIdx.x*blockDim.x + threadIdx.x;
        if (indx < n_rows && prev_eleminated > n_rows) {
                /* Then a column was eliminated - 
                In this case row minima are updated and column minima are maintained
                        1. RowMinima exist at indx < n_rows in diff
                        2. Elimiated col indx = prev_eliminated - n_rows
                        3. Now see if the corresponding diff has either ileast-1 or ileast-2 equal to elimiated col indx
                        4. Increment
                */
                int rejectColIdx = prev_eleminated - n_rows;
                // In row_book index from [indx*n_cols] upto [(indx+1)*n_cols] is the sorted order of row costs 
                MatrixCell least1 = row_book[indx*n_cols + diff[indx].ileast_1]; // Minimum
                MatrixCell least2 = row_book[indx*n_cols + diff[indx].ileast_2]; // Second Minimum

        }

        else if (indx < n_rows + n_cols && prev_eleminated < n_rows) {
                /* Then a row was eliminated - 
                In this case col minima are updated and row minima are maintained
                        1. ColMinima exist at indx > n_rows in diff
                        2. Elimiated row indx = prev_eliminated
                        3. Now see if the corresponding diff has either ileast-1 or ileast-2 equal to elimiated col indx
                        4. Increment
                */
        }


}

/*
Step 0: For each of the rows and columns - Determine the sorted order of col indexes and row indexes respectively
*/
__host__ void find_vogel_bfs_parallel(int * supplies, int * demands, MatrixCell * costMatrix, 
        int * flows, int matrixSupplies, int matrixDemands) {
        
        // Step 0 :
        std::cout<<"Vogel Kernel - Step 0"<<std::endl; 

        int number_of_blocks = ceil(1.0*matrixSupplies/blockSize);

        thrust::device_vector<int> device_supplies(supplies, supplies+matrixSupplies);
        thrust::device_vector<int> device_demands(demands, demands+matrixDemands);
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

        // Freeup Some Memory
        device_rowSegments.clear();
        device_colSegments.clear();
        
        /* 
        Illustration:

        - Row-current Minima pointer [[0,1],[0,1],[0,1] . . .] 
        - Col-current Minima pointer [[0,1],[0,1],[0,1] . . .]
        
        vogelDifference vector was created to accomodate the above need
        Further currentMinimaVect concat's both
        */
        // Vector of indexes of row and col minima in current iteration of vogel - 
        thrust::device_vector<vogelDifference> currentMinimaVect(matrixSupplies+matrixDemands);
        thrust::device_vector<bool> rowColCovered(matrixSupplies+matrixDemands);
        thrust::fill(rowColCovered.begin(), rowColCovered.end(), false);

        vogelDifference * vect = thrust::raw_pointer_cast(currentMinimaVect.data());
        
        dim3 dimBlock(blockSize, 1, 1);
        dim3 dimGrid(ceil(1.0*currentMinimaVect.size()/blockSize),1,1);
        initializeDifferencesVector<<<dimGrid, dimBlock>>>(vect, currentMinimaVect.size());


        // for (size_t i = 0; i < device_costMatrixColBook.size(); i++){
        //         std::cout << "device_costMatrixColBook[" << i << "] = " << device_costMatrixColBook[i] << std::endl;
        // }

}