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
        return (i.cost < j.cost);
    }
};


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
        // Row Bookeeping
        // *********************************

        thrust::device_vector<MatrixCell> device_costMatrixRowBook(matrixSupplies*matrixDemands);  
        thrust::copy(thrust::device, device_costMatrix.begin(),device_costMatrix.end(), device_costMatrixRowBook.begin());
        
        // auto rowgen = [=]  __device__ (MatrixCell x) {return x.row;};  
        // -> Alternative way using such lambda exp (compile with --expt-extended-lambda -std=c++11) 
        
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
        // Column Bookeeping
        // *********************************

        thrust::device_vector<MatrixCell> device_costMatrixColBook(matrixSupplies*matrixDemands);  
        thrust::copy(thrust::device, device_costMatrix.begin(),device_costMatrix.end(), device_costMatrixColBook.begin());
        
        // auto rowgen = [=]  __device__ (MatrixCell x) {return x.row;};  
        // -> Alternative way using such lambda exp (compile with --expt-extended-lambda -std=c++11) 
        
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

        /*
        Todo: 
        Row-current Minima pointer [[0,1],[0,1],[0,1] . . .]
        Col-current Minima pointer [[0,1],[0,1],[0,1] . . .]
        */

        // for (size_t i = 0; i < device_costMatrixColBook.size(); i++){
        //         std::cout << "device_costMatrixColBook[" << i << "] = " << device_costMatrixColBook[i] << std::endl;
        // }

}