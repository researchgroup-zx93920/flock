#include "bfs_methods.h"

void find_nw_corner_bfs_seq(int * supplies, int * demands, float * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands) {
    
    std::cout<<"Running Northwest Corner Seq BFS Method"<<std::endl;

    // Step 1 :: Jumpt to NW corner >>
    int current_row_number = 0;
    int current_col_number = 0;
    int current_demand = demands[current_row_number];
    int current_supply = supplies[current_col_number];

    // Allocate flow equal to minimum of demand and supply and update the buffer accordingly >>
    while (current_row_number < matrixSupplies && current_col_number < matrixDemands) {
        
        // std::cout<<"Current Supply Index : "<<current_row_number<<std::endl;
        // std::cout<<"Current Supply : "<<current_supply<<std::endl;
        // std::cout<<"Current Demand Index : "<<current_col_number<<std::endl;
        // std::cout<<"Current Demand : "<<current_demand<<std::endl;
        
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

void find_vogel_bfs_seq(int * supplies, int * demands, float * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands) {
    
    std::cout<<"Vogel's Approximation seq BFS Method"<<std::endl;
    // Book-keeping stuff >>
    int coveredRows = 0 , coveredColumns = 0;
    int *residual_supply = (int *) malloc(matrixSupplies*sizeof(int));
    std::memcpy(residual_supply, supplies, matrixSupplies*sizeof(int));

    int *residual_demand = (int *) malloc(matrixDemands*sizeof(int));
    std::memcpy(residual_demand, demands, matrixDemands*sizeof(int));

    int * rowCovered = (int *) calloc(matrixSupplies, sizeof(int));
    int * colCovered = (int *) calloc(matrixDemands, sizeof(int));    
    int * differences = (int *) calloc(matrixSupplies + matrixDemands, sizeof(int));
    std::cout<<"\tCreated all book-keeping structs"<<std::endl;

    std::cout<<"\tIterating Vogel's Heuristic"<<std::endl;
    while (coveredRows + coveredColumns < matrixDemands+matrixSupplies-1) {
        // std::cout<<"Iteration - "<<coveredColumns+coveredRows<<std::endl;
        float temp1, temp2, tempDiff;
        float costTemp;
        int i_tempDiff, i_minCost;

        // Calculate row differences >> 
        for (int i=0; i< matrixSupplies; i++){
            if (rowCovered[i] == 0) {
                temp1 = INT_MAX;
                temp2 = INT_MAX;
                for (int j=0; j< matrixDemands; j++) {
                    // Only look at columns not covered >> 
                    if (colCovered[j] == 0) {
                        float entry = costMatrix[i*matrixDemands + j];
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

        // Calculate col differences >> 
        for (int j=0; j< matrixDemands; j++){
            if (colCovered[j] == 0) {
                temp1 = INT_MAX;
                temp2 = INT_MAX;
                // Only look at rows not covered >>
                for (int i=0; i< matrixSupplies; i++) {
                    if (rowCovered[i] == 0) {
                        float entry = costMatrix[i*matrixDemands + j];
                        if (entry <= temp1) {
                            temp2 = temp1;
                            temp1 = entry;
                        }
                        else if (entry <= temp2) {
                            temp2 = entry;
                        }
                    }
                }
                differences[matrixSupplies + j] = temp2 - temp1;
            }
            else {
                differences[matrixSupplies + j] = INT_MIN;
            }
        }

        // Determine the maximum of differences - (Reduction)
        tempDiff = INT_MIN;
        i_tempDiff = -1;
        for (int i=0; i < matrixSupplies + matrixDemands; i++) {
            if (differences[i] > tempDiff) {
                // tie broken by first seen
                tempDiff = differences[i];
                i_tempDiff = i;
            }
        }
        
        // Check if row or col difference and determine correspinding min cost - Another Reduction
        // Update flow accordingly and increment coveredRows/Columns and row/colCovered - Minor Update
        // Now we have Basic row and col
        // Assign flow based on availability 
        if (i_tempDiff >= matrixSupplies) {
            // This is a col difference
            i_tempDiff -= matrixSupplies;
            // In this column index find the min cost
            costTemp = INT_MAX;
            for (int i=0; i<matrixSupplies; i++) {
                float entry = costMatrix[i*matrixDemands + i_tempDiff];
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
                flows[i_minCost*matrixDemands + i_tempDiff] = residual_supply[i_minCost];
                residual_demand[i_tempDiff] -= residual_supply[i_minCost];
                rowCovered[i_minCost] = 1;
                coveredRows += 1;
            }
            else {
                flows[i_minCost*matrixDemands + i_tempDiff] = residual_demand[i_tempDiff];
                residual_supply[i_minCost] -= residual_demand[i_tempDiff];
                colCovered[i_tempDiff] = 1;
                coveredColumns += 1;
            }
        }
        else {
            // Then this is a row difference
            // In this row find the min cost
            costTemp = INT_MAX;
            
            for (int j=0; j<matrixDemands; j++) {
                float entry = costMatrix[i_tempDiff*matrixDemands + j];
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
                flows[i_tempDiff*matrixDemands + i_minCost] = residual_supply[i_tempDiff];
                residual_demand[i_minCost] -= residual_supply[i_tempDiff];
                rowCovered[i_tempDiff] = 1;
                coveredRows += 1;
            }
            else {
                flows[i_tempDiff*matrixDemands + i_minCost] = residual_demand[i_minCost];
                residual_supply[i_tempDiff] -= residual_demand[i_minCost];
                colCovered[i_minCost] = 1;
                coveredColumns += 1;
            }  
        }
        // printLocalDebugArray(flows, matrixSupplies, matrixDemands, "Flows");
    }
    std::cout<<"\tVogel complete!"<<std::endl;
}


/* 
    Find least two elements in a row/column of a matrix
    flatMatrix2D : ptr to a flattened 2D matrix
    diff: container for storing differences
    orientation: 0 for rows, 1 for columns >> bug: doesn't check for integrity here
    vectorIndex: index or row/col
    width: matrixWidth
    height: matrixHeight
*/
__global__ void find_least_two_with_indexes(float * flatMatrix2D, vogelDifference * diff, int orientation, 
    int * rowCovered, int *colCovered,
    int width, int height, int offset) {
    
    int indx = blockDim.x*blockIdx.x + threadIdx.x;
    
    int iterations = orientation == 0?width:height; 
    // If computing row differences - for indx'th row - iterate over all cols, else iterate over all rows 
    
    int max_indx = orientation == 0?height:width;
    // If computing row differences - for max indx is height, else if col diff then max index is width

    bool skip_flag = orientation == 0?rowCovered[indx]:colCovered[indx];
    // Skip flag tells if a row-col is to be ignored

    int (* fx) (int a, int b, int c) = orientation==0?&rowIndxInFlat:&colIndxInFlat;
    // fx is an indexing method for flattened arrays =>
    //  if computing row differences, indx is the row ID, iteration over columns
    //  if computing col differences, indx is the col ID, iteration over rows

    float temp1, temp2;
    int itemp1, itemp2;
    
    temp1 = INT_MIN;
    temp2 = INT_MIN;
    
    if (indx < max_indx && !skip_flag) {

        for (int i=0; i< iterations; i++) {
            // Only look at columns not covered >>     
            float entry = flatMatrix2D[fx(indx, width, i)];
            if (entry <= temp1) {
                temp2 = temp1;
                itemp2 = itemp1;
                temp1 = entry;
                itemp1 = i;
            }
            else if (entry <= temp2) {
                temp2 = entry;
                itemp2 = i;
            }
        }
        vogelDifference this_diff = {.indx = offset+indx, .diff = temp2-temp1, .ileast_1 = itemp1, .ileast_2 = itemp2};
        // .least_1 = temp1, .least_2 = temp2,
        diff[offset+indx] = this_diff;
    }
}


__global__ void consumeDemandSupply(int * d_supplies, int * d_demands, int * rowCovered, int * columnCovered, 
        int * f, int row_indx, int col_indx) {
        
        int sup = d_supplies[row_indx];
        int dem = d_demands[col_indx];
        if (sup >= dem) {
            // Demand point consumed the supply and got eliminated
            f[0] = dem; // Message to host
            f[1] = 1; // Message to host
            d_supplies[row_indx] = sup - dem;
            columnCovered[col_indx] = 1;
        }
        
        else {
            // Supply point consumed the demand and got eliminated
            f[0] = sup; // Message to host
            f[1] = 0; // Message to host
            d_demands[col_indx] = dem - sup;
            rowCovered[row_indx] = 1;
        }
}

/* 
Step 0 : Initialize
    0.1 : Allocate and copy Structures to GPU global memory
        Allocate : (supplies, demands, costMatrix)
        Copy : (supplies, demands, costMatrix)
        Flow Assignment for BFS are O(1) - Do them on the host
>> Initialize while loop
Step 1 : Find Row and Columns Differences
    1.1 Launch Kernel - 1D grid (Row differences) - async
    1.2 Launch Kernel - 1D grid (Columns Differences) - async
Step 2: Reduce Max of row and col differences
Step 3: Find 
cudaDeviceSynchronize();
*/
__host__ void find_vogel_bfs_parallel(int * supplies, int * demands, MatrixCell * costMatrix, 
        int * flows, int matrixSupplies, int matrixDemands) {
        
        // Step 0 :
        std::cout<<"Vogel Kernel - Step 0"<<std::endl; 

        int number_of_blocks = ceil(1.0*matrixSupplies/blockSize);

        // thrust::device_vector<int> device_supplies(supplies, supplies+matrixSupplies);
        // thrust::device_vector<int> device_demands(demands, demands+matrixDemands);
        // thrust::device_vector<float> device_costMatrix(costMatrix, costMatrix + matrixSupplies*matrixDemands);

        // // Book-keeping Structures on device and host >>
        // // Kernel-1 : sort-cost matrix rows and store pointers to sorted order
        // thrust::device_vector<int> device_costMatrixRowBook(matrixSupplies*matrixDemands); // This will be used as key
        // thrust::device_vector<int> device_costMatrixColBook(matrixSupplies*matrixDemands); // This will be used as key
        
        // thrust::fill(device_costMatrixColBook.begin(),device_costMatrixColBook.end(), device_costMatrix.begin());
        // thrust::fill(device_costMatrixRowBook.begin(), device_costMatrixRowBook.end(), device_costMatrix.begin());

        // int my_div_start = 0;
        // int my_div(){ // for RowSortSegments
        //     return (my_div_start++)/matrixDemands;
        // }

        // int my_

        // thrust::fill()

        
        // cudaMalloc((void **) &rowCovered, matrixSupplies*sizeof(int));
        // cudaMemset(rowCovered, 0, matrixSupplies*sizeof(int));

        // cudaMalloc((void **) &columnCovered, matrixDemands*sizeof(int));
        // cudaMemset(columnCovered, 0, matrixDemands*sizeof(int));

        // h_f = (int *) malloc(sizeof(int)*2);
        // cudaMalloc((void **) &d_f, sizeof(int)*2);

        // // Step 1 : 
        
        // // Preparation for Step-1 >>
        // // IDEA : use cudaStream_t for all memcopies, add Error Catcher, 
        // //      avoid_recompute with smarter lookups, on top of lookups

        // dim3 blockD(blockSize, 1, 1);
        // dim3 gridD(number_of_blocks, 1, 1);

        // for (int iter=0; iter<matrixSupplies+matrixDemands-1; iter++) {

        //     std::cout<<"Vogel Kernel - Step 1 : Row Differences"<<std::endl; 
        //     find_least_two_with_indexes<<<gridD, blockD>>>(d_costMatrix, d_diff, 0, rowCovered, columnCovered, matrixDemands, matrixSupplies, 0);
        //     std::cout<<"Vogel Kernel - Step 1 : Col Differences"<<std::endl;
        //     find_least_two_with_indexes<<<gridD, blockD>>>(d_costMatrix, d_diff, 1, rowCovered, columnCovered, matrixDemands, matrixSupplies, matrixSupplies);
            
        //     cudaDeviceSynchronize();

        //     // Find max of Row and Col Differences >>
        //     // d_diff is still on device => Directly Call the reduction kernel
        //     find_max<<<blockD, gridD>>>(d_diff, d_diff_buffer, matrixSupplies+ matrixDemands);
        //     cudaDeviceSynchronize();

        //     cudaMemcpy(h_diff_buffer, d_diff_buffer, sizeof(vogelDifference)*number_of_blocks, cudaMemcpyDeviceToHost);
        //     // Now Reduce a small segment on device for h_diff >>
        //     // Recall - We're still finding max of differences but now from a very small set
        //     tempDiff = h_diff_buffer[0];
        //     for (int i=1; i<number_of_blocks; i++) {
        //         if (h_diff_buffer[i].diff >= tempDiff.diff) {
        //             tempDiff = h_diff_buffer[i];
        //         }
        //     }
            
        //     // vogelDifference d = tempDiff;
        //     // std::cout<<"Max Diff : "<<d.diff<<std::endl;
        //     // std::cout<<"Max Diff : indx "<<d.indx<<std::endl;
        //     // std::cout<<"Max Diff : leastCost "<<d.ileast_1<<std::endl;
            
        //     // Identify this is a row difference or col difference
        //     // Now the flow assignment cell is - tempDiff.idx and tempDiff.ileast_1 

        //     if (tempDiff.indx > matrixSupplies) {
        //         // This is a col difference
        //         row_indx = tempDiff.ileast_1;
        //         col_indx = tempDiff.indx - matrixSupplies;
        //     }
        //     else {
        //         col_indx = tempDiff.ileast_1;
        //         row_indx = tempDiff.indx;
        //     }
            
        //     // Assign Supply Demand smartly with kernel - Direcly update device copy of demands and 
        //     // supplies to determine residual, use 1 thread for this and update flow in host
            
        //     // Get some way around this - do we have a device struc messaging protocol >> 
        //     consumeDemandSupply<<<1, 1>>>(d_supplies, d_demands, rowCovered, columnCovered, d_f, row_indx, col_indx);
        //     cudaDeviceSynchronize();

        //     cudaMemcpy(h_f, d_f, sizeof(int)*2, cudaMemcpyHostToDevice);
        //     flows[row_indx*matrixSupplies+col_indx] = h_f[0];

        // }
        
        // printLocalDebugArray(flows, matrixSupplies, matrixDemands, "flows");

        // cudaFree(d_supplies);
        // cudaFree(d_demands);
        // cudaFree(d_costMatrix);
        // cudaFree(d_diff);
        // cudaFree(rowCovered);
        // cudaFree(columnCovered);

    }



// #include <thrust/device_vector.h>
// #include <thrust/device_ptr.h>
// #include <thrust/host_vector.h>
// #include <thrust/sort.h>
// #include <thrust/execution_policy.h>
// #include <thrust/generate.h>
// #include <thrust/equal.h>
// #include <thrust/sequence.h>
// #include <thrust/for_each.h>
// #include <iostream>
// #include <stdlib.h>

// #define NSORTS 16000
// #define DSIZE 1000

// int my_mod_start = 0;
// int my_mod(){
//   return (my_mod_start++)/DSIZE;
// }


// bool validate(thrust::device_vector<int> &d1, thrust::device_vector<int> &d2){
//   return thrust::equal(d1.begin(), d1.end(), d2.begin());
// }

// struct sort_functor
// {
//   thrust::device_ptr<int> data;
//   int dsize;
//   __host__ __device__
//   void operator()(int start_idx)
//   {
//     thrust::sort(thrust::device, data+(dsize*start_idx), data+(dsize*(start_idx+1)));
//   }
// };



// #include <time.h>
// #include <sys/time.h>
// #define USECPSEC 1000000ULL

// unsigned long long dtime_usec(unsigned long long start) {

//   timeval tv;
//   gettimeofday(&tv, 0);
//   return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;

// }

// int main(){
  
//   cudaDeviceSetLimit(cudaLimitMallocHeapSize, (16*DSIZE*NSORTS));
//   thrust::host_vector<int> h_data(DSIZE*NSORTS);
//   thrust::generate(h_data.begin(), h_data.end(), rand);
//   thrust::device_vector<int> d_data = h_data;

//   //vectorized sort
//   thrust::device_vector<int> d_result2 = d_data;
//   thrust::host_vector<int> h_segments(DSIZE*NSORTS);
//   thrust::generate(h_segments.begin(), h_segments.end(), my_mod);
//   thrust::device_vector<int> d_segments = h_segments;
//   mytime = dtime_usec(0);
//   thrust::stable_sort_by_key(d_result2.begin(), d_result2.end(), d_segments.begin());
//   thrust::stable_sort_by_key(d_segments.begin(), d_segments.end(), d_result2.begin());
//   cudaDeviceSynchronize();
//   mytime = dtime_usec(mytime);
//   std::cout << "vectorized time: " << mytime/(float)USECPSEC << "s" << std::endl;
//   if (!validate(d_result1, d_result2)) std::cout << "mismatch 1!" << std::endl;
  
// }