#include "bfs_methods.h"

void find_nw_corner_bfs_seq(int * supplies, int * demands, double * costMatrix, int * flows, 
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

void find_vogel_bfs_seq(int * supplies, int * demands, double * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands) {
    
    std::cout<<"Vogel's Approximation seq BFS Method"<<std::endl;
    // Book-keeping stuff >>
    int coveredRows = 0 , coveredColumns = 0;
    int *residual_supply = (int *) malloc(matrixSupplies*sizeof(int));
    memcpy(residual_supply, supplies, matrixSupplies*sizeof(int));

    int *residual_demand = (int *) malloc(matrixDemands*sizeof(int));
    memcpy(residual_demand, demands, matrixDemands*sizeof(int));

    int * rowCovered = (int *) calloc(matrixSupplies, sizeof(int));
    int * colCovered = (int *) calloc(matrixDemands, sizeof(int));    
    int * differences = (int *) calloc(matrixSupplies + matrixDemands, sizeof(int));
    std::cout<<"\tCreated all book-keeping structs"<<std::endl;

    std::cout<<"\tIterating Vogel's Heuristic"<<std::endl;
    while (coveredRows + coveredColumns < matrixDemands+matrixSupplies-1) {
        // std::cout<<"Iteration - "<<coveredColumns+coveredRows<<std::endl;
        double temp1, temp2, tempDiff;
        double costTemp;
        int i_tempDiff, i_minCost;

        // Calculate row differences >> 
        for (int i=0; i< matrixSupplies; i++){
            if (rowCovered[i] == 0) {
                temp1 = INT_FAST16_MAX;
                temp2 = INT_FAST16_MAX;
                for (int j=0; j< matrixDemands; j++) {
                    // Only look at columns not covered >> 
                    if (colCovered[j] == 0) {
                        double entry = costMatrix[i*matrixDemands + j];
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
                differences[i] = INT_FAST16_MIN;
            }
        }

        // Calculate col differences >> 
        for (int j=0; j< matrixDemands; j++){
            if (colCovered[j] == 0) {
                temp1 = INT_FAST16_MAX;
                temp2 = INT_FAST16_MAX;
                // Only look at rows not covered >>
                for (int i=0; i< matrixSupplies; i++) {
                    if (rowCovered[i] == 0) {
                        double entry = costMatrix[i*matrixDemands + j];
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
                differences[matrixSupplies + j] = INT_FAST16_MIN;
            }
        }

        // Determine the maximum of differences - (Reduction)
        tempDiff = INT_FAST16_MIN;
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
            costTemp = INT_FAST16_MAX;
            for (int i=0; i<matrixSupplies; i++) {
                double entry = costMatrix[i*matrixDemands + i_tempDiff];
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
            costTemp = INT_FAST16_MAX;
            
            for (int j=0; j<matrixDemands; j++) {
                double entry = costMatrix[i_tempDiff*matrixDemands + j];
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
