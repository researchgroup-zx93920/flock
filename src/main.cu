#include<iostream>
#include "utils.h"
#include "bfs_methods.h"

int main(){

    // const char * problem = "TransportSimplex";
    
    // todo: accept fileName as argv
    std::string fileName = "../data/TransportModel_toy.dat";
    
    int matrixDemands, matrixSupplies, * demands, * supplies, *flows;
    MatrixCell * costMatrix;
    
    // Read problem Instance >> 
    std::cout<<"File Name : "<<fileName<<std::endl;
    readSize(matrixDemands, matrixSupplies, fileName);
    
    // Allocate Appropriate input resources and read matrices/demand supply vectors >>
    std::cout<<"Matrix Supplies : "<<matrixSupplies<<std::endl;
    std::cout<<"Matrix Demands : "<<matrixDemands<<std::endl;
    demands = (int *)malloc(sizeof(int)*matrixDemands);
    supplies = (int *)malloc(sizeof(int)*matrixSupplies);
    costMatrix = (MatrixCell *) malloc(sizeof(MatrixCell)*matrixSupplies*matrixDemands);
    readFile(supplies, demands, costMatrix, fileName);

    // printLocalDebugArray(supplies, 1, matrixSupplies, "Supplies");
    // printLocalDebugArray(demands, 1, matrixDemands, "Demands");
    // printLocalDebugArray(costMatrix, matrixSupplies, matrixDemands, "costs");
    
    // Initialize all flows to zero >>
    flows = (int *) calloc(matrixSupplies*matrixDemands, sizeof(int));

    // Finding BFS
    // Northwest Corner - sequencial
    // find_nw_corner_bfs_seq(supplies, demands, costMatrix, flows, matrixSupplies, matrixDemands);

    // Vogel's Approximation - sequencial
    // find_vogel_bfs_seq(supplies, demands, costMatrix, flows, matrixSupplies, matrixDemands);
    find_vogel_bfs_parallel(supplies, demands, costMatrix, flows, matrixSupplies, matrixDemands);
   
    // Modified Distribution Methods - Parallel >>
    // printLocalDebugArray(flows, matrixSupplies, matrixDemands, "Flows");

    // Inflows we have M+N-1 non-zeros giving m+n-1 equations in m+n variables
    // Solve this equation to find dual and corresponding to each form the reduced costs >>
   

}