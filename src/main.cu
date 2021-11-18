#include<iostream>
#include "utils.h"
#include "bfs_methods.h"

int main(){

    // const char * problem = "TransportSimplex";
    
    // todo: accept fileName as argv
    std::string fileName = "../data/TransportModel_toy.dat";
    
    int matrixDemands, matrixSupplies, * demands, * supplies;
    MatrixCell * costMatrix;
    flowInformation * flows;
    
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
    flows = (flowInformation *) calloc(matrixSupplies+matrixDemands-1, sizeof(flowInformation));

    // Finding BFS
    // Northwest Corner - sequencial
    // find_nw_corner_bfs_seq(supplies, demands, costMatrix, flows, matrixSupplies, matrixDemands);

    // Vogel's Approximation - parallel
    find_vogel_bfs_parallel(supplies, demands, costMatrix, flows, matrixSupplies, matrixDemands);

    // Modified Distribution Method (u-v method) - parallel
    // 1. with-the initial flows (as obtained above) determine dual costs for each row and column constraints

    // - Updates the reduced costs vector
    float * reduced_costs;
    reduced_costs = (float *) malloc(sizeof(float)*matrixSupplies*matrixDemands);
    find_reduced_costs(costMatrix, flows, reduced_costs, matrixSupplies, matrixDemands);
    
    // Finally >>
    printLocalDebugArray(reduced_costs, matrixSupplies, matrixDemands, "Reduced Costs");

    // In flows we have M+N-1 non-zeros giving m+n-1 equations in m+n variables
    // Solve this equation to find dual and corresponding to each form the reduced costs >>
   

}