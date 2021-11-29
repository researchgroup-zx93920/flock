#include<iostream>
#include<chrono>
#include "utils.h"
#include "bfs_methods.h"
#include "uv_method.h"

int main(){
    
    // todo: accept fileName as argv
    std::string fileName = "../data/TransportModel_toy.dat";
    
    int matrixDemands, matrixSupplies, * demands, * supplies;
    MatrixCell * costMatrix;
    flowInformation * flows;
    
    // **************************************
    // 1. Read problem Instance >> 
    // **************************************

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
    
    // **************************************
    // 2. Finding BFS
    // **************************************

    // Initialize all flows to zero >>
    // Flow information stores flow in a COO sparse matrix form
    flows = (flowInformation *) calloc(matrixSupplies+matrixDemands-1, sizeof(flowInformation));

    // Northwest Corner - sequencial
    // find_nw_corner_bfs_seq(supplies, demands, costMatrix, flows, matrixSupplies, matrixDemands);

    // Vogel's Approximation - parallel
    auto start = std::chrono::high_resolution_clock::now();
    find_vogel_bfs_parallel(supplies, demands, costMatrix, flows, matrixSupplies, matrixDemands);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	double solution_time = duration.count();
    std::cout<<"Vogel BFS Found in : "<<solution_time<<" secs."<<std::endl;

    // **************************************
    // 3. Modified Distribution Method (u-v method) - parallel
    // **************************************

    // 1. with-the initial flows (as obtained above) determine dual costs for each row and column constraints, 
    // Todo: there's a memcpy operation that can be eleminated between previous and next step! - Mohit to investigate!

    // Update the reduced costs vector
    float * reduced_costs;

    for (int i=0; i<1; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        reduced_costs = (float *) malloc(sizeof(float)*matrixSupplies*matrixDemands);
        find_reduced_costs(costMatrix, flows, reduced_costs, matrixSupplies, matrixDemands);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	    double solution_time = duration.count();
        std::cout<<"Iteration : "<<i+1<<" of finding reduced costs completed in : "<<solution_time<<" secs."<<std::endl;
    }
    


    // Finally >>
    printLocalDebugArray(reduced_costs, matrixSupplies, matrixDemands, "Reduced Costs");

    // In flows we have M+N-1 non-zeros giving m+n-1 equations in m+n variables
    // Solve this equation to find dual and corresponding to each form the reduced costs >>
   

}