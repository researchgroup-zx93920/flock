#include<iostream>
#include "utils.h"

#ifndef BFS
#define BFS
void find_nw_corner_bfs_seq(int * supplies, int * demands, double * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands);

void find_vogel_bfs_seq(int * supplies, int * demands, double * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands);

void find_russel_bfs_seq(int * supplies, int * demands, double * costMatrix, int * flows, 
        int matrixSupplies, int matrixDemands);
#endif