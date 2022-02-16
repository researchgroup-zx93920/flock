#include <iostream>
#include <cstring>
#include <map>

#include "../structs.h"
#include "parallel_structs.h"

void find_nw_corner_bfs_seq(int *supplies, int *demands, MatrixCell *costMatrix,
                            flowInformation *flows, std::map<std::pair<int, int>, int> &flow_indexes,
                            int matrixSupplies, int matrixDemands);