#include "IBFS_nwc.h"

__host__ void find_nw_corner_bfs_seq(int * supplies, int * demands, MatrixCell * costMatrix, flowInformation * flows, 
        std::map<std::pair<int,int>, int> &flow_indexes, int matrixSupplies, int matrixDemands){

        std::cout<<"Running Northwest Corner Seq BFS Method"<<std::endl;

        // Step 1 :: Jumpt to NW corner >>
        int current_row_number = 0;
        int current_col_number = 0;
        int current_demand = demands[current_row_number];
        int current_supply = supplies[current_col_number];
        int counter = 0;
        // Allocate flow equal to minimum of demand and supply and update the buffer accordingly >>
        while (current_row_number < matrixSupplies && current_col_number < matrixDemands) {

                if (current_demand >= current_supply) {
                        flowInformation _this_flow = {.source = current_row_number, .destination = current_col_number, .qty = std::max(1.0f*current_supply, epsilon)};
                        flows[counter] = _this_flow;
                        flow_indexes.insert(std::make_pair(std::make_pair(current_row_number, current_col_number), counter));
                        current_demand = current_demand -  current_supply;
                        current_row_number++;
                        current_supply = supplies[current_row_number];
                }
                else {
                        flowInformation _this_flow = {.source = current_row_number, .destination = current_col_number, .qty = std::max(1.0f*current_demand, epsilon)};
                        flows[counter] = _this_flow;
                        flow_indexes.insert(std::make_pair(std::make_pair(current_row_number, current_col_number), counter));
                        current_supply = current_supply -  current_demand;
                        current_col_number++;
                        current_demand = demands[current_col_number];
                }
                counter++;
        }
        std::cout<<"Feasible BFS Generated!"<<std::endl;
}