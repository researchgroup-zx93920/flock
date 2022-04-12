#include "vogel_sequencial.h"

// Constructor
VAM_SEQ::VAM_SEQ(ProblemInstance * problem, flowInformation * flows) {
    data = problem;
	optimal_flows = flows;
    VAM_SEQ::cap_size = data->numSupplies;
	VAM_SEQ::req_size = data->numDemands;
	
    cost = (float *) malloc(sizeof(float)*cap_size*req_size);
    allotment = (float *) malloc(sizeof(float)*cap_size*req_size);
    capacity = (int *) malloc(sizeof(int)*cap_size);
    requirement = (int *) malloc(sizeof(int)*req_size);
    row_penalty = (int *) malloc(sizeof(int)*cap_size);
    column_penalty = (int *) malloc(sizeof(int)*req_size);
    
    // Initializing cost, allotment, requirement, capacity matrix
	for(int i = 0; i < cap_size; i++) {
		capacity[i] = 0.0;
		for(int j = 0; j < req_size; j++) {
            requirement[j] = 0;
			cost[i*req_size + j] = 0;
			allotment[i*req_size + j] = 0;
		}
	}
}

void VAM_SEQ::calcPenalties(){
    int *t_arr = NULL;
    // Calculating row_penalty for all rows
    for(int row = 0; row < cap_size; row++){
        // If the row has a zero capacity skip the row
        // and do not calculate penalty for that row
        if(capacity[row] == 0){
            continue;
        }
        // Calculating the number of requirements which are not 0
        int non_zero_req_count = 0;
        for(int i = 0; i < req_size; i++){
            if(requirement[i] != 0){
                non_zero_req_count++;
            }
        }
        // Creating the array and storing the costs corresponding to which
        // the requirements are non zero
        t_arr = new int[non_zero_req_count + 1];
        // Copy the costs to t_arr corresponding to which the requirement is non-zero
        for(int i = 0, index = 0; i < req_size; i++){
            if(requirement[i] != 0){
                t_arr[index] = cost[row*req_size + i];
                index++;
            }
        }
        // Sort the array in ascending order
        sort(t_arr, non_zero_req_count);
        // Store the row_penalty
        // If number of non zero requirements is not 1, then the penalty is the difference
        // of the first two minimum costs else the penalty consists of only one cell
        if(non_zero_req_count != 1){
            row_penalty[row] = abs(t_arr[1] - t_arr[2]);
        }
        else{
            row_penalty[row] = t_arr[1];
        }
        delete[] t_arr;
    }
    // Calculating column penalty all columns
    for(int col = 0; col < req_size; col++){
        // If the column has a zero requirement skip the column
        // and do not calculate penalty for that column
        if(requirement[col] == 0){
            continue;
        }
        // Calculating the number of capacities which are not 0
        int non_zero_cap_count = 0;
        for(int i = 0; i < cap_size; i++){
            if(capacity[i] != 0){
                non_zero_cap_count++;
            }
        }
        // Creating the array and storing the costs corresponding to which
        // the capacities are non zero
        t_arr = new int[non_zero_cap_count + 1];
        // Copy the costs to t_arr corresponding to which the capacity is non-zero
        for(int i = 0, index = 0; i < cap_size; i++){
            if(capacity[i] != 0){
                t_arr[index] = cost[i*req_size + col];
                index++;
            }
        }
        // Sort the array in ascending order
        sort(t_arr, non_zero_cap_count);
        // Store the column penalty
        // If number of non zero capacities is not 1, then the penalty is the difference
        // of the first two minimum costs else the penalty consists of only one cell
        if(non_zero_cap_count != 1){
            column_penalty[col] = abs(t_arr[1] - t_arr[2]);
        }
        else{
            column_penalty[col] = t_arr[1];
        }
        delete[] t_arr;
    }
}

void VAM_SEQ::sort(int arr[], int size){
    // Performing selection sort in ascending order
    for(int i = 1; i < size; i++){
        int index = i;
        for(int j = i + 1; j <= size; j++){
            if(arr[j] < arr[index]){
                index = j;
            }
        }
        int temp = arr[index];
        arr[index] = arr[i];
        arr[i] = temp;
    }
}

int VAM_SEQ::min(int num1, int num2){
	return num1 < num2 ? num1 : num2;
}

void VAM_SEQ::input(){
	
	// Taking input in cost matrix
	for(int i = 0; i < cap_size; i++){
		for(int j = 0; j < req_size; j++){
			cost[i*req_size + j] = data->costs[i*req_size + j];
		}
	}
	// Taking input in capacity matrix
	for(int i = 0; i < cap_size; i++){
		capacity[i] = data->supplies[i];
	}
	// Taking input in requirement matrix
	for(int i = 0; i < req_size; i++){
		requirement[i] = data->demands[i];
	}
}

void VAM_SEQ::displayAllotment(bool show_cap_req){
	for(int i = 0; i < cap_size; i++){
        for(int j = 0; j < req_size; j++){
                cout << allotment[i*req_size + j] << "\t";
        }
        if(show_cap_req){
            cout << capacity[i] << " (" << row_penalty[i] << ")";
        }
        cout << endl;
	}
	if(show_cap_req){
		// Printing Requirement Array
		for(int i = 0; i < req_size; i++){
            cout << requirement[i] << "\t";
		}
		cout << endl;
		// Printing the column penalties
        for(int i = 0; i < req_size; i++){
            cout << "(" << column_penalty[i] << ")" << "\t";
		}
	}
	cout << endl;
}

void VAM_SEQ::displayCost(){
	for(int i = 0; i < cap_size; i++){
		for(int j = 0; j < req_size; j++){
			cout << cost[i*req_size + j] << "\t";
		}
		cout << endl;
	}
}

void VAM_SEQ::findMaxPenalty(int &row_num, int &col_num){
    // We note that row_num or col_num
    // has a valid value withtin the range 1 - cap_size
    // for row_num and 1 - req_size for col_num
    // thus -1 is an invalid value for these variables
    // and would indicate that the column_num is invalid
    // Initially we assume that the first row encountered whose capacity
    // is not zero has the maximum penalty
    for(int row = 0; row < cap_size; row++){
        if(capacity[row] != 0){
            row_num = row;
            break;
        }
    }
    col_num = -1;
    int max_penalty = row_penalty[row_num];
    // Find the row or column that has the maximum penalty
    // Searching in row
    for(int row = 0; row < cap_size; row++){
        // We are ignoring those rows whose capacity is exhausted
        if(capacity[row] == 0){
            continue;
        }
        if(row_penalty[row] > max_penalty){
            row_num = row;
            max_penalty = row_penalty[row_num];
            col_num = -1;
        }
    }
    // Searching in column
    for(int col = 0; col < req_size; col++){
        // We are ignoring those columns whose requirement is satisfied
        if(requirement[col] == 0){
            continue;
        }
        if(column_penalty[col] > max_penalty){
            col_num = col;
            max_penalty = column_penalty[col_num];
            row_num = -1;
        }
    }
}

void VAM_SEQ::computeTransportationCost(){
    // We note that row_num or col_num
    // has a valid value withtin the range 1 - cap_size
    // for row_num and 1 - req_size for col_num
    // thus -1 is an invalid value for these variables
    // and would indicate that the column_num is invalid
    int row_of_max_penalty, col_of_max_penalty, min_cost, min_col;
    int min_allotment, min_row;
    int iteration = 0;
    while(!requirementFulfilled()){
        calcPenalties();
        findMaxPenalty(row_of_max_penalty, col_of_max_penalty);
        // If a row has the maximum penalty value
        if(row_of_max_penalty != -1 && col_of_max_penalty == -1){
            // Find the minimum cost in row_of_max_penalty and
            // give it an allotment
            // Assuming minimum cost is present in the first column encountered
            // whose requirement is not satisfied
            // We ignore the columns whose requirements is satisfied
            for(int col = 0; col < req_size; col++){
                if(requirement[col] != 0){
                    min_cost = cost[row_of_max_penalty*req_size + col];
                    min_col = col;
                    break;
                }
            }
            for(int col = 0; col < req_size; col++){
                if(requirement[col] == 0){
                    continue;
                }
                if(cost[row_of_max_penalty*req_size + col] < min_cost){
                    min_cost = cost[row_of_max_penalty*req_size + col];
                    min_col = col;
                }
            }
            // Assign it the min of the capacity and requirement
            min_allotment = min(capacity[row_of_max_penalty], requirement[min_col]);
            allotment[row_of_max_penalty*req_size + min_col] = min_allotment;
            capacity[row_of_max_penalty] -= min_allotment;
            requirement[min_col] -= min_allotment;
        }
        else {
            // Find the minimum cost in col_of_max_penalty and
            // give it an allotment
            // Assuming minimum cost is present in the first row encountered
            // whose capacity is not exhausted
            // We ignore the rows whose capacity is exhausted
            for(int row = 0; row < cap_size; row++){
                if(capacity[row] != 0){
                    min_cost = cost[row*req_size + col_of_max_penalty];
                    min_row = row;
                    break;
                }
            }
            for(int row = 0; row < cap_size; row++){
                if(capacity[row] == 0){
                    continue;
                }
                if(cost[row*req_size + col_of_max_penalty] < min_cost){
                    min_cost = cost[row*req_size + col_of_max_penalty];
                    min_row = row;
                }
            }
            // Assign it the min of the capacity and requirement
            min_allotment = min(capacity[min_row], requirement[col_of_max_penalty]);
            allotment[min_row*req_size + col_of_max_penalty] = min_allotment;
            capacity[min_row] -= min_allotment;
            requirement[col_of_max_penalty] -= min_allotment;
        }
        iteration += 1;
        // cout << "After iteration " << iteration << "---->\n";
        // displayAllotment();
        // cin.get();
        // cout << "\n\n\n\n";
    }
    int total_cost = 0;
    for(int i = 0; i < cap_size; i++){
        for(int j = 0; j < req_size; j++){
            total_cost += allotment[i*req_size + j] * cost[i*req_size + j];
        }
    }
    // cout << "Elements in allotment matrix-->" << endl;
    // displayAllotment(false);
    // cout << endl << endl;
    // cout << "Elements in cost matrix-->" << endl;
    // displayCost();
    // cout << endl << endl;
    cout << "Total transportation cost using sequencial VAM (basic) = " << total_cost << endl;
}

bool VAM_SEQ::requirementFulfilled(){
	bool req = true, cap = true;
	// Checking if all the elements of the requirement matrix is 0
	for(int i = 0; i < req_size; i++){
		if(requirement[i] != 0){
			req = false;
			break;
		}
	}
	// Checking if all the elements of the capacity matrix is 0
	for(int i = 0; i < cap_size; i++){
		if(capacity[i] != 0){
			cap = false;
			break;
		}
	}
	return req && cap;
}


void VAM_SEQ::execute() {
    
    BOOST_LOG_TRIVIAL(info) << "Initializing Sequencial Vogel's Approximation Method";
    input();
    auto start = std::chrono::high_resolution_clock::now();
    // Core >> 
    BOOST_LOG_TRIVIAL(info) << "Running VAM - sequencial (CPU)";
    computeTransportationCost();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double solution_time = duration.count();
    std::cout << "Vogel Sequencial - " << "BFS Found in : " << solution_time << " millisecs." << std::endl;
}


void VAM_SEQ::create_flows() {
    int _counter = 0;
    for(int i = 0; i < cap_size; i++){
        for(int j = 0; j < req_size; j++){
                if (allotment[i*req_size + j] > 0) {
                    flowInformation _flow = {.source = i, .destination = j, .qty = allotment[i*req_size + j]};
                    optimal_flows[_counter] = _flow;
                    _counter++;
                }
        }
	}
    // Record number of active flows
    data->active_flows = _counter;
}