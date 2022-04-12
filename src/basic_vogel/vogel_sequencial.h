// ABOUT THIS: Mohit Stiched this std. available code in his own setitng 
// Credits: https://gist.github.com/tigerjoy/1009c8526bb8bffbcad6c9c05c53c4bf
/*************************************************************************************
*   This program uses 1 indexed array, meaning that the elements are stored in the   *
*   array starting from position 1 and all the methods which work with arrays        *
*   follow the above convention                                                      *
**************************************************************************************/
// PENDING COMPLETION >>

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <chrono>

#include "../structs.h"
#include "../logger.h"

using namespace std;

class VAM_SEQ{
	float * cost, * allotment;
    int * capacity, * requirement, * row_penalty, * column_penalty;
	int cap_size, req_size;
    ProblemInstance * data;
    flowInformation * optimal_flows;
	public:
		VAM_SEQ(ProblemInstance * problem, flowInformation * flows);
		void input();
		void displayAllotment(bool show_cap_req = true);
		void displayCost();
		void computeTransportationCost();
		void findMaxPenalty(int &row_num, int &column_num);
		bool requirementFulfilled();
		void sort(int arr[], int size);
		void calcPenalties();
		int min(int num1, int num2);
        void execute();
        void create_flows();
};