#include <gurobi_c++.h>
#include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include <chrono>

#include "../structs.h"
#include "../logger.h"

// PARAMETERS
#define GRB_LOGS 1

/*
0 : Silenced gurobi
1 : Verbose gurobi
*/

#define RELAXED_X_VAR 0

/*
0 : Continous nature of x-variables
1 : Integer nature of x-variables
*/

#define DISABLE_AUTOGRB 1

/*
0 : Use gurobi's auto-configuration on parameter choice
1 : Turn off barrier methods and enforce dual-simplex
*/

#define GRB_TIMEOUT 3600

// END OF PARAMETERS 

class lpModel
{

public:
    GRBModel *model;
    ProblemInstance * data;
    flowInformation * optimal_flows;
    std::map<int, GRBVar> x_ij; // Flow variable - represeting flow from supply - i to demand j

    void execute();
    void get_dual_costs();
    void create_flows();

    lpModel(ProblemInstance * problem, flowInformation * flows);
    ~lpModel();

private:
    bool solved;
    void create_variables();
    void add_constraints();
    void add_objective();
    void solve();
};