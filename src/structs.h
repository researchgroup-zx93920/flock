#include <iostream>
#include <string>

#ifndef STRUCT
#define STRUCT


/*
Container for the problem construct
*/
class ProblemInstance
{

public:
    bool read_mode;
    std::string filename;
    enum my_algo {cpu_lp_solve, parallel_uv, vogel_sequencial};
    my_algo algo;
    int numDemands, numSupplies, *demands, *supplies, active_flows;
    float readTime, preprocessTime, solveTime, postprocessTime;
    double totalFlowCost;
    float *costs;
    void allocate_memory();
    ~ProblemInstance();
};

/*
Information for a flow variable
*/
struct flowInformation
{
    // from source - supply index
    // to destination - demand index
    // move this many units - qty
    int source, destination;
    float qty;
};

std::ostream& operator << (std::ostream& o, const flowInformation& x);


/*
FUTURE : Generic Algorithm class selection for solving the problem instance
Idea is to achieve a cleaner code
*/
// auto selectAlgorithm(ProblemInstance &problem)
// {
//     switch (problem.algo)
//     {
//     case 1:
//     {
//         int r = 0;
//         return r;
//     }
//     case 2:
//     {
//         std::string r = "0.1";
//         return r;
//     }
//     }
// }

#endif