#include "structs.h"

void ProblemInstance::allocate_memory(void)
{
    supplies = (int *)malloc(sizeof(int) * numSupplies);
    demands = (int *)malloc(sizeof(int) * numDemands);
    costs = (float *)malloc(sizeof(float) * numSupplies * numDemands);
}

ProblemInstance::~ProblemInstance(void)
{
    free(supplies);
    free(demands);
    free(costs);
}

std::ostream& operator << (std::ostream& o, const flowInformation& x) {
    o << "Flow - " << x.qty <<" units from source/rowNo: "<<x.source <<" to destination/colNo: "<<x.destination;
    return o;
}