// #include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#include "logger.h"
#include "utils.h"
#include "instance.h"
#include "structs.h"
#include "algos.h"

/*
  __ _            _
 / _| |          | |
| |_| | ___   ___| | __
|  _| |/ _ \ / __| |/ /
| | | | (_) | (__|   <
|_| |_|\___/ \___|_|\_\
*/

int main(int argc, char **argv)
{
  set_logger();
  BOOST_LOG_TRIVIAL(info) << "Start!";
  InputParser input(argc, argv);

  // Problem Construct
  ProblemInstance problem = ProblemInstance();

  // **************************************
  // 1. Read problem Instance >>
  // **************************************
  make_problem(input, problem);
  BOOST_LOG_TRIVIAL(info) << "Problem Dimension => Supplies: " << problem.numSupplies << "\tDemands: " << problem.numDemands;

  /*
  **************************************
  At this point the program has received the problem construct
  The problem is fed to some algorithm which does - preprocess, solve and postprocess
    All algorithm classes should have a algo.execute method that is called to do the required
  The output is finally a collection of 'flows'

  Fact:
    If
      number of supplies = M
      number of demands = N
    Then, for a degenrate case the balance could be achieved in M + N - 1 flows

  *************************************
  */

  flowInformation *flows;
  flows = (flowInformation *)malloc((problem.numSupplies + problem.numDemands - 1) * sizeof(flowInformation));

  if (problem.algo == ProblemInstance::my_algo::cpu_lp_solve)
  {
    lpModel model = lpModel(&problem, flows);
    model.execute();
    model.create_flows();
  }
  else if (problem.algo == ProblemInstance::my_algo::parallel_uv)
  {
    uvModel_parallel model = uvModel_parallel(&problem, flows);
    // model.execute();
    model.execute();
    model.create_flows();
  }

  std::cout << "Flows created successfully!" << std::endl;

  for (int i=0; i<(problem.active_flows); i++){
    std::cout<<flows[i]<<std::endl;
  }
  
  free(flows);
  std::cout << "Flows freed successfully!" << std::endl;
}