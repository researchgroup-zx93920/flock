#include "logger.h"
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

  add_log_msg("info", "Start!");
  InputParser input(argc, argv);
  std::stringstream log_msg;

  // Problem Construct
  ProblemInstance problem = ProblemInstance();

  // **************************************
  // 1. Read problem Instance >>
  // **************************************
  make_problem(input, problem);
  log_msg << "Problem Dimension => Supplies: " << problem.numSupplies << "\tDemands: " << problem.numDemands;
  add_log_msg("info", log_msg.str());
  log_msg.str("");

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

    if (MAKE_WITH_GUROBI != 0) {

      lpModel model = lpModel(&problem, flows);
      model.execute();
      model.create_flows();
      log_msg << ">>>> BASIC STATISTICS | Objective: " << model.objVal << " | Iterations: "<<model.totalIterations<<" | Time: "<<model.totalSolveTime \
      <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
      add_log_msg("info", log_msg.str());
      log_msg.str("");
    }

    else {
      add_log_msg("error", "*****************************************************");
      add_log_msg("error", "Build the executable with GUROBI option, Ensure a valid gurobi solver exists!");
      add_log_msg("error", "Refer to docs for more details ... ");
      add_log_msg("error", "*****************************************************");
      exit(-2);
    }

  }
  else if (problem.algo == ProblemInstance::my_algo::parallel_uv)
  {
    uvModel_parallel model = uvModel_parallel(&problem, flows);
    model.execute();
    model.create_flows();
    log_msg << ">>>> BASIC STATISTICS | Objective: "<<model.objVal<<" | Iterations: "<<model.totalIterations<<" | Time: "<<model.totalSolveTime \
    <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
    add_log_msg("info", log_msg.str());
    log_msg.str("");

    log_msg << ">>>> LVL1 STATISTICS | Total Time: "<<model.totalSolveTime \
    <<" | UV Time: "<<round(model.uv_time/1000)                                          \
    <<" | R Time: "<<round(model.reduced_cost_time/1000)                                 \
    <<" | PIVOT Time: "<<round(model.pivot_time/1000)                                    \
    <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
    add_log_msg("info", log_msg.str());
    log_msg.str("");

    log_msg << ">>>> LVL2 PIVOT STATISTICS | Total Pivot Time: "<<round(model.pivot_time/1000)  \
    <<" | CYCLE Time: "<<round(model.cycle_discovery_time/1000)                                      \
    <<" | RESOLVE Time: "<<round(model.resolve_time/1000)                              \
    <<" | ADJUST Time: "<<round(model.adjustment_time/1000)                            \
    <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
    add_log_msg("info", log_msg.str());
    log_msg.str("");
  }
  else if (problem.algo == ProblemInstance::my_algo::parallel_ss)
  {
    ssModel_parallel model = ssModel_parallel(&problem, flows);
    model.execute();
    model.create_flows();
    log_msg << ">>>> BASIC STATISTICS | Objective: "<<model.objVal<<" | Iterations: "<<model.totalIterations<<" | Time: "<<model.totalSolveTime \
    <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
    add_log_msg("info", log_msg.str());
    log_msg.str("");

    log_msg << ">>>> LVL1 STATISTICS | Total Time: "<<model.totalSolveTime \
    <<" | PIVOT Time: "<<round(model.pivot_time/1000)                                    \
    <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
    add_log_msg("info", log_msg.str());
    log_msg.str("");

    log_msg << ">>>> LVL2 PIVOT STATISTICS | Total Pivot Time: "<<round(model.pivot_time/1000)  \
    <<" | CYCLE Time: "<<round(model.cycle_discovery_time/1000)                                      \
    <<" | RESOLVE Time: "<<round(model.resolve_time/1000)                              \
    <<" | ADJUST Time: "<<round(model.adjustment_time/1000)                            \
    <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
    add_log_msg("info", log_msg.str());
    log_msg.str("");
  }

  else if (problem.algo == ProblemInstance::my_algo::switch_hybrid)
  {
    switchModel_parallel model = switchModel_parallel(&problem, flows);
    model.execute();
    model.create_flows();
    log_msg << ">>>> BASIC STATISTICS | Objective: "<<model.objVal<<" | Iterations: "<<model.totalIterations<<" | Time: "<<model.totalSolveTime \
    <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
    add_log_msg("info", log_msg.str());
    log_msg.str("");

    log_msg << ">>>> LVL1 STATISTICS(A) | PIVOT Time: "<<round(model.pivot_time/1000)     \
    <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
    add_log_msg("info", log_msg.str());
    log_msg.str("");

    log_msg << ">>>> LVL1 STATISTICS(B) | UV Time: "<<round(model.uv_time/1000)          \
    <<" | R Time: "<<round(model.reduced_cost_time/1000)                                 \
    <<" | PIVOT Time: "<<round(model.pivot_time/1000)                                    \
    <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
    add_log_msg("info", log_msg.str());
    log_msg.str("");

    log_msg << ">>>> LVL2 PIVOT STATISTICS | Total Pivot Time: "<<round(model.pivot_time/1000)  \
    <<" | CYCLE Time (parallel BFS + seq DFS): "<<round(model.cycle_discovery_time/1000)                                      \
    <<" | RESOLVE Time: "<<round(model.resolve_time/1000)                              \
    <<" | ADJUST Time: "<<round(model.adjustment_time/1000)                            \
    <<" | MATRIX : "<<problem.numSupplies<<" X "<<problem.numDemands;
    add_log_msg("info", log_msg.str());
    log_msg.str("");

  }

  add_log_msg("info","Flows created successfully!");

  // for (int i=0; i<(problem.active_flows); i++) {
  //   std::cout<<flows[i]<<std::endl;
  // }
  
  free(flows);
  add_log_msg("info","Flows freed successfully!");
}
