#include "lp_model.h"

lpModel::lpModel(ProblemInstance *problem, flowInformation *flows)
{
	BOOST_LOG_TRIVIAL(debug) << "Initializing LP Model";
	GRBEnv env = GRBEnv();
	env.set(GRB_IntParam_OutputFlag, GRB_LOGS);
	model = new GRBModel(env);
	// data = (ProblemInstance *)malloc(sizeof(ProblemInstance));
	data = problem;
	optimal_flows = flows; // Might require a malloc
						   // construction provision - perform some-more future things here :
	BOOST_LOG_TRIVIAL(debug) << "An LP model object was successfully created";
}

lpModel::~lpModel()
{
	delete model;
}

void lpModel::create_variables()
{
	// Add variables
	BOOST_LOG_TRIVIAL(debug) << "Creating Variables";
	for (int i = 0; i < data->numSupplies; i++)
	{
		for (int j = 0; j < data->numDemands; j++)
		{
			// Objective coeff's defined with variable >>
			int _key = i * data->numDemands + j;
			float _cost = data->costs[_key];
			std::pair<int, GRBVar> _pair(
				_key, model->addVar(0, GRB_INFINITY, _cost, (RELAXED_X_VAR == 0) ? GRB_CONTINUOUS : GRB_INTEGER));
			x_ij.insert(_pair);
		}
	}
	BOOST_LOG_TRIVIAL(debug) << "Creating Variables - completed!";
}

void lpModel::add_constraints()
{
	// Add constraints

	// Constraint 1 :: Demand Statisfaction of index - j
	BOOST_LOG_TRIVIAL(debug) << "Creating Consraint - Demand Statisfaction";
	for (int j = 0; j < data->numDemands; j++)
	{
		GRBLinExpr lhs = 0;
		for (int i = 0; i < data->numSupplies; i++)
		{
			int _key = i * data->numDemands + j;
			std::map<int, GRBVar>::iterator _iter = x_ij.find(_key);
			lhs += _iter->second;
		}
		std::stringstream s;
		s << "satisfyDemand_" << j;
		model->addConstr(lhs >= data->demands[j], s.str());
	}

	// Constraint 2 :: Supply Restrictions on index - i
	BOOST_LOG_TRIVIAL(debug) << "Creating Consraint - Demand Supply Restriction";
	for (int i = 0; i < data->numSupplies; i++)
	{
		GRBLinExpr lhs = 0;
		for (int j = 0; j < data->numDemands; j++)
		{
			int _key = i * data->numDemands + j;
			std::map<int, GRBVar>::iterator _iter = x_ij.find(_key);
			lhs += _iter->second;
		}
		std::stringstream s;
		s << "restrictSupply_" << i;
		model->addConstr(lhs <= data->supplies[i], s.str());
	}
}

void lpModel::add_objective()
{
	// Void : Objective is added on the GRB variable arugment
}

void lpModel::get_dual_costs()
{
	// Void : Future provision
}

void lpModel::solve()
{
	BOOST_LOG_TRIVIAL(debug) << "Starting Solver";
	auto start = std::chrono::high_resolution_clock::now();
	model->set(GRB_DoubleParam_TimeLimit, GRB_TIMEOUT);
	
	if (DISABLE_AUTOGRB == 1){
		// Enforce dual simplex 
		model->set(GRB_IntParam_Method, 0);
		model->set(GRB_IntParam_Sifting, 0);
	}

	model->optimize();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	data->solveTime = duration.count();

	int optim_status = model->get(GRB_IntAttr_Status);
	if (optim_status == 9)
	{
		data->solveTime *= -1;
		BOOST_LOG_TRIVIAL(error) << "Solver did not complete successfully!";
		exit(-1);
	}

	else
	{
		// fetch solution >>
		BOOST_LOG_TRIVIAL(debug) << "Solver Success! Optimal value: " << model->get(GRB_DoubleAttr_ObjVal) << std::endl;
		solved = true;
	}
}

/*
Create flows from the solved model
*/
void lpModel::create_flows()
{
	BOOST_LOG_TRIVIAL(debug) << "Creating flows ...";
	if (solved)
	{	
		BOOST_LOG_TRIVIAL(debug) << "verified model = solved";
		int _counter = 0;
		for (int i = 0; i < data->numSupplies; i++)
		{
			for (int j = 0; j < data->numDemands; j++)
			{
				int _key = i * data->numDemands + j;
				std::map<int, GRBVar>::iterator _iter = x_ij.find(_key);
				if (_iter->second.get(GRB_DoubleAttr_X) > 0)
				{
					flowInformation this_flow = {.source = i, .destination = j, .qty = float(_iter->second.get(GRB_DoubleAttr_X))};
					optimal_flows[_counter] = this_flow;
					_counter++;
				}
			}
		}

		data->active_flows = _counter;
	
	}
	else {
		BOOST_LOG_TRIVIAL(error) << "Model is not yet solved";
	}
}

void lpModel::execute()
{
	BOOST_LOG_TRIVIAL(info) << "Executing LP Model ... ";
	create_variables();
	add_constraints();
	add_objective();
	solve();
	BOOST_LOG_TRIVIAL(debug) << "LP Model was successfully solved!";
}