#include "instance.h"

void get_input_mode_and_filename(InputParser input, ProblemInstance &problem)
{

    if (input.cmdOptionExists("-i"))
    {
        problem.read_mode = true;
        problem.filename = input.getCmdOption("-i");
        if (problem.filename.empty())
        {
            BOOST_LOG_TRIVIAL(error) << "No filename provided for arugment -i";
            throw FileNotFoundException();
            exit(-1);
        }
        else
        {
            BOOST_LOG_TRIVIAL(debug) << "File Name = " << problem.filename;
        }
    }
    else
    {
        problem.read_mode = false;
        BOOST_LOG_TRIVIAL(error) << "Provide input file with argument -i";
        problem.filename = "/!\\ NO FILE INPUT - FUTURE OF MAKE RANDOMLY GENERATED INSTANCE";
        BOOST_LOG_TRIVIAL(error) << problem.filename;
        throw FileNotFoundException();
        exit(-1);
        // Future: Auto Generate an instance at random
    }
}

void get_algorithm(InputParser input, ProblemInstance &problem)
{
    std::string default_algo = "cpu_lp_solve";
    std::string user_algo;
    if (input.cmdOptionExists("-a"))
    {
        user_algo = input.getCmdOption("-a");
        if (user_algo.empty())
        {
            user_algo = default_algo;
        }
    }
    else
    {
        user_algo = default_algo;
    }

    // Convert string argument to enum -
    if (user_algo == "cpu_lp_solve")
    {
        problem.algo = ProblemInstance::my_algo::cpu_lp_solve;
    }
    else if (user_algo == "parallel_uv"){
        problem.algo = ProblemInstance::my_algo::parallel_uv;
    }
    else if (user_algo == "parallel_ss"){
        problem.algo = ProblemInstance::my_algo::parallel_ss;
    }
    else if (user_algo == "switch_hybrid"){
        problem.algo = ProblemInstance::my_algo::switch_hybrid;
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "Invalid Algorithm!";
    }
}

/*
In the method readSize - matrix Supplies and matrix Demands is eq. to
numSupplies and numDemands
*/
void readSize(ProblemInstance &problem)
{

    std::ifstream myfile(problem.filename.c_str());

    if (!myfile)
    {
        std::cerr << "Error: input file not found: " << problem.filename.c_str() << std::endl;
        throw FileNotFoundException();
    }

    myfile >> problem.numSupplies;
    myfile >> problem.numDemands;
    myfile.close();
}

void readCosts(ProblemInstance &problem)
{
    std::string s = problem.filename;
    std::ifstream myfile(s.c_str());

    if (!myfile)
    {
        std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
        throw FileNotFoundException();
    }

    while (myfile.is_open() && myfile.good())
    {
        int matrixSupplies, matrixDemands;

        myfile >> matrixSupplies;
        myfile >> matrixDemands;

        // Read Supplies >>
        for (int i = 0; i < matrixSupplies; i++)
        {
            myfile >> problem.supplies[i];
        }

        // Read Demands >>
        for (int j = 0; j < matrixDemands; j++)
        {
            myfile >> problem.demands[j];
        }

        // Read Cost Matrix >>
        float _cost;

        for (int i = 0; i < matrixSupplies; i++)
        {
            for (int j = 0; j < matrixDemands; j++)
            {
                myfile >> _cost;
                int indx = i * matrixDemands + j;
                problem.costs[indx] = _cost;
            }
        }

        myfile.close();
    }
}

bool is_file_exist(std::string fileName)
{
    std::ifstream infile(const_cast<char*>(fileName.c_str()));
    return infile.good();
}

void make_problem(InputParser input, ProblemInstance &problem)
{

    auto start = std::chrono::high_resolution_clock::now();
    
    get_input_mode_and_filename(input, problem);
    // does filePath actually exist?
    
    if (is_file_exist(problem.filename))
    {
        BOOST_LOG_TRIVIAL(debug) << "File available, Now reading file ...";
        readSize(problem);
        BOOST_LOG_TRIVIAL(debug) << "Determined problem dimensions, Now reading costs ...";
        problem.allocate_memory();
        readCosts(problem);
        BOOST_LOG_TRIVIAL(debug) << "Reading input file - complete!";
    }
    else
    {
        BOOST_LOG_TRIVIAL(error) << "File :" << problem.filename << " doesn't exist!";
        throw FileNotFoundException();
        exit(-1);
    }

    get_algorithm(input, problem);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    problem.readTime += duration.count();
    problem.active_flows = problem.numSupplies + problem.numDemands - 1; 

}