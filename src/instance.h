#include <iostream>
#include <boost/filesystem.hpp>

#include "logger.h"
#include "utils.h"
#include "structs.h"

struct FileNotFoundException : public std::exception
{
    const char *what() const throw()
    {
        return "File not found!";
    }
};

class InputParser
{
public:
    InputParser(int &argc, char **argv)
    {
        for (int i = 1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }
    /// @author iain
    const std::string &getCmdOption(const std::string &option) const
    {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end())
        {
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }
    /// @author iain
    bool cmdOptionExists(const std::string &option) const
    {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

private:
    std::vector<std::string> tokens;
};

/*
Some future todo's for Mohit:
    - Auto Balance the transporation problem
*/
void get_input_mode_and_filename(InputParser input, ProblemInstance &problem);
void get_algorithm(InputParser input, ProblemInstance &problem);
void make_problem(InputParser input, ProblemInstance &problem);
void readSize(ProblemInstance &problem);
void readCosts(ProblemInstance &problem);