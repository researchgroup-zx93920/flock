#ifndef PARAMS
#define PARAMS

// ************************
// Parameters >>
// ************************
#define LOG_TO_FILE 0
/*
Do not sink logs to file - 0
Sink logs to file - otherwise
*/

#define LOG_LEVEL "debug"
/*
Not fully implemented
*/


#ifndef MAKE_WITH_GUROBI
// Do not change - makefiles will modify this appropriately
#define MAKE_WITH_GUROBI 0
#endif

#endif