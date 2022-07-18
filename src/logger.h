#include <iostream>
#include <chrono>

#include "spdlog/spdlog.h"

// #include <boost/log/core.hpp>
// #include <boost/log/trivial.hpp>
// #include <boost/log/expressions.hpp>
// #include <boost/log/sinks/text_file_backend.hpp>
// #include <boost/log/utility/setup/file.hpp>
// #include <boost/log/utility/setup/common_attributes.hpp>
// #include <boost/log/utility/setup/console.hpp>
// #include <boost/log/sources/severity_logger.hpp>
// #include <boost/log/sources/record_ostream.hpp>

#ifndef LOGGER
#define LOGGER

// ************************
// Parameters >>
// ************************
#define LOG_LEVEL 0
/*
0 : Debug
1 : Info
2 : Warning
3 : Error
*/

#define LOG_TO_FILE 0
/*
Do not sink logs to file - 0
Sink logs to file - otherwise
*/

// END OF PARAMETERS 

// namespace logging = boost::log;
// namespace src = boost::log::sources;
// namespace sinks = boost::log::sinks;
// namespace keywords = boost::log::keywords;

void set_logger(int loglvl = LOG_LEVEL);

/*
LOGGER EXAMPLES >>

spdlog::debug("This message should be displayed..");
spdlog::info("Welcome to spdlog!");
spdlog::error("Some error message with arg: {}", 1);

spdlog::warn("Easy padding in numbers like {:08d}", 12);
spdlog::critical("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
spdlog::info("Support for floats {:03.2f}", 1.23456);
spdlog::info("Positional args are {1} {0}..", "too", "supported");
spdlog::info("{:<30}", "left aligned");
*/

#endif