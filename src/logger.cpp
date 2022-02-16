#include "logger.h"

void set_logger(int loglvl)
{
    std::cout<<"FLOCK: Setting up logger"<<std::endl;
    // Now initialize logfile and set log formatting >>
    std::time_t epoch = std::time(nullptr);
    std::stringstream logfile;
    logfile << epoch;
    logging::add_common_attributes();
    logging::register_simple_formatter_factory<logging::trivial::severity_level, char>("Severity");
    if (!LOG_TO_FILE==0){
        logging::add_file_log(
        keywords::file_name = "./logs/FLOCK_" + logfile.str() + ".log",
        // keywords::rotation_size = 10 * 1024 * 1024,
        // keywords::time_based_rotation = sinks::file::rotation_at_time_point(0, 0, 0),
        keywords::format = "[%TimeStamp%] | %Severity% : %Message%");
    }

    logging::add_console_log(std::cout, boost::log::keywords::format = "[%TimeStamp%]: %Message%");

    switch (loglvl)
    {
    case 0:
    {
        logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::debug);
        break;
    }
    case 1:
    {
        logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::info);
        break;
    }
    case 2:
    {
        logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::warning);
        break;
    }
    case 3:
    {
        logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::error);
        break;
    }

    default:
    {
        std::cout << "Invalid logging level" << std::endl;
        throw std::exception();
        break;
    }
    }
}