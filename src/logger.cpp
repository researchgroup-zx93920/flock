#include "logger.h"

void add_log_msg(std::string level, std::string log_msg) {
    // For now just print everything >> 
    std::cout<< "["<<level<<"] => "<<log_msg<<std::endl;
}
