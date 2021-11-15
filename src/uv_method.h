#include<iostream>
#include "utils.h"

#define blockSize 512

#ifndef UV
#define UV

__host__ void find_current_uv();

__host__ void find_reduced_costs();

__host__ void pivot();

#endif