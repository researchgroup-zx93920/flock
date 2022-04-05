#include "parallel_structs.h"

std::ostream &operator<<(std::ostream &o, const MatrixCell &x)
{
    o << "Cost(" <<x.row<<", " << x.col <<") = " << x.cost;
    return o;
}

std::ostream &operator<<(std::ostream &o, const vogelDifference &x)
{
    o << x.diff << " | Index of Minimum : " << x.ileast_1 << " | Index of second Minimum : " << x.ileast_2;
    return o;
}

std::ostream& operator << (std::ostream& o, const Variable& x) {
    o << x.value;
    return o;
}

// Helper function for printing device errors.
void cudaSafeCall(cudaError_t error, const char *message) {
	if (error != cudaSuccess) {
		std::cerr << "Error " << error << ": " << message << ": " << cudaGetErrorString(error) << std::endl;
		exit(-1);
	}
}

// Helper function for printing device memory info.
void printMemoryUsage(float memory) {
	size_t free_byte;
	size_t total_byte;

	cudaSafeCall(cudaMemGetInfo(&free_byte, &total_byte), "Error in cudaMemGetInfo");

	float free_db = (float) free_byte;
	float total_db = (float) total_byte;
	float used_db = total_db - free_db;

	if (memory < used_db)
		memory = used_db;

	printf("used = %f MB, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}
