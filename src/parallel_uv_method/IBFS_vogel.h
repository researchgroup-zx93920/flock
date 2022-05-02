#include "parallel_structs.h"

// Retrieve row attribute of a Matrix Cell object
struct rowgen
{
        __host__ __device__ int operator()(MatrixCell &x) const
        {
                return x.row;
        }
};

// Retrieve column attribute of a Matrix Cell object
struct colgen
{
        __host__ __device__ int operator()(MatrixCell &x) const
        {
                return x.col;
        }
};

// Unary method to compare two cells of a cost matrix
struct compareCells
{
        __host__ __device__ bool operator()(const MatrixCell i, const MatrixCell j) const
        {
                return (i.cost < j.cost);
        }
};

// Unary method to compare differences/penalties b/w any pair of row/col 
struct compareDiff
{
        __host__ __device__ bool operator()(const vogelDifference i, const vogelDifference j) const
        {
                return (i.diff < j.diff);
        }
};

// Create a differences vector (of row and column differences)
__global__ void initializeDifferencesVector(vogelDifference *minima, MatrixCell *row_book, MatrixCell *col_book,
                                            int n_rows, int n_cols);

// Update differences after every iteration
__global__ void updateDifferences(vogelDifference *minima, MatrixCell *row_book, MatrixCell *col_book,
                                  bool *covered, int n_rows, int n_cols, int prev_eliminated);

// Orchestrator function for executing vogel's algorithm
__host__ void find_vogel_bfs_parallel(int *supplies, int *demands, MatrixCell *costMatrix,
                                      flowInformation *flows, std::map<std::pair<int, int>, int> &flow_indexes,
                                      int numSupplies, int numDemands);