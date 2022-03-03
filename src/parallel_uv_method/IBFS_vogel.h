#include <iostream>
#include <cstring>
#include <map>


#include "../structs.h"
#include "parallel_structs.h"

struct rowgen
{
        __host__ __device__ int operator()(MatrixCell &x) const
        {
                return x.row;
        }
};

struct colgen
{
        __host__ __device__ int operator()(MatrixCell &x) const
        {
                return x.col;
        }
};

struct compareCells
{
        __host__ __device__ bool operator()(const MatrixCell i, const MatrixCell j) const
        {
                return (i.cost <= j.cost);
        }
};

struct compareDiff
{
        __host__ __device__ bool operator()(const vogelDifference i, const vogelDifference j) const
        {
                return (i.diff <= j.diff);
        }
};

__global__ void initializeDifferencesVector(vogelDifference *minima, MatrixCell *row_book, MatrixCell *col_book,
                                            int n_rows, int n_cols);

__global__ void updateDifferences(vogelDifference *minima, MatrixCell *row_book, MatrixCell *col_book,
                                  bool *covered, int n_rows, int n_cols, int prev_eliminated);

__host__ void find_vogel_bfs_parallel(int *supplies, int *demands, MatrixCell *costMatrix,
                                      flowInformation *flows, std::map<std::pair<int, int>, int> &flow_indexes,
                                      int numSupplies, int numDemands);