#include "parallel_structs.h"

// Create a differences vector (of row and column differences)
__global__ void initializeDifferencesVector(vogelDifference *minima, MatrixCell *row_book, MatrixCell *col_book,
                                            int n_rows, int n_cols);

// Update differences after every iteration
__global__ void updateDifferences(vogelDifference *minima, MatrixCell *row_book, MatrixCell *col_book,
                                  bool *covered, int n_rows, int n_cols, int prev_eliminated);

// Orchestrator function for executing vogel's algorithm
__host__ void find_vogel_bfs_parallel(int *supplies, int *demands, MatrixCell *costMatrix,
                                      flowInformation *flows, int numSupplies, int numDemands);

__host__ void find_vogel_bfs_sequencial(int *supplies, int *demands, MatrixCell *costMatrix,
                                      flowInformation *flows, int numSupplies, int numDemands);