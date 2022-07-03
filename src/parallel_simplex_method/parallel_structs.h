#include<iostream>
#include<cstring>
#include<map>
#include<vector>
#include<queue>
#include<algorithm>
#include<utility>
#include<chrono>
#include<math.h>
#include<stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> 
#include <device_launch_parameters.h>
#include <cusparse.h>
#include <cusolverSp.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/scan.h> 
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "../structs.h"

// Profiling
#include <nvToolsExt.h>

#ifndef PARALLEL_STRUCTS
#define PARALLEL_STRUCTS

// PARAMETERS
#define blockSize 8
#define reducedcostBlock 16
#define parallelBFSBlock 64
#define resolveBlockSize 64

// Degeneracy resolve
#define epsilon 0.000001f
#define epsilon2 10e-3

#define BFS_METHOD "vam_device"
/*
ALL THESE METHODS/KERNELS ARE IMPLEMENTED COLLECTIVELY IN - BFS_* PREFIXED FILES

nwc_host : Northwest Corner - sequential implementation
vam_host : Northwest Corner - sequential implementation
vam_device : vogel's approximation - parallel regret implementation
*/

#define CALCULATE_DUAL "host_bfs"
/*
ALL THESE METHODS/KERNELS ARE IMPLEMENTED COLLECTIVELY IN - DUAL_solver.h/.cu FILES

device_bfs : traverse the tree (bfs fashion) in parallel to find values on verties
host_bfs : just like bfs - performed on host
host_sparse_linear_solver : solver system of sparse lin_equations on host
host_dense_linear_solver : solver system of dense lin_equations on host
device_sparse_linear_solver : solve system of sparse lin_equations on device (sparse linear algebra :: cusparse)
device_dense_linear_solver : solve system of dense lin_equations (dense linear algebra :: cublas)
*/

#define MAX_DECONFLICT_CYCLES(M, N) ((M+N-1)/3)
/* 
IDEA:
Theoretically k = FLOOR[(M + N - 1)/3] deconflcited cycles can exist
We'll discover COMPACTION_FACTOR * k parallel pivots and resolve conflicts within them
in the hope that we'll get the most of the deconflicted cycles from the 
top cream of negative reduced costs
*/

#define MAX_ITERATIONS 100000

/* Upper bound on max number of independent pivots */


#define REDUCED_COST_MODE "parallel"
/*
A mode for switching to pure sequencial algorithm
change it to "sequencial" or "parallel"
*/

#define SEQ_CYCLE_SEARCH "bfs"
/*
dfs, bfs 
*/

// >>>>>>>>>> END OF PARAMETERS // 

#define TREE_LOOKUP(row, col, V) (col>=row?((row*V)-(row*(row+1)/2))+col:((col*V)-(col*(col+1)/2))+row)

// Credit : https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Credit: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/utils/cusolver_utils.h
// cusolver API error checking
/*
    CUSOLVER_STATUS_SUCCESS=0,
    CUSOLVER_STATUS_NOT_INITIALIZED=1,
    CUSOLVER_STATUS_ALLOC_FAILED=2,
    CUSOLVER_STATUS_INVALID_VALUE=3,
    CUSOLVER_STATUS_ARCH_MISMATCH=4,
    CUSOLVER_STATUS_MAPPING_ERROR=5,
    CUSOLVER_STATUS_EXECUTION_FAILED=6,
    CUSOLVER_STATUS_INTERNAL_ERROR=7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
    CUSOLVER_STATUS_NOT_SUPPORTED = 9,
    CUSOLVER_STATUS_ZERO_PIVOT=10,
    CUSOLVER_STATUS_INVALID_LICENSE=11,
    CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED=12,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID=13,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC=14,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE=15,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER=16,
    CUSOLVER_STATUS_IRS_INTERNAL_ERROR=20,
    CUSOLVER_STATUS_IRS_NOT_SUPPORTED=21,
    CUSOLVER_STATUS_IRS_OUT_OF_RANGE=22,
    CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES=23,
    CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED=25,
    CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED=26,
    CUSOLVER_STATUS_IRS_MATRIX_SINGULAR=30,
    CUSOLVER_STATUS_INVALID_WORKSPACE=31
*/

#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUSPARSE_CHECK(err)                                                                        \
    do {                                                                                           \
        cusparseStatus_t err_ = (err);                                                             \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
            printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusparse error");                                            \
        }                                                                                          \
    } while (0)

/*
Container for a transportation simplex matrix cell C-ij. It's needed to retrive back the original 
position of the cells after rearragnement in preprocessing step
    - Stores i,j
    - Stores C_ij

*/
struct MatrixCell {
    int row, col;
    float cost;
};

/*
VogelDifference is the struct to store the regret and memoize the pointer 
to corresponding positions where
idx stores itselves index in difference array
ileast_1 and ileast2 are indexes 
of min-2 values (minimum and second minimum)
*/
struct vogelDifference {
        int idx, ileast_1 = 0, ileast_2 = 1;
        float diff;
};

struct stackNode {
    int index, depth;
};

struct is_zero
{
    __host__ __device__ bool operator()(const flowInformation &x)
    {
        return (x.qty == 0);
    }
};

struct is_nonzero_entry
{
    __host__ __device__ bool operator()(const int &x)
    {
        return (x != 0);
    }
};

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

struct PivotTimer {
    double cycle_discovery = 0, resolve_time = 0, adjustment_time = 0;
};

struct vertexPin {
    int from, via, to, skip;
    float recirculation;
};

struct DualHandler {

        // Commons ::
        // - dual costs towards supply constraints (u_vars_ptr)
        //  - dual costs towards demand constraints (v_vars_ptr)
        float * u_vars_ptr, * v_vars_ptr;

        // DUAL :: Sequencial BFS >>
        bool * h_visited;
        float * h_variables;
};

struct PivotHandler {
    
    // Useful for the sequencial strategy >>
    int pivot_row, pivot_col;
    float reduced_cost;
        
    // Useful for both seq and parallel strategy >>
    
    // DFS >>
    int * backtracker, * depth, * via_points, * depth_tracker; 
    bool * visited;
    stackNode * stack;
     
    // SS Method specific pointers ->
    int * d_adjMtx_transform, * d_pathMtx;
    int * d_pivot_cycles, * deconflicted_cycles; 
    int * deconflicted_cycles_depth, * deconflicted_cycles_backtracker;

    float * opportunity_cost;
    vertexPin * d_bfs_frontier_current, * d_bfs_frontier_next;
    int * current_frontier_length, * next_frontier_length;
    MatrixCell * d_reducedCosts_ptr;
    int * d_num_NegativeCosts;

    // Oppotunity cost - reduced cost - float version that stores the .cost attribute
    // Used when row col indexes are not needed --> like bfs and parallel reduced cost compute
};

/* 
It's a struct to carry appropriate pointers to pass 
along in different functions. Not necessary but created for prettier code. 

adjMtx and flowMtx has been designed with special consideration. [Adjacency matrix] and [flow matrix] together represent the feasible tree. Such a separation has been created to 
resolve degeneracies sometimes zero flows would make life better :D. For a simplex iteration, current feasible tree 
is maintained on both host (h_) and device (d_). The tree is maintained in the form of adjMtx as well as adjList
 
Adjacency matrix is a upper triangular matrix to store the tree information. Example: 
Let's say you're looking at edge (i,j)
Now you want to find flow on i,j in the feasible tree
Then, note this: 
    1. (i,j) is a feasible edge if adjMtx(i,j) > 0
    2. Further, flow on (i,j) = flow[adjMtx(i,j)-1], 
In words: entries in  adjMtx point to the position of flow values stored in flowMtx

Compressed transformation of adjMtx on host device, Consider a vertex - i,
    - Vertex - i has d_vertex_degree[i] number of neighbours
    - Neighbours are in d_adjVertices array starting from from index d_vertex_start[i] upto 
        d_vertex_start[i] + d_vertex_degree[i]
    - Such a transformation is created to perfrom BFS and DFS efficiently in case of dual,
        Also, applying this transformation is massively parallel
*/
struct Graph {

    // Number of vertices
    int V;

    // adjMatrix and flow on host device
    int * h_adjMtx_ptr, * d_adjMtx_ptr;
    float * h_flowMtx_ptr, * d_flowMtx_ptr; 

    // Compressed transformation
    int * d_vertex_start, * d_vertex_degree, * d_adjVertices;
    int * h_vertex_start, * h_vertex_degree, * h_adjVertices;

    std::vector<std::vector<int>> h_Graph;
};

std::ostream& operator << (std::ostream& o, const vertexPin& x);
std::ostream& operator << (std::ostream& o, const MatrixCell& x);
std::ostream& operator << (std::ostream& o, const vogelDifference& x);


#endif