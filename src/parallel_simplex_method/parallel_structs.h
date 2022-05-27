#include<iostream>
#include<cstring>
#include<map>
#include<vector>
#include<queue>
#include<algorithm>
#include<utility>
#include<chrono>
#include<math.h>

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
// #include <nvToolsExt.h>

#ifndef UV_STRUCTS
#define UV_STRUCTS

// PARAMETERS
#define blockSize 32

// Degeneracy resolve
#define epsilon 0.000001f
#define epsilon2 10e-3

#define DETAILED_LOGS 1
/*
0 : Silent model
1 : Verbose model
*/

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

#define SPARSE_SOLVER "qr"
/*
qr, chol
*/

#define PIVOTING_STRATEGY "parallel_fw"
/*
sequencial_dfs : perform pivoting one at a time based on dantzig's rule
parallel_dfs : perform parallel pivoting by DFS strategy to build cycles
parallel_fw : perform parallel pivoting by floyd warshall strategy to build cycles
*/

#define COMPACTION_FACTOR 3
/*
Lets the solver decide how many parallel cycles need to be discovered in one case
*/

#define NUM_THREADS_LAUNCHING(M, N, strategy) (strategy==1?((floor((M + N - 1)/3))*(COMPACTION_FACTOR)):2*(M+N))
/* 
IDEA 1 :
Theoretically k = FLOOR[(M + N - 1)/3] deconflcited cycles can exist
We'll discover COMPACTION_FACTOR * k parallel pivots and resolve conflicts within them
in the hope that we'll get the most of the deconflicted cycles from the 
top cream of negative reduced costs

IDEA 2 : 
Follow this paper's parameter - DOI: 10.1080/10556788.2016.1260568 
Just consider at most 2(M + N) negative reduced costs 
*/

#define PARALLEL_PIVOT_IDEA 2
/*
Idea to use from above 1/2
*/

#define PARALLEL_PIVOTING_METHOD "r"
/*
r : deconflict pivots purely based on reduced costs
delta : deconflict parallel pivots based on delta -> currently appliable to stepping stone method
*/

#define MAX_ITERATIONS 30000

/* Upper bound on max number of independent pivots */
#define MAX_DECONFLICT_CYCLES(M, N) ((M+N-1)/3)

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
        int idx, ileast_1, ileast_2;
        float diff;
};

struct Variable {

    bool assigned = false;
    float value = -99999;

    __host__ __device__ Variable& operator=(const float& x)
    {
        value=x;
        assigned=true;
        return *this;
    }
};

struct stackNode {
    int index, depth;
};

struct pathEdge {
    int index;
    pathEdge * next;
};


typedef union  {
  float floats[2];                 // floats[0] = lowest savings on this loop
  int ints[2];                     // ints[1] = index -> thread-gid
  unsigned long long int ulong;    // for atomic update
} vertex_conflicts;

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


struct DualHandler {

        // Commons:
        // - dual costs towards supply constraints (u_vars_ptr)
        //  - dual costs towards demand constraints (v_vars_ptr)
        float * u_vars_ptr = NULL, * v_vars_ptr = NULL;

        // DUAL :: Solving system of equations 
        float u_0 = 0, u_0_coef = 1;
        int * d_csr_offsets, * d_csr_columns;
        float * d_csr_values, * d_x, * d_A, * d_b;
        int64_t nnz;

        // DUAL :: Solving using a breadth first traversal on Tree
        float * variables;
        bool * Xa, * Fa;

        // DUAL :: Sequencial BFS >>
        bool * h_visited;
        float * h_variables;

};

struct PivotHandler {
    
    // Useful for the sequencial strategy >>
    int pivot_row, pivot_col;
        
    // Useful for both seq and parallel strategy >>
    
    // DFS >>
    int * backtracker, * depth; 
    bool * visited;
    stackNode * stack;

    // Flow adjustment on pivot - 
    // ******** DEPRECATED 
    // int * loop_min_from, * loop_min_to, * loop_min_id;
    // float * loop_minimum;
    
    // Resolving conflicts in parallel pivoting
    // vertex_conflicts * v_conflicts;

    // Floyd warshall specific pointers ->
    int * d_adjMtx_transform, * d_pathMtx;
    int * d_pivot_cycles, * deconflicted_cycles; 
    int * deconflicted_cycles_depth, * deconflicted_cycles_backtracker;

    // Stepping stone specific pointers (this method also uses some of the fw pointers)
    float * opportunity_cost;
    // Oppotunity cost - Cost Improvement observed per unit along the cycle
    // Delta - Possible Recirculation along the cycle 
    // (computation of delta requires communication of flows to device - this is time taking)

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

};

std::ostream& operator << (std::ostream& o, const MatrixCell& x);
std::ostream& operator << (std::ostream& o, const vogelDifference& x);
std::ostream& operator << (std::ostream& o, const Variable& x);


#endif