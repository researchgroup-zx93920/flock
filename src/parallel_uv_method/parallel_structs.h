#include<iostream>
#include<map>
#include<vector>
#include<algorithm>
#include<utility>

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
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

// Profiling
// #include <nvToolsExt.h>

#ifndef UV_STRUCTS
#define UV_STRUCTS

// PARAMETERS
#define blockSize 16

// Degeneracy resolve
#define epsilon 0.000001f

#define DETAILED_LOGS 1
/*
0 : Silent model
1 : Verbose model
*/

#define BFS_METHOD "vam"
/*
nwc : Northwest Corner - sequential implementation
vam : vogel's approximation - parallel regret implementation
*/

#define CALCULATE_DUAL "lin_solver"
/*
tree : traverse the tree in parallel to find values on verties
lin_solver : solve system of lin_equations
*/

#define PIVOTING_STRATEGY "sequencial"
/*
sequencial : perform pivoting one at a time based on dantzig's rule
parallel : perform parallel pivoting
*/

#define MAX_ITERATIONS 0

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
    float value = -99999;
    bool assigned = false;

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

std::ostream& operator << (std::ostream& o, const MatrixCell& x);
std::ostream& operator << (std::ostream& o, const vogelDifference& x);
std::ostream& operator << (std::ostream& o, const Variable& x);

#endif