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
#include <nvToolsExt.h>

#define blockSize 16
#define TREE_LOOKUP(row, col, V) (col>=row?((row*V)-(row*(row+1)/2))+col:((col*V)-(col*(col+1)/2))+row)
#define epsilon 0.000001

#ifndef UV_STRUCTS
#define UV_STRUCTS
/*
Container for a transportation simplex matrix cell C-ij. It's needed to retrive back the original 
position of the cells after rearragnement in preprocessing step
    - Stores i,j
    - Stores C_ij

*/
struct MatrixCell {
    int row, col;
    float cost;

    // assignment operator >>
    __host__ __device__ MatrixCell& operator=(const MatrixCell& x)
    {
        row=x.row;
        col=x.col;
        cost=x.cost;
        return *this;
    }

    // equality comparison - doesn't modify so const
    bool operator==(const MatrixCell& x) const
    {
        return cost == x.cost;
    }

    bool operator<=(const MatrixCell& x) const
    {
        return cost <= x.cost;
    }

    bool operator<(const MatrixCell& x) const
    {
        return cost < x.cost;
    }

    bool operator>=(const MatrixCell& x) const
    {
        return cost >= x.cost;
    }

    bool operator>(const MatrixCell& x) const
    {
        return cost > x.cost;
    }
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

std::ostream& operator << (std::ostream& o, const MatrixCell& x);
std::ostream& operator << (std::ostream& o, const vogelDifference& x);

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

struct rowNodes {
    std::vector<int> child;
    bool covered;
};

struct colNodes {
    std::vector<int> parent;
    bool covered;
};

struct Edge {
    int left, right;
};

// A class to represent a graph object
class Graph
{
public:
 
    // a vector of vectors to represent an adjacency list
    std::vector<std::vector<int>> adjList;
 
    // Graph Constructor
    Graph(std::vector<Edge> const &edges, int n)
    {
        // resize the vector to hold `n` elements of type `vector<int>`
        adjList.resize(n);
 
        // add edges to the undirected graph
        for (auto &edge: edges)
        {
            adjList[edge.left].push_back(edge.right);
            adjList[edge.right].push_back(edge.left);
        }
    }
};

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

#define CHECK_CUDA(func) {func;}
#define CHECK_CUSPARSE(func) {func;}

std::ostream& operator << (std::ostream& o, const Variable& x);

#endif