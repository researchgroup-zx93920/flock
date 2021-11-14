#include<iostream>
#include<fstream>

#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#ifndef UTILS
#define UTILS

struct FileNotFoundException : public std::exception
{
	const char * what () const throw ()
    {
    	return "Input file not found!";
    }
};

__host__ __device__ struct MatrixCell {
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

struct vogelDifference {
        float diff;
        int indx, ileast_1, ileast_2;
        // idx it itselves index in diff array
        // ileast_1 and ileast2 are indexes of min-2 values
        // least_1,least_2,
};

void readSize(int &matrixDemands, int &matrixSupplies, std::string filename);
void readFile(int * supplies, int * demands, MatrixCell * costMatrix, std::string filename);

std::ostream& operator << (std::ostream& o, const MatrixCell& x);

template <typename T>
void printLocalDebugArray(T * d_array, int rows, int columns, const char *name) {
    std::cout<<"\n"<<name<<"\n"<<std::endl;   
    for (int i = 0; i < rows; i++){
        for (int j=0; j < columns; j++){
            // std::cout << " " << i*columns+j << " " << d_array[i*columns+j] << "\t";
            std::cout << d_array[i*columns+j] << "\t";
        }
        std::cout<<std::endl;
    }
    std::cout << std::endl;
}

#endif