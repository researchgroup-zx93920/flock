#include<iostream>
#include<fstream>

#ifndef UTILS
#define UTILS
struct FileNotFoundException : public std::exception
{
	const char * what () const throw ()
    {
    	return "Input file not found!";
    }
};


void readSize(int &matrixDemands, int &matrixSupplies, std::string filename);
void readFile(int * supplies, int * demands, double * costMatrix, std::string filename);

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