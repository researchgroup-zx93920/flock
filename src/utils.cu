#include "utils.h"

void readSize(int &matrixDemands, int &matrixSupplies, std::string filename)
{

	std::ifstream myfile(filename.c_str());

	if (!myfile)
	{
		std::cerr << "Error: input file not found: " << filename.c_str() << std::endl;
		throw FileNotFoundException();
	}

	myfile >> matrixSupplies;
	myfile >> matrixDemands;
	myfile.close();
}

void readFile(int * supplies, int * demands, double *costMatrix, std::string filename)
{
	std::string s = filename;
	std::ifstream myfile(s.c_str());

	if (!myfile)
	{
		std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
		throw FileNotFoundException();
	}

	while (myfile.is_open() && myfile.good())
	{
		int matrixSupplies, matrixDemands;

		myfile >> matrixSupplies;
		myfile >> matrixDemands;

		// std::cout<<"File Name : "<<fileName<<std::endl;
    	// std::cout<<"Matrix Supplies : "<<matrixSupplies<<std::endl;
    	// std::cout<<"Matrix Demands : "<<matrixDemands<<std::endl;
        
        // Read Supplies >> 
        for (int i=0; i < matrixSupplies; i++){
            myfile >> supplies[i];
        }

        // Read Demands >> 
        for (int j=0; j < matrixDemands; j++){
            myfile >> demands[j];
        }

        // Read Cost Matrix >>
		long largestIndx = matrixSupplies*matrixDemands;

		for (long i = 0; i < largestIndx; i++)
		{	
			myfile >> costMatrix[i];
		}
		
		myfile.close();
	}

	
}
