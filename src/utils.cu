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

void readFile(int * supplies, int * demands, MatrixCell * costMatrix, std::string filename)
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
		float _cost;

		for (int i=0; i<matrixSupplies; i++){
			for (int j=0; j<matrixDemands; j++) {
				myfile >> _cost;
				int indx = i*matrixDemands + j;
				MatrixCell m = {.row = i, .col = j, .cost = _cost};
				costMatrix[indx] = m;
			}
		}

		myfile.close();
	}
}

std::ostream& operator << (std::ostream& o, const MatrixCell& x) {
    o << x.cost;
    return o;
}

std::ostream& operator << (std::ostream& o, const vogelDifference& x) {
    o << x.diff <<" | Least 1 : "<<x.ileast_1 <<" | Least 2: "<<x.ileast_2;
    return o;
}

std::ostream& operator << (std::ostream& o, const flowInformation& x) {
    o << "Flow " << x.qty <<" units from row: "<<x.row <<" to col: "<<x.col<<std::endl;
    return o;
}

