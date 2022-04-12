
# include "./gurobi_lp_method/lp_model.h"
# include "./parallel_uv_method/uv_model_parallel.h"
# include "./basic_vogel/vogel_sequencial.h"

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