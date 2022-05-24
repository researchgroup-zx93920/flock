CPP=g++ -std=c++14 -DBOOST_LOG_DYN_LINK 
CUDA_COMPILER=nvcc -arch=sm_75 -w -std=c++14 -Xcompiler -fopenmp -O3 
SRC=./src

BOOST_INCLUDE_PATH=/home/mohitm3/cpp_libs/boost_1_78_0
BOOST_LIB_PATH=$(BOOST_INCLUDE_PATH)/stage
LIB_BOOST=-lboost_regex -lboost_program_options -lboost_log -lboost_log_setup -lboost_system -lboost_thread -lpthread -lboost_filesystem

GUROBI_HOME=/home/mohitm3/gurobi912/linux64
LIB_GUROBI=-lgurobi_c++ -lgurobi91

CUDA=/usr/local/cuda
LIB_CUDA=-lcudart -lnvToolsExt -lcusparse -lcusolver

all: flock

flock: ensureDir lpMethod.o seq_simplex.o parallel_method.o
	$(CPP) $(SRC)/*.cpp ./bin/*.o -I$(GUROBI_HOME)/include/ -I$(BOOST_INCLUDE_PATH)/ -I$(CUDA)/include/ -L$(GUROBI_HOME)/lib/ -L$(BOOST_LIB_PATH)/lib/ -L$(CUDA)/lib64/ $(LIB_GUROBI) $(LIB_BOOST) $(LIB_CUDA) -lgomp -o ./bin/flock

lpMethod.o:
	$(CPP) -c $(SRC)/gurobi_lp_method/*.cpp -I$(GUROBI_HOME)/include/ -I$(BOOST_INCLUDE_PATH)/ -L$(GUROBI_HOME)/lib/ -L$(BOOST_LIB_PATH)/lib/ $(LIB_GUROBI) $(LIB_BOOST) -o ./bin/lpMethod.o

seq_simplex.o:
	$(CPP) -c $(SRC)/basic_vogel/*.cpp -I$(BOOST_INCLUDE_PATH)/ -L$(BOOST_LIB_PATH)/lib/ $(LIB_BOOST) -o ./bin/basic_vogel.o

parallel_method.o:
	$(CUDA_COMPILER) -c $(SRC)/parallel_uv_method/PIVOT_dfs.cu -o ./bin/PIVOT_dfs.o
	$(CUDA_COMPILER) -c $(SRC)/parallel_uv_method/uv_model_parallel.cu -o ./bin/uv_model_parallel.o
	$(CUDA_COMPILER) -c $(SRC)/parallel_uv_method/DUAL_solver.cu -o ./bin/DUAL_solver.o
	$(CUDA_COMPILER) -c $(SRC)/parallel_uv_method/IBFS_nwc.cu -o ./bin/IBFS_nwc.o
	$(CUDA_COMPILER) -c $(SRC)/parallel_uv_method/IBFS_vogel.cu -o ./bin/IBFS_vogel.o
	$(CUDA_COMPILER) -c $(SRC)/parallel_uv_method/parallel_kernels.cu -o ./bin/parallel_kernels.o
	$(CUDA_COMPILER) -c $(SRC)/parallel_uv_method/parallel_structs.cu -o ./bin/parallel_structs.o

ensureDir:
	mkdir -p ./bin
	touch ./bin/flock

clean:
	rm ./bin/*.o

