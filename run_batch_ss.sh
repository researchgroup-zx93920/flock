#!/bin/sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:"/u/samiran2/mohit/boost_1_79_0/stage/lib/"

./bin/flock -i  ./data/TransportModel_100_100_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 01 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_100_2_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 02 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_100_3_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 03 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_250_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 04 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_500_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 05 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_100_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 06 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 07 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_2_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 08 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_3_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 09 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_500_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 10 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_1000_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 11 | PARALLEL SS SOLVER COMPLETE ***************** "

# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a parallel_ss
# echo " ******************** TEST 12 | PARALLEL SS SOLVER COMPLETE ***************** "
# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a parallel_uv
# echo " ******************** TEST 12 | GPU UV METHOD COMPLETE ***************** "
# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a parallel_ss
# echo " ******************** TEST 12 | GPU SS METHOD COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_250_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 13 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 14 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_2_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 15 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_3_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 16 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_750_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 17 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_1000_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 18 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_500_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 19 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 20 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_2_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 21 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_3_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 22 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_1000_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 23 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_500_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 24 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_750_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 25 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 26 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_2_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 27 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_3_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 28 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 29 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_2_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 30 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_3_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 31 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 32 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_2_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 33 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_3_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 34 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_1_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 35 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_2_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 36 | PARALLEL SS SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_3_equalityConstr.dat  -a parallel_ss
echo " ******************** TEST 37 | PARALLEL SS SOLVER COMPLETE ***************** "


















































































