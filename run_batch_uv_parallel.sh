#!/bin/sh

./bin/flock -i  ./data/TransportModel_100_100_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 01 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_100_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 02 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_100_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 03 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_250_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 04 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 05 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_100_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 06 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 07 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 08 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 09 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 10 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_1000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 11 | PARALLEL UV SOLVER COMPLETE ***************** "

# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a parallel_uv
# echo " ******************** TEST 12 | PARALLEL UV SOLVER COMPLETE ***************** "
# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a parallel_uv
# echo " ******************** TEST 12 | GPU UV METHOD COMPLETE ***************** "
# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a parallel_ss
# echo " ******************** TEST 12 | GPU SS METHOD COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_250_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 13 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 14 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 15 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 16 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_750_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 17 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_1000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 18 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 19 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 20 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 21 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 22 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_1000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 23 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 24 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_750_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 25 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 26 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 27 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 28 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 29 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 30 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 31 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 32 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 33 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 34 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 35 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 36 | PARALLEL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 37 | PARALLEL UV SOLVER COMPLETE ***************** "


















































































