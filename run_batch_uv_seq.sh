#!/bin/sh

./bin/flock -i  ./data/TransportModel_100_100_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 01 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_100_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 02 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_100_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 03 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_250_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 04 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 05 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_100_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 06 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 07 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 08 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 09 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 10 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_1000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 11 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a parallel_uv
# echo " ******************** TEST 12 | SEQUENCIAL UV SOLVER COMPLETE ***************** "
# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a parallel_uv
# echo " ******************** TEST 12 | GPU UV METHOD COMPLETE ***************** "
# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a parallel_ss
# echo " ******************** TEST 12 | GPU SS METHOD COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_250_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 13 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 14 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 15 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 16 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_750_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 17 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_1000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 18 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 19 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 20 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 21 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 22 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_1000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 23 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 24 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_750_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 25 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 26 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 27 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 28 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 29 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 30 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 31 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 32 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 33 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 34 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_1_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 35 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_2_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 36 | SEQUENCIAL UV SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_3_equalityConstr.dat  -a parallel_uv
echo " ******************** TEST 37 | SEQUENCIAL UV SOLVER COMPLETE ***************** "


















































































