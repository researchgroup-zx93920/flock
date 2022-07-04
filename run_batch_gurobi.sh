#!/bin/sh

./bin/flock -i  ./data/TransportModel_100_100_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 01 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_100_2_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 02 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_100_3_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 03 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_250_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 04 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_100_500_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 05 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_100_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 06 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 07 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_2_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 08 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_250_3_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 09 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_500_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 10 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_250_1000_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 11 |LP SOLVER COMPLETE ***************** "

# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a cpu_lp_solve
# echo " ******************** TEST 12 |LP SOLVER COMPLETE ***************** "
# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a cpu_lp_solve
# echo " ******************** TEST 12 | GPU UV METHOD COMPLETE ***************** "
# ./bin/flock -i  ./data/TransportModel_500_100_1_equalityConstr.dat  -a parallel_ss
# echo " ******************** TEST 12 | GPU SS METHOD COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_250_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 13 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 14 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_2_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 15 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_500_3_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 16 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_750_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 17 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_500_1000_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 18 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_500_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 19 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 20 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_2_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 21 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_750_3_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 22 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_750_1000_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 23 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_500_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 24 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_750_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 25 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 26 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_2_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 27 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_1000_3_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 28 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 29 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_2_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 30 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_2500_3_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 31 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 32 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_2_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 33 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_5000_3_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 34 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_1_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 35 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_2_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 36 |LP SOLVER COMPLETE ***************** "

./bin/flock -i  ./data/TransportModel_1000_10000_3_equalityConstr.dat  -a cpu_lp_solve
echo " ******************** TEST 37 |LP SOLVER COMPLETE ***************** "


















































































