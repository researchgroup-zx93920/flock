# Run a bunch of small problems >>
# Progressively gets more complex
# 100_1 to 100_6 are robust cases testing different number scales

# Simple Problems
./bin/flock -i  ./tests/test_simple1.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_simple2.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_simple3.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "

# Test Slight complication
./bin/flock -i  ./tests/test_simple1_degenerate.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "

# Slight more work
./bin/flock -i  ./tests/test_TransportModel_5_5_0.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_10_10_0.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_100_100_0.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "

# Test Numerical Robustness
./bin/flock -i  ./tests/test_TransportModel_100_100_1.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_100_100_2.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_100_100_3.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_100_100_4.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_100_100_5.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_100_100_6.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "

# Test Scaling 
./bin/flock -i  ./tests/test_TransportModel_100_250_1.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_100_500_1.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_250_250_1.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_250_500_1.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "
./bin/flock -i  ./tests/test_TransportModel_500_500_1.dat  -a parallel_uv_solve
echo " ******************** COMPLETE ***************** "

