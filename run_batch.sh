#!bin/sh/

echo "Running all instances on gurobi"
./run_batch_gurobi.sh

echo "Running all instances on switching mode"
./run_batch_switch.sh

echo "Running all instances on parallel uv"
./run_batch_uv_parallel
