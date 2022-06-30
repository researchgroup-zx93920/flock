#!/bin/sh
# make clean
# make
nv-nsight-cu-cli \
  -o ./profiles/profile_bfs9 \
  --kernel-id ::computeReducedCosts:4 \
  --set full \
  --target-processes all \
  /home/mohitm3/workspace/flock/bin/flock -i /home/mohitm3/workspace/flock/data/TransportModel_500_500_1_equalityConstr.dat -a parallel_uv

# ex.
# ../build/exe/src/main.cu.exe -g ../../dataset/gbin/cit-Patents_adj.bel -t ../../dataset/templates/mtx/cq8m1_template.mtx -o full -d 3 -m sgm -p node
#  --section ".*" \  
# nv-nsight-cu-cli -o profile_bfs --sections ".*" --kernel-id::update_distance_path_and_create_next_frontier:4 