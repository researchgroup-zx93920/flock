nv-nsight-cu-cli \
  -o ./profiles/profile_flock \
  --kernel-id ::find_loops_and_savings:1 \
  ./bin/flock -i  ./tests/test_TransportModel_500_500_1.dat  -a parallel_uv
# ex.
# ../build/exe/src/main.cu.exe -g ../../dataset/gbin/cit-Patents_adj.bel -t ../../dataset/templates/mtx/cq8m1_template.mtx -o full -d 3 -m sgm -p node
#  --section ".*" \  
 
