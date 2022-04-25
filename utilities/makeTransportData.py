__version__="0.01"
__author__="Mohit Mahajan"

"""
This utility helps to create test files for testing the transportation simplex,
There's no prefix configuration, Demand, Supply and costs are geenrated at random. Configuration has been provided to 
switch between balanced and imbalanced problem instances (balanced means demand=supply)

Generated costs, demands and supplies are integer

The output .dat will be spit in the pwd, it'll need to be moved to the appropriate directory for specific purpose
The template of the dat file is as follows:: 

------ TransportModel<something>.dat --------

{No. of supplies} {No. of demands}
{supply values}
{demand values}
{cost_matrix indexed 1D}

------       END OF DAT FILE       --------

Note : To find the appropriate value in cost matrix for supply i to demand j - 
Use indexing formula - {i}*{No. of demands} + {j}  | Tip: Check the reader function in src 
"""

# import gurobipy as grb
import random

# >>>>>>>>>>>> Set Configuration Here >>>>>>>>>>>>>>
randomize = True # Creates a new instance on every run
r_seed = 1 # related to randomization

exportEx = "dat" # Export file extension (.dat)
balancedProblem = True
assignmnetCase = False

matrix_demands = 50
matrix_supplies = 50

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# DO NOT CHANGE ANYTHING BEYOND THIS LINE
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Verify Configuration
assert isinstance(randomize, bool)
assert isinstance(r_seed, int), "param: r_seed must be a valid integer"
assert isinstance(exportEx, str), "param: exporEx must be a valid string"
assert exportEx in ["dat"], "param: exporEx must be in [lp, mps]"
assert isinstance(balancedProblem, bool)
assert isinstance(matrix_demands, int), "param: matrix_demands must be a valid integer"
assert isinstance(matrix_supplies, int), "param: matrix_supplies must be a valid integer"
if assignmnetCase:
    assert (matrix_demands == matrix_supplies), "In an assignment problem no. of supplies and demand are equal!"

print(f"Demands = {matrix_demands}")
print(f"Supplies = {matrix_supplies}")

# Create a gurobi problem instance thorugh parameters >>
if not randomize:
    random.seed(r_seed)


vars = []
print("Creating Cost Matrix variables")
for i in range(matrix_supplies):
    varRows = []
    for j in range(matrix_demands):
        varRows.append(round(random.uniform(100,999),2))
    vars.append(varRows)


print("Generating Demand and Supplies")
if not assignmnetCase:
    demands = [random.randint(10,99) for i in range(matrix_demands)]
    supplies = [random.randint(10,99) for j in range(matrix_supplies)]
else:
    demands = [1 for i in range(matrix_demands)]
    supplies = [1 for j in range(matrix_supplies)]

# Balance Demands and Supplies
def runConsumption(diff, const_diff, augumented):
    i = 0
    while diff > 0:
        consump = random.randint(0,const_diff)
        if consump > diff:
            consump = diff
        augumented[i] += consump
        diff = diff - consump
        i += 1
    return augumented

if balancedProblem:
    print("Required a balanced Demand <-> Supply")
    print("\tBalanacing the problem instance ... ")
    totalD = sum(demands)
    totalS = sum(supplies)

    if totalD > totalS:
        const_diff = totalD - totalS
        diff = totalD - totalS
        supplies = runConsumption(diff, const_diff,supplies)
        

    if totalS > totalD:
        const_diff = totalS - totalD
        diff = totalS - totalD
        demands = runConsumption(diff, const_diff,demands)
    
    assert sum(demands) == sum(supplies)

print("Exporting File .. ")
outfile = open(f"../tests/test_TransportModel_{matrix_supplies}_{matrix_demands}_{r_seed}.{exportEx}", "w+")
outfile.write(f"{matrix_supplies} {matrix_demands}\n")
for s in supplies:
    outfile.write(f"{s}\t")
outfile.write("\n")
for d in demands:
    outfile.write(f"{d}\t")
outfile.write("\n")
for i in range(matrix_supplies):
    for j in range(matrix_demands):
        outfile.write(f"{vars[i][j]}\n")
outfile.close()
print("Dump Complete!")
