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

import random
import os.path
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# >>>>>>>>>>>> Set Configuration Here >>>>>>>>>>>>>>
randomize = True # Creates a new instance on every run
r_seed = 1 # related to randomization

exportEx = "dat" # Export file extension (.dat)
balancedProblem = True
assignmnetCase = False
data = "usazip" # None = Complete Random or "usazip" : Using realiztic using USA Zip codes


matrix_demands = 10000
matrix_supplies = 1000

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# DO NOT CHANGE ANYTHING BEYOND THIS LINE
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

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
if data == "usazip":
    if not os.path.isfile("./uszips.csv"):
        print("""Please download the USA Zip code dabase from the following link and run this script again: 
        https://simplemaps.com/data/us-zips#:~:text=Includes%20data%20from%202020%20Census,geocoded%20to%20latitude%20and%20longitude
        """)
        exit(-1)
    df = pd.read_csv("./uszips.csv")[["zip", "lat", "lng"]]
    latlongdict = df.set_index("zip").to_dict(orient="index")
    supply_points = random.sample(list(latlongdict.keys()), matrix_supplies)
    demand_points = random.sample(list(latlongdict.keys()), matrix_demands)
    for i in range(matrix_supplies):
        varRows = []
        for j in range(matrix_demands):
            varRows.append(round(haversine(latlongdict[supply_points[i]]["lng"], latlongdict[supply_points[i]]["lat"], 
                                            latlongdict[demand_points[j]]["lng"], latlongdict[demand_points[j]]["lat"])))
        vars.append(varRows)
else:
    for i in range(matrix_supplies):
        varRows = []
        for j in range(matrix_demands):
            varRows.append(round(random.uniform(100,999),2))
        vars.append(varRows)

print("Generating Demand and Supplies")
if not assignmnetCase:
    demands = [random.randint(1,5) for i in range(matrix_demands)]
    supplies = [random.randint(100,999) for j in range(matrix_supplies)]
else:
    demands = [1 for i in range(matrix_demands)]
    supplies = [1 for j in range(matrix_supplies)]

# Balance Demands and Supplies
def runConsumption(diff, const_diff, augumented, rng):
    i = random.choice(range(rng))
    while diff > 0:
        consump = min(random.randint(0,const_diff), diff)
        if consump < augumented[i]:
            augumented[i] -= consump    
            diff = diff - consump
        i = random.choice(range(rng))
    return augumented

if balancedProblem:
    print("Required a balanced Demand <-> Supply")
    print("\tBalanacing the problem instance ... ")
    totalD = sum(demands)
    totalS = sum(supplies)

    if totalD > totalS:
        const_diff = totalD - totalS
        diff = totalD - totalS
        demands = runConsumption(diff, const_diff, demands, matrix_demands)
        
    if totalS > totalD:
        const_diff = totalS - totalD
        diff = totalS - totalD
        supplies = runConsumption(diff, const_diff, supplies, matrix_supplies)
    
    assert sum(demands) == sum(supplies)

print("Exporting File .. ")
if data=="usazip":
    prefix = "uszip_"
else:
    prefix = None
outfile = open(f"../data/{prefix}TransportModel_{matrix_supplies}_{matrix_demands}_{r_seed}.{exportEx}", "w+")
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
