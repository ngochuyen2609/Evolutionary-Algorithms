from mfea.mfea_core import mfea_tsp_knapsack
import numpy as np
import tsplib95
import os
from mfea.tasks import decode_knapsack_fill, decode_tsp

def tsp_data(path):
    problem = tsplib95.load(path)
    print("Số thành phố:", problem.dimension)

    nodes = list(problem.get_nodes())
    matrix = [[problem.get_weight(i, j) for j in nodes] for i in nodes]
    return np.array(matrix)

def knapsack_data(path):
    values, weights = [], []
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        capacity = float(lines[1])

        n = int(lines[1])                   
        for line in lines[2:n+1]:
            v, w = map(float, line.split())
            values.append(v)
            weights.append(w)
    print(len(values))
        
    return np.array(values), np.array(weights), capacity

# Chạy MFEA
if __name__ == "__main__":
    # ---- TSP ----
    dist_matrix = tsp_data("data/TSP/eil51.tsp")

    # ---- Knapsack ----
    values, weights, capacity = knapsack_data("data/Knapsack/kp.kp")

    # ---- Chạy MFEA ----
    best_tsp, best_knap = mfea_tsp_knapsack(
        dist_matrix, 
        values, weights, capacity,
        pop_size=50, rmp=0.3
    )

    print("\n Best TSP Path:", decode_tsp(best_tsp, dist_matrix))
    print(" Best Knapsack:", decode_knapsack_fill(best_knap, values, weights, capacity) )
