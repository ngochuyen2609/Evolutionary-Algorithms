from pathlib import Path
import sys
from mfea.mfea_core import mfea_tsp_knapsack
import numpy as np
import tsplib95
import os

from mfea.tasks import decode_knapsack_fill, decode_tsp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)
    
from utils.plot import plot_scores

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

    n = int(lines[0])
    capacity = float(lines[1])

    for i in range(n):
        v, w = map(float, lines[2 + i].split())
        weights.append(v)
        values.append(w)

    return np.array(values, dtype=float), np.array(weights, dtype=float), float(capacity)

# Chạy MFEA
if __name__ == "__main__":
    # ---- TSP ----
    dist_matrix = tsp_data("data/TSP/eil51.tsp")

    # ---- Knapsack ----
    values, weights, capacity = knapsack_data("data/Knapsack/kp.kp")

    # ---- Chạy MFEA ----
    best_tsp, best_knap, hist_tsp, hist_knap = mfea_tsp_knapsack(
        dist_matrix, 
        values, weights, capacity,
        pop_size=50, rmp=0.3
    )

    print("\n Best TSP Path:", decode_tsp(best_tsp, dist_matrix))
    print(" Best Knapsack:", decode_knapsack_fill(best_knap, values, weights, capacity) )

    plot_scores(hist_knap, title="Knapsack — value per generation (MFEA)", filename="images/Knapsack_scores.png")
    plot_scores(hist_tsp, title="TSP — tour length per generation (MFEA)", filename="images/TSP_scores.png")
