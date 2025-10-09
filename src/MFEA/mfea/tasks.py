import numpy as np

# =========================
# TASK 1: Traveling Salesman Problem
# =========================
def tsp_distance(path, dist_matrix):
    """Tính tổng quãng đường của chu trình"""
    total = 0
    for i in range(len(path)):
        total += dist_matrix[path[i - 1]][path[i]]
    return total

def decode_tsp(gen):
    """Chuyển vector thực [0,1]^n -> thứ tự thăm các thành phố"""
    return np.argsort(gen)

def fitness_tsp(gen, dist_matrix):
    path = decode_tsp(gen)
    return 1.0 / (tsp_distance(path, dist_matrix) + 1e-9)


# =========================
# TASK 2: 0/1 Knapsack Problem
# =========================
def fitness_knapsack(gen, values, weights, capacity):
    bits = (gen > 0.5).astype(int)
    total_w = np.sum(weights * bits)
    total_v = np.sum(values * bits)
    if total_w > capacity:
        return 0.0
    return total_v
