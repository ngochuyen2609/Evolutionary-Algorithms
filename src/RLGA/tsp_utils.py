import numpy as np
import tsplib95
import random

def load_tsplib_tsp(path):
    problem = tsplib95.load(path)
    nodes = list(problem.get_nodes())  # 1..n
    n = len(nodes)
    dist = np.zeros((n, n), dtype=float)
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            dist[i, j] = problem.get_weight(a, b)
    # coords (n,2) nếu có
    coords = None
    if hasattr(problem, "node_coords") and problem.node_coords:
        coords = np.array([problem.node_coords[i] for i in nodes], dtype=float)
    return dist, coords

def tour_length(tour, dist):
    n = len(tour)
    s = 0.0
    for i in range(n):
        a, b = tour[i], tour[(i+1) % n]
        s += dist[a, b]
    return s

def nearest_neighbor_seed(n, dist, start=0):
    unvis = set(range(n))
    tour = [start]
    unvis.remove(start)
    cur = start
    while unvis:
        nxt = min(unvis, key=lambda j: dist[cur, j])
        tour.append(nxt)
        unvis.remove(nxt)
        cur = nxt
    return tour

def _two_opt_once(tour, dist):
    n = len(tour)
    best_delta = 0.0
    best_i, best_k = None, None
    for i in range(n-1):
        for k in range(i+1, n):
            a, b = tour[i], tour[(i+1) % n]
            c, d = tour[k], tour[(k+1) % n]
            delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
            if delta < 0:
                best_delta = delta; best_i, best_k = i, k
    if best_i is not None:
        i, k = best_i, best_k
        tour[i+1:k+1] = reversed(tour[i+1:k+1])
        return True
    return False

def two_opt_local_search(tour, dist, max_swaps=200):
    swaps = 0
    while swaps < max_swaps and _two_opt_once(tour, dist):
        swaps += 1
    return tour
