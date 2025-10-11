import numpy as np
import random
from .tasks import fitness_tsp, fitness_knapsack

# Tournament
def tournament_selection_mfea(population, skill_factor, k,
                              dist_matrix, values, weights, capacity,
                              n_tsp, n_knap):
    """
    Trả về (parent_vector, parent_skill)
    """
    idxs = np.random.choice(len(population), size=k, replace=False)

    best_idx, best_fit = None, -np.inf
    for idx in idxs:
        ind = population[idx]
        sf  = skill_factor[idx]
        fit = fitness_tsp(ind[:n_tsp], dist_matrix) if sf == 0 \
              else fitness_knapsack(ind[:n_knap], values, weights, capacity)
        if fit > best_fit:
            best_fit, best_idx = fit, idx

    return population[best_idx], skill_factor[best_idx]

# Simulated Binary Crossover (SBX)
def sbx_crossover(p1, p2, eta=15):
    u = np.random.rand(len(p1))
    beta = np.where(u <= 0.5, (2*u)**(1/(eta+1)), (1/(2*(1-u)))**(1/(eta+1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)

# Gaussian Mutation
def gaussian_mutation(ind, sigma=0.1, pm=0.2):
    for i in range(len(ind)):
        if random.random() < pm:
            ind[i] += np.random.normal(0, sigma)
    return np.clip(ind, 0, 1)
