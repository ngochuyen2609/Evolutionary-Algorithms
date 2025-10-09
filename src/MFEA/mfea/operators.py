import numpy as np
import random

# ===== Simulated Binary Crossover (SBX) =====
def sbx_crossover(p1, p2, eta=15):
    u = np.random.rand(len(p1))
    beta = np.where(u <= 0.5, (2*u)**(1/(eta+1)), (1/(2*(1-u)))**(1/(eta+1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)

# ===== Gaussian Mutation =====
def gaussian_mutation(ind, sigma=0.1, pm=0.2):
    for i in range(len(ind)):
        if random.random() < pm:
            ind[i] += np.random.normal(0, sigma)
    return np.clip(ind, 0, 1)
