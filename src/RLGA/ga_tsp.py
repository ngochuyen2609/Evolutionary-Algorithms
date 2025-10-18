import random
from tsp_utils import tour_length, nearest_neighbor_seed, two_opt_local_search

def fitness_perm(ind, dist):
    return 1.0 / (tour_length(ind, dist) + 1e-9)

def tournament_select(pop, dist, k=5):
    cand = random.sample(pop, k)
    cand.sort(key=lambda t: fitness_perm(t, dist), reverse=True)
    return cand[0]

def crossover_OX(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b] = p1[a:b]
    pos = b
    for x in p2:
        if x not in child:
            if pos == n: pos = 0
            child[pos] = x; pos += 1
    return child

def mutate_swap(ind, rate=0.2):
    if random.random() < rate:
        a, b = random.sample(range(len(ind)), 2)
        ind[a], ind[b] = ind[b], ind[a]
    return ind

def GA_tsp(dist, init_pop, pop_size=80, gens=600, cx_rate=0.9,
           mut_rate=0.2, use_2opt_every=30, two_opt_swaps=80):
    # khởi tạo
    pop = [t[:] for t in init_pop[:pop_size]]
    n = dist.shape[0]
    while len(pop) < pop_size:
        start = random.randrange(n)
        pop.append(nearest_neighbor_seed(n, dist, start))

    best = min(pop, key=lambda t: tour_length(t, dist))
    best_cost = tour_length(best, dist)
    history = [best_cost]

    for g in range(1, gens + 1):
        new_pop = []
        for _ in range(pop_size):
            p1 = tournament_select(pop, dist)
            p2 = tournament_select(pop, dist)
            if random.random() < cx_rate:
                c = crossover_OX(p1, p2)
            else:
                c = p1[:]
            if random.random() < mut_rate:
                c = mutate_swap(c, rate=1.0)
            new_pop.append(c)

        pop = pop + new_pop

        if use_2opt_every and g % use_2opt_every == 0:
            for i in range(len(pop)):
                pop[i] = two_opt_local_search(pop[i], dist, max_swaps=two_opt_swaps)

        pop.sort(key=lambda t: tour_length(t, dist))
        pop = pop[:pop_size]

        cur = tour_length(pop[0], dist)
        if cur < best_cost:
            best = pop[0][:]
            best_cost = cur
        history.append(best_cost)

    return best, best_cost, history
