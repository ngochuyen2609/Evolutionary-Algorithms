import numpy as np
import random
from .operators import sbx_crossover, gaussian_mutation, random_parents_mfea, polynomial_mutation
from .tasks import fitness_tsp, fitness_knapsack, fitness


def mfea_tsp_knapsack(dist_matrix, values, weights, capacity,
                      pop_size=50, rmp=0.2, 
                      patience=200, max_gens=10000):
    """
    Multifactorial Evolutionary Algorithm
    - Một quần thể duy nhất giải đồng thời TSP và Knapsack
    - Mỗi cá thể có skill factor = {0: TSP, 1: Knapsack}
    
    Pipeline:
    - Init population
    - Set up SKILL FACTOR 
    - Calc fitness for each task
    - Select parent
    - Crossover
    - Mutation
    - Set up SKILL FACTOR for child
    - Evaluate FITNESS for child
    - Union population (parent +child)
    - Ckeck end-condition
    """

    # Init population
    """
    Chỉ có 1 quần thể chung: mỗi cá thể là 1 vector X = [x₁, x₂, ..., x_D]
    TSP và Knapsack có D khác nhau
    → ta cần đệm (padding) hoặc cắt (truncate) sao cho cùng chiều D_common.
    """
    n_tsp  = dist_matrix.shape[0]
    n_knap = len(values)
    D = max(n_tsp, n_knap)
    population   = np.random.rand(pop_size, D)
    
    # Set up SKILL FACTOR  # 0 = TSP, 1 = Knapsack
    skill_factor = np.random.choice([0, 1], pop_size) 

    # Calc fitness for each task
    fitnesses = np.zeros(pop_size)
    for i, ind in enumerate(population):
        if skill_factor[i] == 0:
            fitnesses[i] = fitness_tsp(ind[:n_tsp], dist_matrix)
        else:
            fitnesses[i] = fitness_knapsack(ind[:n_knap], values, weights, capacity)

    # Vòng lặp tiến hóa
    best_tsp_fit   = np.max(fitnesses[skill_factor == 0]) if np.any(skill_factor == 0) else -np.inf
    best_knap_fit  = np.max(fitnesses[skill_factor == 1]) if np.any(skill_factor == 1) else -np.inf
    no_improve = 0
    gen = 0
    hist_tsp, hist_knap = [], []
    
    while no_improve < patience and gen < max_gens:
        gen += 1
        offspring, offspring_skill = [], []
        
        for _ in range(50):
            k = 5
            # chọn cha mẹ qua tournament
            (p1, sf1), (p2, sf2) = random_parents_mfea(population, skill_factor)
            # print(f"parent1:{fitness(p1,sf1,dist_matrix,values, weights, capacity)},{sf1}   parent2:{fitness(p2,sf2,dist_matrix,values, weights, capacity)},{sf2}")

            # lai ghép (cross-task nếu random < rmp)
            if sf1 != sf2 and random.random() > rmp:
                continue
            c1, c2 = sbx_crossover(p1, p2)

            # đột biến
            c1, c2 = polynomial_mutation(c1), polynomial_mutation(c2)

            offspring.append(c1)
            offspring.append(c2)
            offspring_skill.extend([
                random.choice([sf1, sf2]),
                random.choice([sf1, sf2])
            ])
            

        if len(offspring) == 0:
            # không sinh được con (rmp quá nhỏ) → bỏ qua thế hệ
            no_improve += 1
            continue

        offspring = np.array(offspring)
        # Set up SKILL FACTOR for child
        offspring_skill = np.array(offspring_skill)

        # Evaluate FITNESS for child
        offspring_fitness = np.zeros(len(offspring))
        for i, ind in enumerate(offspring):
            if offspring_skill[i] == 0:
                offspring_fitness[i] = fitness_tsp(ind[:n_tsp], dist_matrix)
            else:
                offspring_fitness[i] = fitness_knapsack(ind[:n_knap], values, weights, capacity)

        # Union (cha + con)
        population = np.vstack([population, offspring])
        fitnesses = np.hstack([fitnesses, offspring_fitness])
        skill_factor = np.hstack([skill_factor, offspring_skill])

        # Elitism per task
        new_pop, new_fit, new_sf = [], [], []
        half = pop_size // 2
        for sf in np.unique(skill_factor):
            mask = np.where(skill_factor == sf)[0]
            take = min(half, len(mask))
            top = mask[np.argsort(fitnesses[mask])[::-1][:take]]
            new_pop.extend(population[top])
            new_fit.extend(fitnesses[top])
            new_sf.extend(skill_factor[top])

        if len(new_pop) < pop_size:
            remain = pop_size - len(new_pop)
            rest_idx = np.argsort(fitnesses)[::-1][:remain]
            new_pop.extend(population[rest_idx])
            new_fit.extend(fitnesses[rest_idx])
            new_sf.extend(skill_factor[rest_idx])

        population = np.array(new_pop)
        fitnesses = np.array(new_fit)
        skill_factor = np.array(new_sf)

        # Kiểm tra cải thiện
        cur_best_tsp = np.max(fitnesses[skill_factor == 0]) if np.any(skill_factor == 0) else -np.inf
        cur_best_knap = np.max(fitnesses[skill_factor == 1]) if np.any(skill_factor == 1) else -np.inf
        hist_tsp.append(1/cur_best_tsp if cur_best_tsp != -np.inf else 0.0)
        hist_knap.append(-cur_best_knap if cur_best_knap != -np.inf else 0.0)

        improved = False
        if cur_best_tsp > best_tsp_fit:
            best_tsp_fit = cur_best_tsp
            improved = True
        if cur_best_knap > best_knap_fit:
            best_knap_fit = cur_best_knap
            improved = True

        no_improve = 0 if improved else no_improve + 1
        
        if gen % 10 == 0:
            print(f"[Gen {gen:04d}]  TSP={best_tsp_fit:.6f}  |  Knapsack={best_knap_fit:.6f}  |  no_improve={no_improve}")

    # Extract best individuals
    best_tsp_ind = population[np.argmax([
        fitness_tsp(ind[:n_tsp], dist_matrix) if sf == 0 else -np.inf
        for ind, sf in zip(population, skill_factor)
    ])] if np.any(skill_factor == 0) else None

    best_knap_ind = population[np.argmax([
        fitness_knapsack(ind[:n_knap], values, weights, capacity) if sf == 1 else -np.inf
        for ind, sf in zip(population, skill_factor)
    ])] if np.any(skill_factor == 1) else None

    print(f"\n Dừng sau {gen} thế hệ (no_improve={no_improve})")
    print(f"   Best TSP: {1/best_tsp_fit:.6f}")
    print(f"   Best Knapsack : {best_knap_fit:.6f}")

    return best_tsp_ind, best_knap_ind, hist_tsp, hist_knap
