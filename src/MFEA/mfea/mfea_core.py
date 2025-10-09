import numpy as np
import random
from .operators import sbx_crossover, gaussian_mutation
from .tasks import fitness_tsp, fitness_knapsack


def mfea_tsp_knapsack(dist_matrix, values, weights, capacity,
                      pop_size=50, gens=200, rmp=0.3):
    """
    Multifactorial Evolutionary Algorithm
    - Một quần thể duy nhất giải đồng thời TSP và Knapsack
    - Mỗi cá thể có skill factor = {0: TSP, 1: Knapsack}
    """

    n = len(values)
    population = np.random.rand(pop_size, n)
    skill_factor = np.random.choice([0, 1], pop_size)  # 0 = TSP, 1 = Knapsack

    # --- Khởi tạo fitness ---
    fitnesses = np.zeros(pop_size)
    for i, ind in enumerate(population):
        if skill_factor[i] == 0:
            fitnesses[i] = fitness_tsp(ind, dist_matrix)
        else:
            fitnesses[i] = fitness_knapsack(ind, values, weights, capacity)

    # --- Vòng lặp tiến hóa ---
    for g in range(gens):
        offspring = []
        offspring_skill = []

        # === Tạo offspring ===
        for _ in range(pop_size // 2):
            i, j = np.random.choice(pop_size, 2, replace=False)
            p1, p2 = population[i], population[j]
            sf1, sf2 = skill_factor[i], skill_factor[j]

            # Chỉ lai khác task khi random < rmp
            if sf1 != sf2 and random.random() > rmp:
                continue

            c1, c2 = sbx_crossover(p1, p2)
            c1, c2 = gaussian_mutation(c1), gaussian_mutation(c2)
            offspring.append(c1)
            offspring.append(c2)
            offspring_skill.extend([random.choice([sf1, sf2]),
                                    random.choice([sf1, sf2])])

        offspring = np.array(offspring)
        offspring_skill = np.array(offspring_skill)

        # === Đánh giá fitness offspring ===
        offspring_fitness = np.zeros(len(offspring))
        for i, ind in enumerate(offspring):
            if offspring_skill[i] == 0:
                offspring_fitness[i] = fitness_tsp(ind, dist_matrix)
            else:
                offspring_fitness[i] = fitness_knapsack(ind, values, weights, capacity)

        # === Hợp quần thể (cha + con) ===
        population = np.vstack([population, offspring])
        fitnesses = np.hstack([fitnesses, offspring_fitness])
        skill_factor = np.hstack([skill_factor, offspring_skill])

        # === Chọn lọc tự nhiên (elitism) ===
        idxs = []
        for sf in [0, 1]:
            mask = np.where(skill_factor == sf)[0]
            if len(mask) > 0:
                top = mask[np.argsort(fitnesses[mask])[::-1][:pop_size // 2]]
                idxs.extend(top)

        population = population[idxs]
        fitnesses = fitnesses[idxs]
        skill_factor = skill_factor[idxs]

        # === Theo dõi kết quả ===
        if g % 20 == 0:
            best_tsp = max(fitnesses[skill_factor == 0]) if np.any(skill_factor == 0) else 0
            best_knap = max(fitnesses[skill_factor == 1]) if np.any(skill_factor == 1) else 0
            print(f"[Gen {g:03d}] TSP best={best_tsp:.4f} | Knapsack best={best_knap:.4f}")

    # === Trích xuất nghiệm tốt nhất ===
    best_tsp_ind = population[np.argmax([
        fitness_tsp(ind, dist_matrix) if sf == 0 else 0
        for ind, sf in zip(population, skill_factor)
    ])]

    best_knap_ind = population[np.argmax([
        fitness_knapsack(ind, values, weights, capacity) if sf == 1 else 0
        for ind, sf in zip(population, skill_factor)
    ])]

    return best_tsp_ind, best_knap_ind
