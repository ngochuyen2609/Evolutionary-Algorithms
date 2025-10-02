import random
from matplotlib import pyplot as plt
import tsplib95

# ---------- B1: Khởi tạo quần thể (mỗi cá thể là một hoán vị) ----------
def init_population(pop_size, num_genes):
    """Khởi tạo quần thể (mỗi cá thể là một hoán vị)"""
    population = []
    for _ in range(pop_size):
        individual = list(range(num_genes))
        random.shuffle(individual)
        population.append(individual)
    return population


# ---------- B2: Chọn cha mẹ ----------
def selection_parent_Tournament(population, matrix, k=5):
    """Tournament: chọn random k cá thể -> lấy cá thể tốt nhất"""
    selected = random.sample(population, k)
    selected.sort(key=lambda ind: fitness(ind, matrix), reverse=True)
    return selected[0]

def selection_parent_Roulette(population, matrix):
    """
    Roulette Wheel Selection
    - Tính fitness cho tất cả cá thể.
    - Tính tổng F = ∑ fi và xác suất tích lũy.
    - Rút số ngẫu nhiên r ∈ [0, F), chọn cá thể có tích lũy ≥ r.
    """
    fitnesses = [fitness(ind, matrix) for ind in population]
    total_fit = sum(fitnesses)
    pick = random.uniform(0, total_fit)
    current = 0
    for ind, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return ind
    return population[-1]


# ---------- B3: Lai ghép ----------
def crossover_OX(parent1, parent2):
    """
    Order Crossover (OX) — trả về 1 con:
    B1: Chọn hai điểm cắt a < b
    B2: Copy đoạn parent1[a:b] sang child[a:b]
    B3: Duyệt parent2 theo thứ tự; nếu gene chưa có trong child thì chèn vào vị trí trống (theo vòng)
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = parent1[a:b]
    pos = b
    for x in parent2:
        if x not in child:
            if pos == size:
                pos = 0
            child[pos] = x
            pos += 1
    return child

def crossover_PMX(parent1, parent2):
    """
    Partially Mapped Crossover (PMX) — trả về 1 con:
    B1: Chọn hai điểm cắt a < b
    B2: Copy đoạn parent1[a:b] -> child[a:b]
    B3: Tạo mapping giữa parent1 và parent2 trong đoạn [a:b]
    B4: Với vị trí ngoài đoạn [a:b], fill bằng parent2[i], nhưng nếu trùng với vùng copy thì dịch theo mapping cho tới khi hợp lệ
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[a:b] = parent1[a:b]

    mapping = {parent2[i]: parent1[i] for i in range(a, b)}
    mapping.update({parent1[i]: parent2[i] for i in range(a, b)})

    for i in range(size):
        if a <= i < b:
            continue
        candidate = parent2[i]
        while candidate in child[a:b]:
            candidate = mapping[candidate]
        child[i] = candidate
    return child


# ---------- B4: Đột biến ----------
def mutation_Inversion(individual):
    """Đột biến Inversion: đảo ngược một đoạn gen"""
    a, b = sorted(random.sample(range(len(individual)), 2))
    individual[a:b] = reversed(individual[a:b])
    return individual

def mutation_Swap(individual):
    """Đột biến Swap: hoán đổi 2 gene"""
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]
    return individual


# ---------- Fitness ----------
def fitness(individual, matrix):
    cost = 0
    for i in range(len(individual) - 1):
        cost += matrix[individual[i]][individual[i+1]]
    cost += matrix[individual[-1]][individual[0]]
    return 1 / (cost + 1e-9)


# ---------- GA chính ----------
def GA(
    matrix,
    selection_parent: str,
    crossover: str,
    mutation: str,
    pop_size=50,
    crossover_rate=0.9,
    mutation_rate=0.1,
    patience=100
):
    """
    - B1: Khởi tạo quần thể
    - B2: Chọn cha mẹ (Tournament / Roulette)
    - B3: Lai ghép (OX / PMX) + Đột biến (Swap / Inversion)
    - B4: Elitism: gộp cha+con, giữ top pop_size
    - B5: Dừng khi không cải thiện sau `patience` thế hệ
    """
    n = len(matrix)
    population = init_population(pop_size, n)
    best = max(population, key=lambda ind: fitness(ind, matrix))
    best_cost = 1 / fitness(best, matrix)

    history = [best_cost]
    no_improve = 0
    g = 0

    while no_improve < patience:
        g += 1
        new_pop = []

        for _ in range(pop_size):
            # chọn cha mẹ
            if selection_parent == "tournament":
                p1 = selection_parent_Tournament(population, matrix)
                p2 = selection_parent_Tournament(population, matrix)
            elif selection_parent == "roulette":
                p1 = selection_parent_Roulette(population, matrix)
                p2 = selection_parent_Roulette(population, matrix)

            # lai ghép
            if random.random() < crossover_rate:
                if crossover == "ox":
                    child = crossover_OX(p1, p2)
                elif crossover == "pmx":
                    child = crossover_PMX(p1, p2)
            else:
                child = p1[:]

            # đột biến
            if random.random() < mutation_rate:
                if mutation == "swap":
                    child = mutation_Swap(child)
                elif mutation == "inversion":
                    child = mutation_Inversion(child)

            new_pop.append(child)

        # elitism
        population = population + new_pop
        population.sort(key=lambda ind: fitness(ind, matrix), reverse=True)
        population = population[:pop_size]

        # cập nhật best
        current_best = population[0]
        current_cost = 1 / fitness(current_best, matrix)
        history.append(current_cost)

        if current_cost < best_cost:
            best = current_best
            best_cost = current_cost
            no_improve = 0
        else:
            no_improve += 1

        if g % 20 == 0:
            print(f"Gen {g}: cost = {best_cost}")

    return best, best_cost, history


# ---------- Test ----------
if __name__ == "__main__":
    problem = tsplib95.load("data/eil51.tsp")
    print("Số thành phố:", problem.dimension)

    nodes = list(problem.get_nodes())
    n = len(nodes)
    matrix = [[problem.get_weight(i, j) for j in nodes] for i in nodes]

    best_path, best_cost, history = GA(
        matrix,
        selection_parent="tournament",
        crossover="ox",
        mutation="swap",
        pop_size=50,
        crossover_rate=0.9,
        mutation_rate=0.1,
        patience=200
    )

    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.title("GA - Lịch sử Cost")
    plt.show()

    coords = [problem.node_coords[i] for i in nodes]
    x = [coords[i][0] for i in best_path] + [coords[best_path[0]][0]]
    y = [coords[i][1] for i in best_path] + [coords[best_path[0]][1]]
    plt.plot(x, y, 'o-r')
    plt.title(f"TSP Best Tour (Cost={best_cost})")
    plt.show()

    best_path, best_cost, history = GA(
        matrix,
        selection_parent="roulette",
        crossover="pmx",
        mutation="inversion",
        pop_size=50,
        crossover_rate=0.9,
        mutation_rate=0.1,
        patience=200
    )

    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.title("GA - Lịch sử Cost")
    plt.show()

    coords = [problem.node_coords[i] for i in nodes]
    x = [coords[i][0] for i in best_path] + [coords[best_path[0]][0]]
    y = [coords[i][1] for i in best_path] + [coords[best_path[0]][1]]
    plt.plot(x, y, 'o-r')
    plt.title(f"TSP Best Tour (Cost={best_cost})")
    plt.show()