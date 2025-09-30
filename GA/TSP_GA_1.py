import random
import time
import tsplib95

# ========================
# B1: Khởi tạo quần thể (mỗi cá thể là một hoán vị)
# ========================
def init_population(pop_size, num_genes):
    """Khởi tạo quần thể (mỗi cá thể là một hoán vị)"""
    population = []
    for _ in range(pop_size):
        individual = list(range(num_genes))
        random.shuffle(individual)
        population.append(individual)
    return population

# def init_population_by_greedy(pop_size):
    
# ========================
# B2: Chọn cha mẹ
# ========================
def selection_parent_Tournament(population, matrix, k=3):
    """Tournament: chọn random k cá thể -> lấy cá thể tốt nhất"""
    selected = random.sample(population, k)
    selected.sort(key=lambda ind: fitness(ind, matrix), reverse=True)
    return selected[0]

def selection_parent_Roulette(population, matrix):
    """
    Roulette Wheel Selection
    Tính fitness cho tất cả cá thể. (Nếu có giá trị âm/0) dịch và chuẩn hóa để tổng > 0.
    Tính tổng F = ∑ fi và xác suất tích lũy.
    Rút số ngẫu nhiên  r∈[0,1); tìm cá thể có tích lũy ≥ r
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

# ========================
# B3: Lai ghép
# ========================
def crossover_OX(parent1, parent2):
    """
    Order Crossover (OX)
    Chọn hai điểm cắt a, b.
    Copy đoạn [a:b] từ cha 1 sang con.
    Trong đoạn [a:b], tạo bản đồ ánh xạ giữa gene cha 1 và cha 2.
    Duyệt cha 2, điền gene còn lại vào con theo ánh xạ (tránh trùng lặp).
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None]*size
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
    Partially Mapped Crossover (PMX)
    Chọn ngẫu nhiên hai điểm cắt a, b trên chuỗi.
    Copy đoạn [a:b] từ cha 1 sang con.
    Duyệt qua cha 2, lấy các gene chưa có trong con, chèn lần lượt vào các ô còn trống (theo thứ tự).   
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))

    child = [None] * size
    # copy đoạn từ parent1
    child[a:b] = parent1[a:b]

    # tạo ánh xạ giữa cha1 và cha2 trong đoạn [a:b]
    mapping = {parent2[i]: parent1[i] for i in range(a, b)}
    mapping.update({parent1[i]: parent2[i] for i in range(a, b)})

    # điền phần còn lại từ parent2
    for i in range(size):
        if i >= a and i < b:
            continue
        candidate = parent2[i]
        # nếu candidate đã có trong đoạn copy thì thay theo ánh xạ cho đến khi hợp lệ
        while candidate in child[a:b]:
            candidate = mapping[candidate]
        child[i] = candidate

    return child


# ========================
# B3: Đột biến
# ========================
def mutation_Inversion(individual):
    """Đột biến Inversion: đảo ngược một đoạn gen"""
    a, b = sorted(random.sample(range(len(individual)), 2))
    individual[a:b] = reversed(individual[a:b])
    return individual

def mutation_swap(individual):
    """Đột biến Swap: hoán đổi 2 gene"""
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]
    return individual

# ========================
# Fitness
# ========================
def fitness(individual, matrix):
    """Tính độ thích nghi: nghịch đảo tổng chi phí tour"""
    cost = 0
    for i in range(len(individual) - 1):
        cost += matrix[individual[i]][individual[i+1]]
    cost += matrix[individual[-1]][individual[0]]  # quay về điểm đầu
    return 1 / (cost + 1e-9)

# ========================
# GA chính
# ========================
def GA(
    matrix,
    selection_parent: str,
    crossover: str,
    mutation: str,
    pop_size=50,
    crossover_rate=0.9,
    mutation_rate=0.1,
    time_limit_sec=250,
):
    """
    - B1: Khởi tạo quần thể
    - B2: Chọn cha mẹ (Tournament / Roulette / Random)
    - B3: Lai ghép (OX / PMX / Random) + Đột biến (Swap / Inversion / Random)
    - B4: Elitism: gộp cha+con, giữ top pop_size
    - B5: Dừng theo time_limit_sec
    """
    n = len(matrix)
    population = init_population(pop_size, n)
    best = max(population, key=lambda ind: fitness(ind, matrix))

    # Chuẩn hóa tham số lựa chọn
    selection_parent = selection_parent.strip().lower()
    crossover = crossover.strip().lower()
    mutation = mutation.strip().lower()

    t0 = time.time()
    while time.time() - t0 < time_limit_sec:
        new_pop = []
        for _ in range(pop_size):
            # --------- B2: chọn cha mẹ ----------
            if selection_parent == "tournament":
                p1 = selection_parent_Tournament(population, matrix)
                p2 = selection_parent_Tournament(population, matrix)
            elif selection_parent == "roulette":
                p1 = selection_parent_Roulette(population, matrix)
                p2 = selection_parent_Roulette(population, matrix)
            else:  # "random"
                if random.random() < 0.5:
                    p1 = selection_parent_Tournament(population, matrix)
                else:
                    p1 = selection_parent_Roulette(population, matrix)
                if random.random() < 0.5:
                    p2 = selection_parent_Tournament(population, matrix)
                else:
                    p2 = selection_parent_Roulette(population, matrix)

            # --------- B3: lai ghép ----------
            if random.random() < crossover_rate:
                if crossover == "ox":
                    child = crossover_OX(p1, p2)
                elif crossover == "pmx":
                    child = crossover_PMX(p1, p2)
                else:  # "random"
                    child = crossover_OX(p1, p2) if random.random() < 0.5 else crossover_PMX(p1, p2)
            else:
                child = p1[:]

            # --------- B3: đột biến ----------
            if random.random() < mutation_rate:
                if mutation == "swap":
                    child = mutation_swap(child)
                elif mutation == "inversion":
                    child = mutation_Inversion(child)
                else:  # "random"
                    child = mutation_swap(child) if random.random() < 0.5 else mutation_Inversion(child)

            new_pop.append(child)

        # --------- B4: elitism ----------
        population = population + new_pop
        population.sort(key=lambda ind: fitness(ind, matrix), reverse=True)
        population = population[:pop_size]

        if fitness(population[0], matrix) > fitness(best, matrix):
            best = population[0]

    return best, 1/fitness(best,matrix)

# ========================
# Chạy thử
# ========================
if __name__ == "__main__":
    problem = tsplib95.load("data/eil51.tsp")
    print("Số thành phố:", problem.dimension)

    # Tạo ma trận khoảng cách theo thứ tự node 1..n -> index 0..n-1
    nodes = list(problem.get_nodes())              # [1..51]
    n = len(nodes)
    matrix = [[problem.get_weight(i, j) for j in nodes] for i in nodes]

    # Thử cấu hình 1
    best_path, best_cost = GA(
        matrix,
        selection_parent="Tournament",
        crossover="OX",
        mutation="Swap",
        pop_size=50,
        crossover_rate=0.9,
        mutation_rate=0.1,
    )
    print("selection_parent: Tournament, crossover: OX, mutation: Swap")
    print("Best path (index 0-based):", best_path)
    print("Best path (city IDs 1-based):", [x+1 for x in best_path])
    print("Best cost:", best_cost)

    # Thử cấu hình 2
    best_path, best_cost = GA(
        matrix,
        selection_parent="Roulette",
        crossover="PMX",
        mutation="Inversion",
        pop_size=50,
        crossover_rate=0.9,
        mutation_rate=0.1,
    )
    print("\nselection_parent: Roulette, crossover: PMX, mutation: Inversion")
    print("Best path (index 0-based):", best_path)
    print("Best path (city IDs 1-based):", [x+1 for x in best_path])
    print("Best cost:", best_cost)