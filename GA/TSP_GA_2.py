import random
import math
import numpy as np
import tsplib95
import matplotlib.pyplot as plt

# -------- Encoding/Decoding ----------
def decode_tour(individual):
    """
    Giải mã một cá thể (vector số thực) thành tour:
    - Mỗi gene là số thực trong [0,1]
    - argsort(individual) -> sắp xếp chỉ số gene theo giá trị tăng dần
    """
    order = np.argsort(individual)
    return order.tolist()

def tour_cost(tour, matrix):
    cost = 0
    for i in range(len(tour) - 1):
        cost += matrix[tour[i]][tour[i+1]]
    cost += matrix[tour[-1]][tour[0]]
    return cost

def fitness(individual, matrix):
    tour = decode_tour(individual)
    return 1.0 / (tour_cost(tour, matrix) + 1e-9)

# -------- Crossover ----------
def sbx_crossover(p1, p2, eta_c=2):
    """
    Simulated Binary Crossover (SBX)
    Trong GA nhị phân, crossover lấy một đoạn từ cha này và ghép vào cha kia.
    Với mã hóa số thực, không có bit 0/1 → ta mô phỏng bằng cách sinh ra các giá trị con quanh khoảng giữa hai cha.
    1. Với từng gene, xác suất 0.5 chọn để crossover
    2. Tính giá trị trung gian x1, x2 giữa p1[i], p2[i]
    3. Sinh ra beta và betaq theo công thức SBX
    4. Tạo 2 con lai c1, c2 quanh đoạn [x1, x2]
    """
    n = len(p1)
    c1, c2 = np.copy(p1), np.copy(p2)
    for i in range(n):
        if random.random() <= 0.5:
            if abs(p1[i] - p2[i]) > 1e-14:
                x1 = min(p1[i], p2[i])
                x2 = max(p1[i], p2[i])
                rand = random.random()

                # Tạo beta và betaq
                beta = 1.0 + (2.0 * (x1 - 0.0) / (x2 - x1))
                alpha = 2.0 - pow(beta, -(eta_c+1))
                if rand <= 1.0/alpha:
                    betaq = pow(rand * alpha, 1.0/(eta_c+1))
                else:
                    betaq = pow(1.0/(2.0 - rand*alpha), 1.0/(eta_c+1))

                c1[i] = 0.5*((x1+x2) - betaq*(x2-x1))

                beta = 1.0 + (2.0 * (1.0 - x2) / (x2 - x1))
                alpha = 2.0 - pow(beta, -(eta_c+1))
                if rand <= 1.0/alpha:
                    betaq = pow(rand * alpha, 1.0/(eta_c+1))
                else:
                    betaq = pow(1.0/(2.0 - rand*alpha), 1.0/(eta_c+1))

                c2[i] = 0.5*((x1+x2) + betaq*(x2-x1))

                # Giới hạn trong [0,1]
                c1[i] = min(max(c1[i], 0.0), 1.0)
                c2[i] = min(max(c2[i], 0.0), 1.0)
    return c1, c2

# --------- Mutation -----------
def polynomial_mutation(ind, eta_m=20, pm=None):
    """
    Polynomial Mutation
    Trong GA nhị phân: đột biến = lật bit (0 → 1 hoặc 1 → 0).
    Với mã hóa số thực: ta không lật mà dịch chuyển giá trị gene một chút, theo một phân phối có kiểm soát.  => khám phá cục bộ quanh cha.
    
    eta_m: chỉ số phân tán (distribution index).
    eta_m lớn → bước đột biến nhỏ (gene thay đổi ít).
    eta_m nhỏ → bước đột biến lớn (gene thay đổi mạnh).
    pm: xác suất mỗi gene đột biến (thường = 1/n). Nếu cá thể có n gene → trung bình 1 gene sẽ bị đột biến mỗi lần.
    
    1. Với từng gene, xác suất pm để đột biến
    2. Tính delta1, delta2 = khoảng cách tới biên trái/phải
    3. Sinh số ngẫu nhiên rand, xác định hướng biến đổi
    4. Tính deltaq dựa trên phân phối đa thức
    """
    n = len(ind)
    if pm is None:
        pm = 1.0/n
    for i in range(n):
        if random.random() < pm:  # xác suất chọn gene để đột biến
            x = ind[i]
            delta1 = x - 0.0  
            delta2 = 1.0 - x  
            rand = random.random()
            mut_pow = 1.0/(eta_m+1.)
            if rand < 0.5:   # 50%: dịch sang trái
                xy = 1.0 - delta1
                val = 2.0*rand + (1.0-2.0*rand)*(pow(xy, (eta_m+1)))
                deltaq = pow(val, mut_pow) - 1.0
            else:              # 50%: dịch sang phải
                xy = 1.0 - delta2
                val = 2.0*(1.0-rand) + 2.0*(rand-0.5)*(pow(xy, (eta_m+1)))
                deltaq = 1.0 - pow(val, mut_pow)
            x = x + deltaq
            x = min(max(x, 0.0), 1.0)
            ind[i] = x
    return ind

def gaussian_mutation(ind, sigma=0.1, pm=None):
    """
    Gaussian Mutation
    1. Với từng gene, xác suất pm để đột biến
    2. Cộng thêm giá trị ngẫu nhiên từ phân phối Gaussian N(0, sigma)
    """
    n = len(ind)
    if pm is None:
        pm = 1.0/n
    for i in range(n):
        if random.random() < pm:
            ind[i] += random.gauss(0, sigma)
            ind[i] = min(max(ind[i], 0.0), 1.0)
    return ind

# -------- GA Main ----------
def GA(
    matrix,
    mutation: str,
    pop_size=50,
    generations=200,
    crossover_rate=0.9,
    mutation_rate=0.1,
    patience=30
    ):
    n = len(matrix)
    # Khởi tạo quần thể: vector số thực trong [0,1]
    population = [np.random.rand(n).tolist() for _ in range(pop_size)]
    best = max(population, key=lambda ind: fitness(ind, matrix))
    best_cost = tour_cost(decode_tour(best), matrix)

    history = [best_cost]
    no_improve = 0  # đếm số thế hệ không cải thiện

    for g in range(generations):
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population, 2)
            # lai ghép
            if random.random() < crossover_rate:
                c1, c2 = sbx_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
                
            # đột biến
            if mutation == "polynomial":
                if random.random() < mutation_rate:
                    c1 = polynomial_mutation(c1)
                if random.random() < mutation_rate:
                    c2 = polynomial_mutation(c2)
            elif mutation == "gaussian":
                if random.random() < mutation_rate:
                    c1 = gaussian_mutation(c1)
                if random.random() < mutation_rate:
                    c2 = gaussian_mutation(c2)
            new_pop.append(c1)
            new_pop.append(c2)

        # cập nhật quần thể
        population = sorted(new_pop, key=lambda ind: fitness(ind, matrix), reverse=True)[:pop_size]
        current_best = population[0]
        current_cost = tour_cost(decode_tour(current_best), matrix)
        history.append(current_cost)

        if current_cost < best_cost:
            best, best_cost = current_best, current_cost
            no_improve = 0
        else:
            no_improve += 1

        # In theo chu kỳ
        if g % 20 == 0:
            print(f"Gen {g}: cost = {best_cost}")

        # ---------- Kiểm tra điều kiện dừng ----------
        if no_improve >= patience:
            print(f"Dừng sớm tại gen {g}, không cải thiện sau {patience} thế hệ")
            break

    return decode_tour(best), best_cost, history

if __name__ == "__main__":
    problem = tsplib95.load("data/eil51.tsp")
    nodes = list(problem.get_nodes())
    matrix = [[problem.get_weight(i, j) for j in nodes] for i in nodes]

    # Chạy GA với polynomial mutation
    best_path, best_cost, history = GA(matrix, mutation="polynomial", pop_size=50, generations=500)
    print("Best cost (Polynomial):", best_cost)

    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.title("GA với Polynomial Mutation - Lịch sử Cost")
    plt.show()

    coords = [problem.node_coords[i] for i in nodes]
    x = [coords[i][0] for i in best_path] + [coords[best_path[0]][0]]
    y = [coords[i][1] for i in best_path] + [coords[best_path[0]][1]]
    plt.plot(x, y, 'o-r')
    plt.title(f"TSP Best Tour (Cost={best_cost})")
    plt.show()
    
    # Chạy GA với gaussian mutation
    best_path, best_cost, history = GA(matrix, mutation="gaussian", pop_size=50, generations=500)
    print("Best cost (gaussian):", best_cost)

    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.title("GA với gaussian Mutation - Lịch sử Cost")
    plt.show()

    coords = [problem.node_coords[i] for i in nodes]
    x = [coords[i][0] for i in best_path] + [coords[best_path[0]][0]]
    y = [coords[i][1] for i in best_path] + [coords[best_path[0]][1]]
    plt.plot(x, y, 'o-r')
    plt.title(f"TSP Best Tour (Cost={best_cost})")
    plt.show()
