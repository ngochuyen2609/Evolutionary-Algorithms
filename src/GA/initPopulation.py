import random
import math

def _nearest_neighbor_tour(n, dist, start, alpha=0.0):
    """
    Tạo tour bằng Nearest Neighbor:
    - start: đỉnh bắt đầu
    - alpha ∈ [0,1]: mức ngẫu nhiên hoá (GRASP). 0 => chọn gần nhất tuyệt đối.
      >0 => chọn ngẫu nhiên trong danh sách ứng viên tốt (RCL).
    """
    unvis = list(range(n))
    tour = [start]
    unvis.remove(start)

    while unvis:
        last = tour[-1]
        # khoảng cách tới các đỉnh chưa thăm
        cand = sorted(unvis, key=lambda j: dist[last][j])
        if alpha > 0.0 and len(cand) > 1:
            # RCL: các ứng viên trong top m dựa trên alpha
            m = max(1, int(math.ceil(alpha * len(cand))))
            nxt = random.choice(cand[:m])
        else:
            nxt = cand[0]
        tour.append(nxt)
        unvis.remove(nxt)
    return tour

def _two_opt_cost_change(dist, tour, i, k):
    """Lợi ích khi đảo đoạn [i+1..k] (chuẩn 2-opt)."""
    n = len(tour)
    a, b = tour[i], tour[(i+1) % n]
    c, d = tour[k], tour[(k+1) % n]
    return (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

def _two_opt(dist, tour, max_swaps=200):
    """Cải thiện tour bằng 2-opt đơn giản (giới hạn số swap)."""
    n = len(tour)
    swaps = 0
    improved = True
    while improved and swaps < max_swaps:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n - (0 if i > 0 else 1)):
                delta = _two_opt_cost_change(dist, tour, i, k)
                if delta < 0:  # giảm chi phí
                    tour[i+1:k+1] = reversed(tour[i+1:k+1])
                    swaps += 1
                    improved = True
                    if swaps >= max_swaps:
                        break
            if swaps >= max_swaps:
                break
    return tour

def init_population_greedy(pop_size, num_genes, distance_matrix=None,
                           alpha=0.0, use_2opt=False):
    """
    Khởi tạo quần thể TSP chất lượng bằng tham lam:
    - distance_matrix: ma trận khoảng cách (bắt buộc để dùng tham lam). Nếu None -> fallback random.
    - alpha ∈ [0,1]: GRASP randomness (0 = thuần NN; 0.2..0.4 gợi ý để đa dạng).
    - use_2opt: True để local search nhanh sau khi có tour NN.
    Trả về: list[list[int]] kích thước pop_size.
    """
    population = []

    if distance_matrix is None:
        # fallback: random hoán vị
        for _ in range(pop_size):
            ind = list(range(num_genes))
            random.shuffle(ind)
            population.append(ind)
        return population

    n = num_genes
    starts = list(range(n))
    random.shuffle(starts)

    # đảm bảo có đủ số cá thể (lặp lại start + tạo thêm bằng alpha khác)
    while len(population) < pop_size:
        if starts:
            s = starts.pop()
        else:
            s = random.randrange(n)

        tour = _nearest_neighbor_tour(n, distance_matrix, start=s, alpha=alpha)
        if use_2opt:
            tour = _two_opt(distance_matrix, tour, max_swaps=200)

        population.append(tour)

        # thêm đa dạng: có thể tạo thêm 1–2 tour biến thể từ cùng start khi cần
        if len(population) < pop_size and alpha > 0.0:
            tour2 = _nearest_neighbor_tour(n, distance_matrix, start=s, alpha=alpha)
            if use_2opt:
                tour2 = _two_opt(distance_matrix, tour2, max_swaps=120)
            population.append(tour2)

    # cắt đúng kích thước
    return population[:pop_size]