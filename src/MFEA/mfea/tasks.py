import numpy as np

"""
- TSP:
Mỗi cá thể là hoán vị của các thành phố: [0, 3, 2, 1, 4, ...]
Fitness: tổng quãng đường cần minimize

- Knapsack:
Mỗi cá thể là vector nhị phân hoặc thực [0,1]: [1, 0, 1, 0, 1, ...]
Fitness: tổng giá trị tối đa hóa, có ràng buộc trọng lượng ≤ capacity

Ý tưởng: Dùng một vector thực chung [0,1]^D
Mỗi cá thể: x = [x₁, x₂, …, x_D], 0 ≤ xᵢ ≤ 1
Với mỗi task, ta giải mã (decode) vector đó theo cách riêng:
| Task         | Dạng gen           | Giải mã                                          |
| ------------ | ------------------ | ------------------------------------------------ |
| **TSP**      | Dải thực `[0,1]^n` | Sắp xếp thứ tự tăng dần để tạo hoán vị (argsort) |
| **Knapsack** | Dải thực `[0,1]^n` | Nếu `xᵢ > 0.5` → chọn vật i, ngược lại bỏ        |

"""

# TASK 1: Traveling Salesman Problem
def tsp_distance(path, dist_matrix):
    total = 0.0
    for i in range(len(path)):
        total += dist_matrix[path[i]][path[(i + 1) % len(path)]]
    return total

def decode_tsp(gen):
    return np.argsort(gen)

def fitness_tsp(gen, dist_matrix):
    path = decode_tsp(gen)
    return 1.0 / (tsp_distance(path, dist_matrix) + 1e-9)


# TASK 2: 0/1 Knapsack Problem
def decode_knapsack_ranked(gen, values, weights, capacity):
    """
    Sắp xếp item theo gen giảm dần.
    Duyệt theo thứ tự này, nhét item nếu còn sức chứa.
    Trả về vector bits (0/1).
    """
    n = len(gen)
    order = np.argsort(-gen)  # giảm dần
    bits = np.zeros(n, dtype=int)
    rem = capacity

    for idx in order:
        w = weights[idx]
        if w <= rem:
            bits[idx] = 1
            rem -= w
        # dừng sớm nếu không còn item nào có thể nhét được
    return bits

def knapsack_cost(gen, values, weights, capacity):
    bits = decode_knapsack_ranked(gen, values, weights, capacity)
    total_w = np.sum(weights * bits)
    total_v = np.sum(values * bits)
    
    if total_w > capacity:
        return 0.0
    return float(total_v)

def fitness_knapsack(gen, values, weights, capacity):
    return knapsack_cost(gen, values, weights, capacity)