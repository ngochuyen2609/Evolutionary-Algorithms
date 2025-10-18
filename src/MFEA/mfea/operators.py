import numpy as np
import random

# Tournament
def random_parents_mfea(population, skill_factor):
    i, j = np.random.choice(len(population), 2, replace=False)
    return (population[i], skill_factor[i]), (population[j], skill_factor[j])

# Simulated Binary Crossover (SBX)
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

# Gaussian Mutation
def gaussian_mutation(ind, sigma=0.1, pm=0.2):
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