# Multifactorial Evolutionary Algorithm (MFEA)

## 1. Mục tiêu

Thay vì tối ưu từng bài toán riêng biệt, MFEA cho phép tối ưu **nhiều bài toán đồng thời trong cùng một quần thể**, nhờ cơ chế **chia sẻ kiến thức di truyền (Genetic Transfer)**.

| Thuật toán          | Số quần thể | Số task | Mối liên hệ                           |
| ------------------- | ----------- | ------- | ------------------------------------- |
| GA truyền thống     | 1           | 1       | Tối ưu đơn nhiệm                      |
| Multi-population GA | N           | N       | Mỗi task riêng biệt                   |
| MFEA                | 1           | N       | Quần thể duy nhất học cho tất cả task |

---

## 2. Đặc tính của cá thể trong MFEA

Mỗi cá thể trong quần thể MFEA được mô tả bởi 3 thành phần:

| Thành phần            | Ký hiệu                        | Ý nghĩa                                         |
| --------------------- | ------------------------------ | ----------------------------------------------- |
| Genotype              | X_i = [x_1, x_2, ..., x_n]     | Biểu diễn thực hoặc nhị phân dùng chung cho mọi task |
| Skill Factor          | τ_i ∈ {1, ..., K}              | Task mà cá thể này "chuyên" giải                |
| Factorial Fitness     | f_i^(k)                        | Mức độ phù hợp của cá thể i trên task k         |

### Ví dụ

| Cá thể | Gen (50 biến)   | Skill Factor | Fitness TSP | Fitness Knapsack |
| ------ | --------------- | ------------ | ------------ | ---------------- |
| A      | [0.1, 0.3, ...] | 0            | 0.95         | -                |
| B      | [0.9, 0.2, ...] | 1            | -            | 0.83             |

---

## 3. Pipeline chi tiết của MFEA

Khởi tạo quần thể P gồm pop_size cá thể ngẫu nhiên
│
├─► Gán skill_factor ngẫu nhiên {0, 1, ..., K-1}
│
├─► Đánh giá fitness theo task tương ứng
│
┌───────────────────────── LOOP ─────────────────────────┐
│ 1. Chọn 2 cha (cha1, cha2)
│ ├ Nếu cùng task → lai bình thường
│ └ Nếu khác task → lai với xác suất rmp (random mating probability)
│
│ 2. Lai ghép → sinh 2 con (SBX, PMX, ...)
│ 3. Đột biến → Gaussian hoặc Polynomial
│ 4. Gán skill_factor cho con (random theo cha)
│ 5. Đánh giá fitness con
│
│ 6. Hợp quần thể (cha + con)
│ 7. Chọn lọc elitism cho từng task (giữ pop_size)
└────────────────────────────────────────────────────────┘

Kết thúc khi đủ số thế hệ hoặc hội tụ.



## 4. Cơ chế truyền tri thức (Knowledge Transfer)

Đây là phần độc đáo nhất của MFEA.

- Khi cá thể của task A lai với cá thể của task B, có thể sinh ra con mang đặc tính tốt cho cả hai.
- Nếu hai task có cấu trúc tương tự (related), gen tốt ở task A có thể giúp task B hội tụ nhanh hơn.

Ví dụ:
- Một giải pháp TSP với trình tự "tối ưu hóa đường đi" có thể gợi ý pattern phân nhóm hiệu quả trong Knapsack.

---

## 5. Toán học cơ bản

Mỗi cá thể i có **factorial rank** cho từng task k:

r_i^(k) = rank của cá thể i trên task k



Sau đó, **factorial skill factor** được xác định:

τ_i = argmin_k r_i^(k)


=> Cá thể được "chuyên hóa" cho task mà nó mạnh nhất.

**Fitness chuyển hóa:**

φ_i = 1 / r_i^(τ_i)


Mục tiêu: maximize φ_i trong toàn quần thể.

---

## 6. Các toán tử phổ biến

| Loại | Ví dụ | Giải thích |
|------|--------|------------|
| Lai ghép (Crossover) | SBX (Simulated Binary), PMX, OX | Kết hợp gen của cha mẹ |
| Đột biến (Mutation) | Gaussian, Polynomial, Swap, Inversion | Tăng đa dạng, tránh hội tụ sớm |
| Lựa chọn (Selection) | Elitism theo task | Giữ lại cá thể tốt nhất từng task |
| RMP | Random Mating Probability | Xác suất lai khác task (thường = 0.3) |

---

## 7. Pipeline minh họa

┌──────────────────────────────────────────────┐
│ POPULATION P │
│──────────────────────────────────────────────│
│ [0.3,0.5,...] | skill=0 (TSP) | fit=0.89 │
│ [0.9,0.1,...] | skill=1 (Knap) | fit=0.77 │
│──────────────────────────────────────────────│
│ [0.2,0.8,...] | skill=0 (TSP) | fit=0.92 │
└──────────────────────────────────────────────┘
│
▼
┌──────────────┐
│ Recombination│
│ + Mutation │
└──────────────┘
│
▼
┌──────────────────────────────┐
│ Evaluate each offspring │
└──────────────────────────────┘
│
▼
┌──────────────┐
│ Selection │
│ (per task) │
└──────────────┘
│
▼
Next Generation


## 8. Ưu điểm nổi bật

| Ưu điểm | Giải thích |
|----------|-------------|
| Chia sẻ kiến thức giữa các task | Nếu 2 task có quan hệ, chúng học lẫn nhau |
| Tiết kiệm tài nguyên | Một quần thể duy nhất thay vì nhiều GA độc lập |
| Hội tụ nhanh hơn | Task yếu học từ task mạnh |
| Khả năng tổng quát hóa cao | Dễ mở rộng cho nhiều task và miền dữ liệu khác nhau |

---

## 9. Hạn chế và lưu ý

| Hạn chế | Hướng khắc phục |
|----------|----------------|
| Task không liên quan → truyền tri thức tiêu cực | Dùng Adaptive RMP (thay đổi rmp động) |
| Kích thước biểu diễn không đồng nhất | Dùng Unified Representation hoặc Padding |
| Cần kiểm soát cân bằng giữa task | Elitism riêng từng task |

---

## 10. Tóm tắt ngắn

| Thành phần | Vai trò |
|-------------|----------|
| population | Quần thể chung cho mọi task |
| skill_factor | Task mà cá thể "chuyên" |
| rmp | Xác suất lai khác task (transfer knowledge) |
| fitness_tsp, fitness_knapsack | Hàm đánh giá từng task |
| elitism | Giữ lại cá thể tốt nhất của mỗi task |

