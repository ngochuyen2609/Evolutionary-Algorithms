# GA với Mã hóa Hoán vị
- Lựa chọn (Selection):
    - Tournament Selection
    - Roulette Wheel Selection

- Lai ghép (Crossover):
    - Order Crossover (OX)
    - Partially Mapped Crossover (PMX)

- Đột biến (Mutation):
    - Swap Mutation (hoán đổi 2 gene)
    - Inversion Mutation (đảo ngược một đoạn gene)

## Kết quả minh họa
1. Roulette + PMX Crossover + Inversion Mutation  
    ![Alt text](images/roulette_PMXCrossover_InversionMutation.png)
    ![Alt text](images/bestTour_Roulette_PMXCrossover_InversionMutation.png)

2. Tournament + OX Crossover + Swap Mutation  
    ![Alt text](images/tournament_OXCrossover_SwapMutation.png)
    ![Alt text](images/bestTour_Tournament_OXCrossover_SwapMutation.png)


# GA với Mã hóa Số thực
- Ý tưởng mã hóa
    - Mỗi cá thể là một vector số thực trong [0,1].
    - Khi giải mã (decode), ta sắp xếp vector để thu được thứ tự thành phố (tour).

- Lai ghép (Crossover):
    - Simulated Binary Crossover (SBX)
- Đột biến (Mutation):
    - Polynomial Mutation
    - Gaussian Mutation

## Kết quả minh họa
1. Real-coded + Polynomial Mutation  
    ![Alt text](images/realNumber_PolynomialMutation.png)
    ![Alt text](images/bestTour_realNumber_PolynomialMutation.png)

2. Real-coded + Gaussian Mutation  
    ![Alt text](images/realNumber_GaussianMutation.png)
    ![Alt text](images/bestTour_realNumber_GaussianMutation.png)


# Kết quả minh họa tour tối ưu .opt.tour
![Alt text](images/bestTour.png)