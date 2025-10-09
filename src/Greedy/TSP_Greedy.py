import tsplib95

if __name__ == "__main__":
    problem = tsplib95.load(r"data/eil51.tsp")
    print("Số thành phố:", problem.dimension)

    # Tạo ma trận khoảng cách theo thứ tự node 1..n -> index 0..n-1
    nodes = list(problem.get_nodes())   # [1..n]
    n = len(nodes)
    matrix = [[problem.get_weight(i, j) for j in nodes] for i in nodes]

    result = []
    visited = [False] * n

    current = 0
    result.append(current)
    visited[current] = True

    while len(result) < n:
        next_idx = -1
        best = float("inf")
        for i in range(n):
            if not visited[i] and matrix[current][i] < best:
                best = matrix[current][i]
                next_idx = i

        if next_idx == -1:  
            for i in range(n):
                if not visited[i]:
                    next_idx = i
                    break

        result.append(next_idx)
        visited[next_idx] = True
        current = next_idx

    cost = 0
    for i in range(len(result) - 1):
        cost += matrix[result[i]][result[i + 1]]
    cost += matrix[result[-1]][result[0]]

    print("Tour (0-based indices):", result)
    print("Tour (city IDs 1-based):", [x + 1 for x in result])
    print("Cost:", cost)
