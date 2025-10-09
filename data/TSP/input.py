import tsplib95

def main():
    problem = tsplib95.load("data\eil51.tsp")

    print("Số thành phố:", problem.dimension)

    for node in list(problem.get_nodes())[:5]:
        print(f"City {node}: {problem.node_coords[node]}")

    n = problem.dimension
    matrix = [[problem.get_weight(i, j) for j in problem.get_nodes()] for i in problem.get_nodes()]

    print("\nMa trận khoảng cách (5x5 đầu):")
    for row in matrix[:5]:
        print(row[:5])

if __name__ == "__main__":
    main()
