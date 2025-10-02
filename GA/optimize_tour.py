import tsplib95
import matplotlib.pyplot as plt

problem = tsplib95.load("data/eil51.tsp")
tour_problem = tsplib95.load("data/eil51.opt.tour")

optimal_tour = list(tour_problem.tours[0])
print("Optimal tour:", optimal_tour)

optimal_cost = 0
for i in range(len(optimal_tour) - 1):
    optimal_cost += problem.get_weight(optimal_tour[i], optimal_tour[i+1])
optimal_cost += problem.get_weight(optimal_tour[-1], optimal_tour[0])  # quay lại điểm đầu
print("Optimal cost:", optimal_cost)

coords = problem.node_coords
x = [coords[i][0] for i in optimal_tour] + [coords[optimal_tour[0]][0]]
y = [coords[i][1] for i in optimal_tour] + [coords[optimal_tour[0]][1]]

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o-r')
plt.title(f"TSP Optimal Tour (Cost={optimal_cost})")
plt.show()
