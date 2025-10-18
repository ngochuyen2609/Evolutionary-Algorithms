from pathlib import Path
import sys
from tsp_utils import load_tsplib_tsp, tour_length, nearest_neighbor_seed, two_opt_local_search
from rl_edge_dql import EdgeDoubleQL
from ga_tsp import GA_tsp

# Thư mục gốc project = 2 cấp trên (__file__/../..)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)
    
from utils.plot import plot_scores, plot_tour

def main():
    # 1) Load TSP
    dist, coords = load_tsplib_tsp("data/TSP/eil51.tsp")
    n = dist.shape[0]
    print(f"TSP n={n}")

    # 2) RL: học bảng Q theo cạnh
    rl = EdgeDoubleQL(n, alpha=0.01, gamma=0.15, eps_schedule=("linear",), seed=0)
    rl.train(dist, episodes=4000, reward="inv", start_mode="all")

    # 3) Tạo seed từ RL + một ít NN
    rl_seeds = rl.make_seeds(dist, k=40, diversify=True, do_2opt=True, two_opt_fn=two_opt_local_search)
    nn_seed = [nearest_neighbor_seed(n, dist, s) for s in range(min(5, n))]
    init_pop = rl_seeds + nn_seed

    # 4) GA: tinh chỉnh
    best, best_cost, history = GA_tsp(
        dist,
        init_pop=init_pop,
        pop_size=80,
        gens=800,
        cx_rate=0.95,
        mut_rate=0.25,
        use_2opt_every=40,
        two_opt_swaps=80
    )

    print("\nBest tour length:", best_cost)
    print("Best tour (1-based):", [x+1 for x in best] + [best[0]+1])

    # 5) Vẽ lịch sử & tour
    plot_scores(history, title="RL+GA TSP — Best tour length per generation", filename="images/history.png")
    plot_tour(coords, best, best_cost, title="RL+GA — Best TSP tour", filename="images/best_tour.png")

if __name__ == "__main__":
    main()
