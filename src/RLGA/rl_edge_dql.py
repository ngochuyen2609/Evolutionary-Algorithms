import random
import numpy as np

class EdgeDoubleQL:
    """
    Double Q-learning theo cạnh:
    Học QA[i,j], QB[i,j] ~ độ hấp dẫn khi đi i -> j.
    Dùng trung bình (QA+QB)/2 để chọn greedy.
    """
    def __init__(self, n, alpha=0.01, gamma=0.15, eps_schedule=("linear",), seed=0):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.QA = np.zeros((n, n), dtype=float)
        self.QB = np.zeros((n, n), dtype=float)
        self.eps_schedule = eps_schedule

    def _epsilon(self, t, T):
        kind = self.eps_schedule[0]
        if kind == "linear":
            return 1.0 - (t / max(T, 1))
        if kind == "pow":
            p = self.eps_schedule[1] if len(self.eps_schedule) > 1 else 6
            x = t / max(T, 1)
            return max(0.0, 1.0 - x**p)
        return 0.1

    def greedy_next(self, i, unvisited):
        q = (self.QA[i] + self.QB[i]) * 0.5
        return max(unvisited, key=lambda j: q[j])

    def train(self, dist, episodes=4000, reward="inv", start_mode="all"):
        """
        reward: 'inv' -> r=1/d,  'neg' -> r=-d
        start_mode: 'random' hoặc 'all'
        """
        n = self.n
        T = episodes
        for ep in range(episodes):
            eps = self._epsilon(ep, T)
            s = ep % n if start_mode == "all" else self.rng.randrange(n)

            visited = {s}
            cur = s

            while len(visited) < n:
                # chọn action
                if self.rng.random() < eps:
                    candidates = [j for j in range(n) if j not in visited]
                    j = self.rng.choice(candidates)
                else:
                    j = self.greedy_next(cur, [x for x in range(n) if x not in visited])

                d = dist[cur, j]
                r = (1.0 / (d + 1e-12)) if reward == "inv" else (-d)

                # update QA hoặc QB
                if self.rng.random() < 0.5:
                    # update QA, lookahead từ QB
                    if len(visited) + 1 < n:
                        mask_next = [a for a in range(n) if a not in visited and a != j]
                        max_next = np.max(self.QB[j, mask_next]) if mask_next else 0.0
                    else:
                        max_next = 0.0
                    td = r + self.gamma * max_next - self.QA[cur, j]
                    self.QA[cur, j] += self.alpha * td
                else:
                    # update QB, lookahead từ QA
                    if len(visited) + 1 < n:
                        mask_next = [a for a in range(n) if a not in visited and a != j]
                        max_next = np.max(self.QA[j, mask_next]) if mask_next else 0.0
                    else:
                        max_next = 0.0
                    td = r + self.gamma * max_next - self.QB[cur, j]
                    self.QB[cur, j] += self.alpha * td

                visited.add(j)
                cur = j

        return self

    def build_tour(self, dist, start=0, random_tie=False):
        n = self.n
        visited = {start}
        cur = start
        tour = [start]
        while len(visited) < n:
            unv = [j for j in range(n) if j not in visited]
            q = (self.QA[cur] + self.QB[cur]) * 0.5
            ranked = sorted(unv, key=lambda j: q[j], reverse=True)
            j = ranked[0] if not random_tie else random.choice(ranked[:max(1, len(ranked)//3)])
            tour.append(j)
            visited.add(j)
            cur = j
        return tour

    def make_seeds(self, dist, k=40, diversify=True, do_2opt=True, two_opt_fn=None):
        seeds = []
        # một phần từ các start khác nhau
        for s in range(min(self.n, max(1, k//2))):
            t = self.build_tour(dist, start=s, random_tie=diversify)
            if do_2opt and two_opt_fn:
                t = two_opt_fn(t, dist, max_swaps=150)
            seeds.append(t)
        # phần còn lại random start
        import random as _r
        while len(seeds) < k:
            s = _r.randrange(self.n)
            t = self.build_tour(dist, start=s, random_tie=diversify)
            if do_2opt and two_opt_fn:
                t = two_opt_fn(t, dist, max_swaps=150)
            seeds.append(t)
        return seeds
