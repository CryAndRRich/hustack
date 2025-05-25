from ortools.linear_solver import pywraplp

def solve(n, s, L, edges):
    solver = pywraplp.Solver.CreateSolver("CBC")

    x = {}
    for (u, v, t, _) in edges:
        x[(u, v)] = solver.BoolVar(f"x_{u}_{v}")

    T = {}
    for i in range(n):
        T[i] = solver.IntVar(0, L, f"T_{i}")

    max_t = max(t for (_, _, t, _) in edges)
    M = L + max_t + 1

    for i in range(n):
        incoming = []
        for (u, v, _, _) in edges:
            if v == i:
                incoming.append(x[(u, v)])
        if i == s:
            solver.Add(solver.Sum(incoming) == 0)
        else:
            solver.Add(solver.Sum(incoming) == 1)

    solver.Add(solver.Sum(x.values()) == n - 1)

    for (u, v, t, _) in edges:
        solver.Add(T[v] >= T[u] + t - M * (1 - x[(u, v)]))

    solver.Add(T[s] == 0)

    objective = solver.Sum(c * x[(u, v)] for (u, v, _, c) in edges)
    solver.Minimize(objective)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print(int(solver.Objective().Value()))
    else:
        print("NO_SOLUTION")

if __name__ == "__main__":
    n, m, s, L = map(int, input().split())
    edges = []
    
    for _ in range(m):
        u, v, t, c = map(int, input().split())
        edges.append((u - 1, v - 1, t, c))
        edges.append((v - 1, u - 1, t, c))

    solve(n, s - 1, L, edges)
