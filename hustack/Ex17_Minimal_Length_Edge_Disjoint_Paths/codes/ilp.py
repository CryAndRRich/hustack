from ortools.linear_solver import pywraplp

def solve(n, edges):
    solver = pywraplp.Solver.CreateSolver("CBC")
    x = {}
    for (i, j, c) in edges:
        for k in (0, 1):
            x[(i, j, k)] = solver.IntVar(0, 1, f"x[{i},{j},{k}]")

    for i, j, _ in edges:
        solver.Add(x[(i, j, 0)] + x[(i, j, 1)] <= 1)

    for k in (0, 1):
        for v in range(1, n + 1):
            out = []
            in_ = []
            for i, j, _ in edges:
                if i == v:
                    out.append(x[(i, j, k)])
                if j == v:
                    in_.append(x[(i, j, k)])
            if v == 1:
                solver.Add(solver.Sum(out) - solver.Sum(in_) == 1)
            elif v == n:
                solver.Add(solver.Sum(out) - solver.Sum(in_) == -1)
            else:
                solver.Add(solver.Sum(out) - solver.Sum(in_) == 0)

    obj = solver.Objective()
    for i, j, c in edges:
        for k in (0, 1):
            obj.SetCoefficient(x[(i, j, k)], c)
    obj.SetMinimization()

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        ans = solver.Objective().Value()
        print(int(ans))
    else:
        print("NOT_FEASIBLE")


if __name__ == "__main__":
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, c = map(int, input().split())
        edges.append((u, v, c))

    solve(n, edges)
