from ortools.linear_solver import pywraplp

def solve_prob(n, edges):
    solver = pywraplp.Solver.CreateSolver("CBC")
    x = {}
    for i in range(1, n + 1):
        x[i] = solver.IntVar(0, 1, f"x_{i}")
    
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            if (i, j) not in edges:
                solver.Add(x[i] + x[j] <= 1)
    
    L = solver.IntVar(0, n, "L")
    solver.Add(sum(x[i] for i in range(1, n + 1)) == L)
    solver.Maximize(L)
    solver.SetTimeLimit = 300_000

    status = solver.Solve()
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        clique = [i for i in range(1, n + 1) if x[i].solution_value() > 0.5] 
        print(len(clique))
        print(*clique)
    else:
        print("NO_SOLUTION")

if __name__ == "__main__":
    n, m = map(int, input().split())
    edges = set()
    for _ in range(m):
        a, b = map(int, input().split())
        edges.add((min(a, b), max(a, b)))
    
    solve_prob(n, edges)