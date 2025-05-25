from ortools.sat.python import cp_model

def solve(n, s, L, edges):
    model = cp_model.CpModel()
    x = {}
    T = {}
    
    for (u, v, t, _) in edges:
        if v != s - 1:
            x[(u, v)] = model.NewBoolVar(f"x_{u}_{v}")
    
    for i in range(n):
        if i != s - 1:
            model.Add(sum(x[(u, i)] for u, v, _, _ in edges if v == i) == 1)
            
            
    for i in range(n):
        T[i] = model.NewIntVar(0, L, f"T_{i}")
    
    model.Add(T[s - 1] == 0)

    for (u, v, t, _) in edges:
        if v != s - 1:
            model.Add(T[v] == T[u] + t).OnlyEnforceIf(x[(u, v)])
            model.Add(T[v] <= L)


    model.Minimize(sum(c * x[(u, v)] for (u, v, _, c) in edges if v != s - 1))    
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL:
        print(int(solver.ObjectiveValue()))

    else:
        print("NO_SOLUTION")

if __name__ == "__main__":
    n, m, s, L = map(int, input().split())
    edges = []
    
    for _ in range(m):
        u, v, t, c = map(int, input().split())
        edges.append((u - 1, v - 1, t, c))
        edges.append((v - 1, u - 1, t, c))
    
    solve(n, s, L, edges)