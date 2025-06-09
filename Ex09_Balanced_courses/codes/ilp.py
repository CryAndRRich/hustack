from ortools.linear_solver import pywraplp

def solve(m, n, P, B):
    solver = pywraplp.Solver.CreateSolver("CBC")
    
    x = {}
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            x[(i, j)] = solver.IntVar(0, 1, f"x({i},{j})")

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if j not in P[i]:
                solver.Add(x[(i, j)] == 0)
    
    for j in range(1, n + 1):
        solver.Add(sum(x[(i, j)] for i in range(1, m + 1)) == 1)

    for i in range(1, m + 1):
        for (a, b) in B:
            solver.Add(x[(i, a)] + x[(i, b)] <= 1)
    
    L = solver.IntVar(0, n, "L")
    for i in range(1, m + 1):
        solver.Add(sum(x[(i, j)] for j in range(1, n + 1)) <= L)
    
    solver.Minimize(L)
    
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print(int(L.solution_value()))
    else:
        print(-1)

if __name__ == '__main__':
    m, n = map(int, input().split())
    
    P = {}
    for i in range(1, m+1):
        lst = list(map(int, input().split()))
        count = lst[0]
        P[i] = lst[1:]
    
    k = int(input())
    B = []
    for _ in range(k):
        a, b = map(int, input().split())
        B.append((a, b))
    
    solve(m, n, P, B)