from ortools.sat.python import cp_model

def solve(m, n, P, B):
    model = cp_model.CpModel()
    
    x = {}
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            x[(i, j)] = model.NewBoolVar(f'x({i},{j})')
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if j not in P[i]:
                model.Add(x[(i, j)] == 0)
    
    for j in range(1, n + 1):
        model.Add(sum(x[(i, j)] for i in range(1, m + 1)) == 1)
    
    for i in range(1, m + 1):
        for (a, b) in B:
            model.Add(x[(i,a)] + x[(i,b)] <= 1)
    
    loads = {}
    for i in range(1, m + 1):
        loads[i] = model.NewIntVar(0, n, f'load_{i}')
        model.Add(loads[i] == sum(x[(i, j)] for j in range(1, n + 1)))
    
    L = model.NewIntVar(0, n, 'L')
    for i in range(1, m + 1):
        model.Add(loads[i] <= L)
    
    model.Minimize(L)
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(solver.Value(L))
    #   for i in range(1, m + 1):
    #       tasks = [j for j in range(1, n + 1) if solver.Value(x[(i, j)])]
    #       print(f'Teacher {i}: Course {tasks}')
    else:
        print(-1)
    #   print('No solution found.')

if __name__ == '__main__':
    m, n = map(int, input().split())
    
    P = {}
    for i in range(1, m + 1):
        lst = list(map(int, input().split()))
        count = lst[0]
        P[i] = lst[1:]
    
    k = int(input())
    B = []
    for _ in range(k):
        a, b = map(int, input().split())
        B.append((a, b))
    
    solve(m, n, P, B)
