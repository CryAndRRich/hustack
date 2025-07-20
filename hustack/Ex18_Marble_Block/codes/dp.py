if __name__ == '__main__':
    T = int(input())
    outputs = []
    for _ in range(T):
        L, W, H = map(int, input().split())
        N = int(input())
        desired = set()  
        for _ in range(N):
            a, b, c = map(int, input().split())
            for x, y, z in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)):
                desired.add((x, y, z))

        dp = [[[0 for _ in range(H + 1)] for _ in range(W + 1)] for _ in range(L + 1)]
        for i in range(1, L + 1):
            for j in range(1, W + 1):
                for k in range(1, H + 1):
                    if (i, j, k) in desired:
                        dp[i][j][k] = 0
                    else:
                        min_waste = i * j * k
                        for x in range(1, i // 2 + 1):
                            waste = dp[x][j][k] + dp[i - x][j][k]
                            if waste < min_waste:
                                min_waste = waste
                        for y in range(1, j // 2 + 1):
                            waste = dp[i][y][k] + dp[i][j - y][k]
                            if waste < min_waste:
                                min_waste = waste
                        for z in range(1, k // 2 + 1):
                            waste = dp[i][j][z] + dp[i][j][k - z]
                            if waste < min_waste:
                                min_waste = waste
                        dp[i][j][k] = min_waste
        outputs.append(dp[L][W][H])
        
    print("\n".join(map(str, outputs)))
