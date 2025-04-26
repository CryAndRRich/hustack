if __name__ == "__main__":
    n, m = map(int, input().split())

    A = [[0] * (m + 1) for _ in range(n + 1)]
    S = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        row = list(map(int, input().split()))
        for j in range(1, m + 1):
            A[i][j] = row[j - 1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S[i][j] = A[i][j] + S[i - 1][j] + S[i][j - 1] - S[i - 1][j - 1]

    maxArea = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if S[i][j] == i * j:
                maxArea = max(maxArea, i * j)

    print(maxArea)
