if __name__ == '__main__':
    n, L1, L2 = list(map(int, input().split()))
    
    a = list(map(int, input().split()))
    
    dp = [0 for _ in range(n)]
    ans = 0
    for i in range(n):
        best_prev = 0
        start = i - L2
        end = i - L1
        if end >= 0:
            for j in range(max(0, start), end + 1):
                if dp[j] > best_prev:
                    best_prev = dp[j]
            dp[i] = a[i] + best_prev
        else:
            dp[i] = a[i]
        if dp[i] > ans:
            ans = dp[i]
    
    print(ans)
