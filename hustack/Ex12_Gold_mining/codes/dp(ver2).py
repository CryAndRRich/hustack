from collections import deque

if __name__ == '__main__':
    n, L1, L2 = list(map(int, input().split()))
    
    a = list(map(int, input().split()))
    
    dp = [0 for _ in range(n)]
    ans = 0
    dq = deque()
    
    for i in range(n):
        while dq and dq[0] < i - L2:
            dq.popleft()
        
        if i - L1 >= 0:
            j = i - L1
            while dq and dp[dq[-1]] <= dp[j]:
                dq.pop()
            dq.append(j)
        
        best_prev = dp[dq[0]] if dq else 0
        dp[i] = a[i] + best_prev
        ans = max(ans, dp[i])
    
    print(ans)
