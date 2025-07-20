def binary_search(segments, left, right, target):
    ans = -1
    while left <= right:
        mid = left + (right - left) // 2
        if segments[mid][1] < target:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
    return ans

if __name__ == "__main__":
    n = int(input())
    segments = []
    for _ in range(n):
        a, b = list(map(int, input().split()))
        segments.append((a, b))
    
    segments.sort(key=lambda seg: (seg[1], seg[0]))
    
    dp = [0] * (n + 1)
    dp[0] = 0
    for i in range(n):
        seg_start = segments[i][0]
        ind = binary_search(segments, 0, i - 1, seg_start)
        candidate = 1
        if ind != -1:
            candidate += dp[ind + 1]
        dp[i + 1] = max(dp[i], candidate)
    
    print(dp[n])
