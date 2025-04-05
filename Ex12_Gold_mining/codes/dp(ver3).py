class SegmentTree:
    def __init__(self, n):
        self.size = 1
        while self.size < n:
            self.size *= 2
        self.tree = [0 for _ in range(2 * self.size)]
    
    def update(self, ind, val):
        ind += self.size
        self.tree[ind] = val
        while ind > 1:
            ind //= 2
            self.tree[ind] = max(self.tree[2 * ind], self.tree[2 * ind + 1])
    
    def query(self, l, r):
        res = 0
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l //= 2
            r //= 2
        return res

if __name__ == '__main__':
    n, L1, L2 = list(map(int, input().split()))
    
    a = list(map(int, input().split()))

    dp = [0 for _ in range(n)]
    ans = 0

    seg_tree = SegmentTree(n)
    for i in range(n):
        if i - L1 >= 0:
            left = max(0, i - L2)
            right = i - L1
            best_prev = seg_tree.query(left, right)
        else:
            best_prev = 0
        
        dp[i] = a[i] + best_prev
        ans = max(ans, dp[i])
        seg_tree.update(i, dp[i])
    
    print(ans)
