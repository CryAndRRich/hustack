class SegmentTree:
    def __init__(self, n):
        self.N = 1
        while self.N < n: self.N <<= 1
        self.tree = [0] * (2 * self.N)

    def update(self, i, v):
        i += self.N
        self.tree[i] = max(self.tree[i], v)
        i >>= 1
        while i:
            self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
            i >>= 1

    def query(self, l, r):
        l += self.N; r += self.N
        res = 0
        while l <= r:
            if (l & 1) == 1:
                res = max(res, self.tree[l])
                l += 1
            if (r & 1) == 0:
                res = max(res, self.tree[r])
                r -= 1
            l >>= 1; r >>= 1
        return res
    
if __name__ == "__main__":
    n = int(input())
    a = list(map(int, input().split()))

    vals = sorted(set(a))
    comp = {v:i for i,v in enumerate(vals)}
    st = SegmentTree(len(vals))
    best = 0
    for x in a:
        idx = comp[x]
        if idx > 0:
            cur = st.query(0, idx-1) + 1
        else:
            cur = 1
        st.update(idx, cur)
        best = max(best, cur)
        
    print(best)