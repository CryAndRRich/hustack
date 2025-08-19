from typing import List

class Solution:
    def beautifulPair(self, nums1: List[int], nums2: List[int]) -> List[int]:
        n = len(nums1)
        p1 = [nums2[i] + nums1[i] for i in range(n)]
        p2 = [nums2[i] - nums1[i] for i in range(n)]
        maxCoord = max(nums1) if n>0 else 0
        size = maxCoord + 1
        tree1 = [-1] * (4 * (size+1))
        tree2 = [-1] * (4 * (size+1))
        def better(v, x, y):
            if x < 0:
                return y
            if y < 0:
                return x
            if v[x] < v[y] or (v[x] == v[y] and y < x):
                return y
            return x
        def query(tree, v, ind, left, right, ql, qr):
            if ql > qr:
                return -1
            if left >= ql and right <= qr:
                return tree[ind]
            mid = (left + right) >> 1
            res = -1
            if ql <= mid:
                res = query(tree, v, ind<<1, left, mid, ql, qr)
            if qr > mid:
                res = better(v, res, query(tree, v, (ind<<1)|1, mid+1, right, ql, qr))
            return res
        def update(tree, v, ind, left, right, pos, val):
            tree[ind] = better(v, tree[ind], val)
            if left == right:
                return
            mid = (left + right) >> 1
            if pos <= mid:
                update(tree, v, ind<<1, left, mid, pos, val)
            else:
                update(tree, v, (ind<<1)|1, mid+1, right, pos, val)
        idx = list(range(n))
        idx.sort(key=lambda x: nums2[x])
        ans_i1 = n
        ans_i2 = n
        d = 2**63-1
        for k in range(n):
            i = idx[k]
            x = nums1[i]
            if x >= 0:
                i1 = query(tree1, p1, 1, 0, size, 0, x)
                if i1 >= 0:
                    may = p1[i] - p1[i1]
                    if may < d or (may == d and (min(i, i1) < ans_i1 or (min(i, i1) == ans_i1 and max(i, i1) < ans_i2))):
                        d = may
                        ans_i1, ans_i2 = min(i, i1), max(i, i1)
            i2 = query(tree2, p2, 1, 0, size, x, size)
            if i2 >= 0:
                may = p2[i] - p2[i2]
                if may < d or (may == d and (min(i, i2) < ans_i1 or (min(i, i2) == ans_i1 and max(i, i2) < ans_i2))):
                    d = may
                    ans_i1, ans_i2 = min(i, i2), max(i, i2)
            update(tree1, p1, 1, 0, size, x, i)
            update(tree2, p2, 1, 0, size, x, i)
        return [ans_i1, ans_i2]
