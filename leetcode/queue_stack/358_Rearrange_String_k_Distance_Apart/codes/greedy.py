import collections

class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        if k == 0:
            return s
        
        mp = collections.Counter(s)
        mf = max(mp.values())
        mfch = [ch for ch, c in mp.items() if c == mf]
        
        head = ''.join(mfch)
        res = [head] * mf
        
        if k * (mf - 1) + len(head) > len(s):
            return ""
        
        A = [(c, ch) for ch, c in mp.items() if c != mf]
        A.sort(reverse=True)
        
        j = 0
        for c, ch in A:
            for _ in range(c):
                res[j] += ch
                j = (j + 1) % (mf - 1)
        
        return ''.join(res)
