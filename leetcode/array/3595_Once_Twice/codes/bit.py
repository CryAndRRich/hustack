from typing import List

class Solution:
    def onceTwice(self, nums: List[int]) -> List[int]:
        bits = [0] * 32
        for num in nums:
            x = num & 0xFFFFFFFF
            for i in range(32):
                bits[i] += (x >> i) & 1
        xor_u = 0
        for i in range(32):
            if bits[i] % 3 != 0:
                xor_u |= (1 << i)
        mask = xor_u & -xor_u
        bitsg = [0]*32
        for num in nums:
            x = num & 0xFFFFFFFF
            if (x & mask) != 0:
                for i in range(32):
                    bitsg[i] += (x >> i) & 1
        cand_u = 0
        for i in range(32):
            if bitsg[i] % 3 != 0:
                cand_u |= (1 << i)
        def to_signed(u):
            if u >= (1 << 31):
                return u - (1 << 32)
            return u
        cand = to_signed(cand_u)
        other = to_signed(cand_u ^ xor_u)
        cnt = sum(1 for x in nums if x == cand)
        if cnt == 1:
            return [cand, other]
        else:
            return [other, cand]
