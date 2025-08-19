MOD = int(1e9 + 7)
MAX_VAL = int(1e6)

def modPlus(a, b, mod=MOD):
    return (a + b) % mod

def modMinus(a, b, mod=MOD):
    return (a - b + mod) % mod

def modMult(a, b, mod=MOD):
    return (a * b) % mod

def modPow(base, exp, mod=MOD):
    result = 1
    while exp:
        if exp & 1:
            result = modMult(result, base, mod)
        base = modMult(base, base, mod)
        exp >>= 1
    return result

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        from collections import defaultdict

        freq = defaultdict(int)
        n = len(nums)
        result = 0
        total = 0

        for r in range(n):
            prevExp = freq[nums[r]]
            freq[nums[r]] += 1

            if prevExp > 0:
                contribution = modPow(nums[r], prevExp)
                total = modMinus(total, contribution)
                contribution = modMult(contribution, nums[r])
                total = modPlus(total, contribution)
            else:
                total = modPlus(total, nums[r])

            if r >= k:
                left = nums[r - k]
                nextExp = freq[left] - 1
                freq[left] -= 1

                if nextExp > 0:
                    contribution = modPow(left, nextExp + 1)
                    total = modMinus(total, contribution)
                    contribution = modPow(left, nextExp)
                    total = modPlus(total, contribution)
                else:
                    total = modMinus(total, left)

            if r >= k - 1:
                result = max(result, total)

        return result
