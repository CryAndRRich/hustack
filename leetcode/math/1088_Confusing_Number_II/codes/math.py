class Solution:
    def confusingNumberII(self, n: int) -> int:
        confusing_digits = [0, 1, 6, 8, 9]
        mapping = {"0": "0", "1": "1", "6": "9", "8": "8", "9": "6"}
        s = str(n)
        half = len(s) // 2
        odd = bool(len(s) % 2)
        ans = 2
        for l in range(2, len(s)):
            ans += 4 * (5 ** (l - 1) - 5 ** (l // 2 - 1) * (3 if l % 2 == 1 else 1))

        def calc(idx: int) -> int:
            num = int(s[idx])
            if num > 8:
                index = 4
            elif num > 6:
                index = 3
            elif num > 1:
                index = 2
            elif num == 1:
                index = 1
            else:
                index = 0
            if idx == len(s) - 1:
                return index + (confusing_digits[index] == int(s[idx]))
            smaller_num = index - 1 if idx == 0 else index
            res = smaller_num * 5 ** (len(s) - idx - 1)
            if idx < half:
                res -= smaller_num * 5 ** (half - idx - 1) * (3 if odd else 1)
            if idx == half and odd:
                res -= 3 if num > 8 else (2 if num > 1 else (1 if num > 0 else 0))
            if confusing_digits[index] == int(s[idx]):
                res += calc(idx + 1)
            return res

        ans += calc(0)
        first_half_num = 0
        for i in range(half - 1, -1, -1):
            if s[i] not in mapping:
                return ans
            first_half_num = first_half_num * 10 + int(mapping[s[i]])
        if odd:
            if s[half] in {"0", "1", "8"} and int(s[half + 1:]) >= first_half_num:
                ans -= 1
        elif int(s[half:]) >= first_half_num:
                ans -= 1
        return ans