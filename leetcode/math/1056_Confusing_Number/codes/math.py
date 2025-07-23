class Solution:
    def confusingNumber(self, n: int) -> bool:
        rotate_map = {"0":"0", "1":"1", "6":"9", "8":"8", "9":"6"}
        s = str(n)
        rotated = []
        
        for ch in reversed(s):
            if ch not in rotate_map:
                return False
            rotated.append(rotate_map[ch])
        
        return int("".join(rotated)) != n
