class Solution:
    def maximumNumberOfOnes(self, width: int, height: int, sideLength: int, maxOnes: int) -> int:
        count = {}
        for i in range(width):
            for j in range(height):
                key = (i % sideLength, j % sideLength)
                count[key] = count.get(key, 0) + 1
        freq = sorted(count.values(), reverse=True)
        return sum(freq[:maxOnes])
