def largest_rectangle_histogram(h):
    stack = []
    max_area = 0
    h.append(0)
    for i, height in enumerate(h):
        while stack and h[stack[-1]] > height:
            H = h[stack.pop()]
            W = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, H * W)
        stack.append(i)
    h.pop()
    return max_area

def maximal_rectangle(matrix, n, m):
    heights = [0] * m
    best = 0
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 1:
                heights[j] += 1
            else:
                heights[j] = 0
        best = max(best, largest_rectangle_histogram(heights))
    return best

if __name__ == "__main__":
    n, m = map(int, input().split())

    A = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        row = list(map(int, input().split()))
        for j in range(m):
            A[i][j] = row[j]
    
    print(maximal_rectangle(A, n, m))
