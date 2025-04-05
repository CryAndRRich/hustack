from collections import deque

def gcd(x: int, y: int) -> int:
    while y:
        x, y = y, x % y
    return x

def bfs(a: int, b: int, c: int) -> int:
    if c > max(a, b) or c % gcd(a, b) != 0:
        return -1

    visited = set()
    q = deque()
    q.append((0, 0, 0))  
    visited.add((0, 0))

    while q:
        x, y, steps = q.popleft()

        if x == c or y == c:
            return steps

        next = []
        next.append((a, y))
        next.append((x, b))
        next.append((0, y))
        next.append((x, 0))
        trans = min(x, b - y)
        next.append((x - trans, y + trans))
        trans = min(y, a - x)
        next.append((x + trans, y - trans))

        for state in next:
            if state not in visited:
                visited.add(state)
                q.append((state[0], state[1], steps + 1))
    return -1

if __name__ == "__main__":
    a, b, c = map(int, input().split())
    res = bfs(a, b, c)
    print(res)
