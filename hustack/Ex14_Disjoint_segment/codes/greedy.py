if __name__ == "__main__":
    n = int(input())
    segments = []
    for _ in range(n):
        a, b = list(map(int, input().split()))
        segments.append((a, b))
    
    segments.sort(key=lambda x: (x[1], x[0]))
    
    count = 0
    last_end = -1
    for seg in segments:
        if seg[0] > last_end:
            count += 1
            last_end = seg[1]
    
    print(count)
