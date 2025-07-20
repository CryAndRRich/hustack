def gcd(a: int, b: int) -> int:
    if a < b:
        a, b = b, a
    if a % b == 0:
        return b
    
    return gcd(b, a % b)

def pourWater(start: int, end: int, target: int) -> int:
    one = start
    two = 0
    step = 1

    while one != target and two != target:
        trans = min(one, end - two)
        two += trans
        one -= trans
        step += 1
        
        if one == target or two == target:
            break

        if one == 0:
            one = start
            step += 1
        if two == end:
            two = 0
            step += 1

    return step

def minStep(a: int, b: int, c: int) -> int:
    meth1 = pourWater(a, b, c)
    meth2 = pourWater(b, a, c)
    return min(meth1, meth2)

if __name__ == "__main__":
    a, b, c = list(map(int, input().split()))

    if c <= max(a, b) and c % gcd(a, b) == 0:
        print(minStep(a, b, c))
    else:
        print(-1)
    