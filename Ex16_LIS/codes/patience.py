from typing import List
import bisect

def lis(nums: List[int]) -> int:
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)

if __name__ == "__main__":
    n = int(input())
    nums = list(map(int, input().split()))
    print(lis(nums))
