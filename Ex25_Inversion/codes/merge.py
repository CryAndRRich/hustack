MOD = 10**9 + 7

def inversions(arr):
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr, 0

        mid = len(arr) // 2
        left, inv_left = merge_sort(arr[:mid])
        right, inv_right = merge_sort(arr[mid:])
        merged, inv_split = merge_and_count(left, right)

        total_inv = (inv_left + inv_right + inv_split) % MOD
        return merged, total_inv

    def merge_and_count(left, right):
        res = []
        i = j = inv_count = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                inv_count += len(left) - i
                j += 1

        res.extend(left[i:])
        res.extend(right[j:])
        return res, inv_count % MOD

    _, total_inversions = merge_sort(arr)
    return total_inversions

if __name__ == "__main__":
    n = int(input())
    arr = list(map(int, input().split()))

    print(inversions(arr))
