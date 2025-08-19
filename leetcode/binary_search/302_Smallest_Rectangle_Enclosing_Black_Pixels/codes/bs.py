class Solution:
    def minArea(self, image: list[list[str]], x: int, y: int) -> int:
        m, n = len(image), len(image[0])

        def is_col_empty(j):
            return all(image[i][j] == '0' for i in range(m))

        def is_row_empty(i):
            return all(image[i][j] == '0' for j in range(n))

        def binarySearch(low, high, is_empty, search_min):
            while low < high:
                mid = (low + high) // 2
                if is_empty(mid):
                    if search_min:
                        low = mid + 1 
                    else:
                        high = mid   
                else:
                    if search_min:
                        high = mid   
                    else:
                        low = mid + 1  
            return low

        left = binarySearch(0, y, is_col_empty, True)
        right = binarySearch(y + 1, n, is_col_empty, False)
        top = binarySearch(0, x, is_row_empty, True)
        bottom = binarySearch(x + 1, m, is_row_empty, False)

        return (right - left) * (bottom - top)
