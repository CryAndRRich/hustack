#include <stdbool.h>

bool binarySearch(int* arr, int size, int target) {
    int l = 0, r = size - 1;
    while (l <= r) {
        int mid = (l + r) / 2;
        if (arr[mid] == target) return true;
        else if (arr[mid] < target) l = mid + 1;
        else r = mid - 1;
    }
    return false;
}

int smallestCommonElement(int** mat, int matSize, int* matColSize) {
    int firstRowSize = matColSize[0];

    for (int k = 0; k < firstRowSize; ++k) {
        int num = mat[0][k];
        bool foundInAll = true;

        for (int i = 1; i < matSize; ++i) {
            if (!binarySearch(mat[i], matColSize[i], num)) {
                foundInAll = false;
                break;
            }
        }

        if (foundInAll) return num;
    }

    return -1;
}
