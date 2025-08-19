#include <stdbool.h>

bool isColEmpty(char** image, int m, int col) {
    for (int i = 0; i < m; ++i) {
        if (image[i][col] == '1') return false;
    }
    return true;
}

bool isRowEmpty(char** image, int n, int row) {
    for (int j = 0; j < n; ++j) {
        if (image[row][j] == '1') return false;
    }
    return true;
}

int binarySearch(char** image, int low, int high, bool (*isEmpty)(char**, int, int), int dim, bool searchMin) {
    while (low < high) {
        int mid = (low + high) / 2;
        if (isEmpty(image, dim, mid) == searchMin)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

int minArea(char** image, int imageSize, int* imageColSize, int x, int y) {
    int m = imageSize, n = imageColSize[0];

    int left = binarySearch(image, 0, y, isColEmpty, m, true);
    int right = binarySearch(image, y + 1, n, isColEmpty, m, false);
    int top = binarySearch(image, 0, x, isRowEmpty, n, true);
    int bottom = binarySearch(image, x + 1, m, isRowEmpty, n, false);

    return (right - left) * (bottom - top);
}
