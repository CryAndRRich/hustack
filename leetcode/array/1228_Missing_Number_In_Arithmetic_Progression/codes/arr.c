#include <limits.h>
#include <stdlib.h>

int missingNumber(int* arr, int arrSize) {
    int d = INT_MAX;

    for (int i = 1; i < arrSize; i++) {
        int diff = arr[i] - arr[i - 1];
        if (diff != 0 && abs(diff) < abs(d)) {
            d = diff;
        }
    }
    if (d == INT_MAX) d = 0;

    for (int i = 0; i < arrSize - 1; i++) {
        if (arr[i + 1] - arr[i] != d) {
            return arr[i] + d;
        }
    }
    return arr[0]; 
}
