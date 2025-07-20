#include <stdio.h>

int canDivide(int* sweetness, int n, int k, int minSweet) {
    int count = 0, total = 0;
    for (int i = 0; i < n; ++i) {
        total += sweetness[i];
        if (total >= minSweet) {
            count++;
            total = 0;
        }
    }
    return count >= k + 1;
}

int maximizeSweetness(int* sweetness, int n, int k) {
    int sum = 0;
    for (int i = 0; i < n; ++i) sum += sweetness[i];

    int left = 1, right = sum;
    while (left < right) {
        int mid = (left + right + 1) / 2;
        if (canDivide(sweetness, n, k, mid)) {
            left = mid;
        } else {
            right = mid - 1;
        }
    }
    return left;
}