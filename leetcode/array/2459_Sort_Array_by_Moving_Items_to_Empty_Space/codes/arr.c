#include <stdio.h>
#include <stdlib.h>

int getCountSorting(int* nums, int n, int offset) {
    int* copy = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) copy[i] = nums[i];

    int count = 0;

    for (int i = 0; i < n; ++i) {
        if (copy[i] == 0 || i == copy[i] - offset) continue;

        int j = i;
        count++;
        while (copy[j] > 0) {
            copy[j] = -copy[j];
            j = -copy[j] - offset;
            if (j == i) {
                count++;
                break;
            }
        }
    }

    free(copy);
    return count;
}

int sortArray(int* nums, int n) {
    int* numsCopy1 = (int*)malloc(sizeof(int) * n);
    int* numsCopy2 = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) {
        numsCopy1[i] = nums[i];
        numsCopy2[i] = nums[i];
    }

    int res1 = getCountSorting(numsCopy1, n, 0);
    int res2 = getCountSorting(numsCopy2, n, 1);

    free(numsCopy1);
    free(numsCopy2);
    return res1 < res2 ? res1 : res2;
}
