#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int kEmptySlots(int* flowers, int flowersSize, int k) {
    if (flowers == NULL || flowersSize == 0 || k + 2 > flowersSize) {
        return -1;
    }
    int *days = (int*)malloc(sizeof(int) * flowersSize);
    for (int i = 0; i < flowersSize; i++) {
        days[flowers[i] - 1] = i + 1;
    }

    int *dayMin = (int*)malloc(sizeof(int) * flowersSize);
    int front = 0, back = 0;
    int res = INT_MAX;

    for (int i = 1; i <= k; i++) {
        while (back > front && days[dayMin[back - 1]] > days[i]) {
            back--;
        }
        dayMin[back++] = i;
    }

    for (int i = k + 1; i < flowersSize; i++) {
        int maxDay = days[i - k - 1] > days[i] ? days[i - k - 1] : days[i];
        if (k == 0) {
            if (maxDay < res) res = maxDay;
        } else if (maxDay < days[dayMin[front]]) {
            if (maxDay < res) res = maxDay;
        }

        if (front < back && dayMin[front] == i - k) {
            front++;
        }
        while (back > front && days[dayMin[back - 1]] > days[i]) {
            back--;
        }
        dayMin[back++] = i;
    }

    free(days);
    free(dayMin);
    return (res == INT_MAX) ? -1 : res;
}
