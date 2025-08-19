#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int kEmptySlots(int* flowers, int flowersSize, int k) {
    int *days = (int*)malloc(sizeof(int) * flowersSize);
    for (int i = 0; i < flowersSize; i++) {
        days[flowers[i] - 1] = i + 1;
    }

    int left = 0, right = k + 1;
    int res = INT_MAX;

    for (int i = 0; right < flowersSize; i++) {
        if (days[i] < days[left] || days[i] <= days[right]) {
            if (i == right) {
                int maxDay = days[left] > days[right] ? days[left] : days[right];
                if (maxDay < res) res = maxDay;
            }
            left = i;
            right = i + k + 1;
        }
    }

    free(days);
    return (res == INT_MAX) ? -1 : res;
}
