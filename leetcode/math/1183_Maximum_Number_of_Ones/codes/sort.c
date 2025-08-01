#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int count;
    int row, col;
} Entry;

int cmp(const void* a, const void* b) {
    return ((Entry*)b) -> count - ((Entry*)a) -> count;
}

int maximumNumberOfOnes(int width, int height, int sideLength, int maxOnes) {
    int freq[101][101] = {0};  
    for (int i = 0; i < sideLength; ++i) {
        for (int j = 0; j < sideLength; ++j) {
            freq[i][j] = ((width - i + sideLength - 1) / sideLength) *
                         ((height - j + sideLength - 1) / sideLength);
        }
    }

    Entry entries[10001];
    int k = 0;
    for (int i = 0; i < sideLength; ++i) {
        for (int j = 0; j < sideLength; ++j) {
            entries[k].row = i;
            entries[k].col = j;
            entries[k].count = freq[i][j];
            ++k;
        }
    }

    qsort(entries, k, sizeof(Entry), cmp);

    int res = 0;
    for (int i = 0; i < maxOnes && i < k; ++i) {
        res += entries[i].count;
    }

    return res;
}