#include <stdio.h>

int totalReplacements(int* ranks, int ranksSize) {
    int best = ranks[0], cnt = 0;
    for (int i = 1; i < ranksSize; i++) {
        if (ranks[i] < best) {
            best = ranks[i];
            cnt++;
        }
    }
    return cnt;
}
