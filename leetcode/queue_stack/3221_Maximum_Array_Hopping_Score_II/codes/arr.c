#include <stdio.h>

int maxScore(int* nums, int n) {
    int max_v = nums[n - 1];
    int max_i = n - 1;

    int idx[n];
    for (int i = n - 2; i >= 0; --i) {
        idx[i] = max_i;
        if (nums[i] > max_v) {
            max_v = nums[i];
            max_i = i;
        }
    }

    int score = 0, i = 0;
    while (i != n - 1) {
        int j = idx[i];
        score += (j - i) * nums[j];
        i = j;
    }
    return score;
}