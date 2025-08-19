#include <stdio.h>
#include <stdlib.h>

#define MAXN 100005

typedef struct {
    int index;
    int gcd;
} Pair;

int gcd(int a, int b) {
    while (b) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

long long maxGcdSum(int* nums, int numsSize, int k) {
    long long prefix[MAXN] = {0};
    for (int i = 0; i < numsSize; ++i)
        prefix[i + 1] = prefix[i] + nums[i];

    long long best = 0;
    Pair gcds[MAXN];
    int gsize = 0;

    for (int i = 0; i < numsSize; ++i) {
        Pair new_gcds[MAXN];
        int new_size = 0;

        gcds[gsize++] = (Pair){i, nums[i]};
        for (int j = 0; j < gsize; ++j) {
            int ng = gcd(gcds[j].gcd, nums[i]);
            if (new_size == 0 || new_gcds[new_size - 1].gcd != ng)
                new_gcds[new_size++] = (Pair){gcds[j].index, ng};
        }

        for (int j = 0; j < new_size; ++j) {
            int len = i - new_gcds[j].index + 1;
            if (len < k) break;
            long long sum = prefix[i + 1] - prefix[new_gcds[j].index];
            long long val = sum * new_gcds[j].gcd;
            if (val > best) best = val;
        }

        for (int j = 0; j < new_size; ++j) gcds[j] = new_gcds[j];
        gsize = new_size;
    }

    return best;
}
