#include <stdio.h>
#include <stdlib.h>

int gcd(int a, int b) {
    while (b) {
        int tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}

int log2_floor(int x) {
    int res = 0;
    while ((1 << (res + 1)) <= x) res++;
    return res;
}

long long maxGcdSum(int* nums, int numsSize, int k) {
    int st[100005][20];
    long long prefix[100005] = {0};

    for (int i = 0; i < numsSize; ++i) {
        prefix[i + 1] = prefix[i] + nums[i];
        st[i][0] = nums[i];
    }

    for (int j = 1; (1 << j) <= numsSize; ++j) {
        for (int i = 0; i + (1 << j) <= numsSize; ++i) {
            st[i][j] = gcd(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
        }
    }

    long long res = 0;
    for (int i = 0; i < numsSize; ++i) {
        int want = nums[i], left = i + 1;
        while (left <= numsSize && (prefix[numsSize] - prefix[i]) * want > res) {
            int lo = left, hi = numsSize - 1, mid, g;
            while (lo <= hi) {
                mid = (lo + hi) / 2;
                int len = mid - i + 1;
                int j = log2_floor(len);
                g = gcd(st[i][j], st[mid - (1 << j) + 1][j]);
                if (g == want) lo = mid + 1;
                else hi = mid - 1;
            }
            if (hi - i + 1 >= k) {
                long long sum = prefix[hi + 1] - prefix[i];
                if (sum * want > res) res = sum * want;
            }
            if (lo >= numsSize) break;
            want = gcd(want, nums[lo]);
            left = lo + 1;
        }
    }

    return res;
}
