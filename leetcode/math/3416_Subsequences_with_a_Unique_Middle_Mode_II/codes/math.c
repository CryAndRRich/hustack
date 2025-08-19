#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOD 1000000007
#define RM ((MOD + 1) / 2)

int mul(long long x, long long y) {
    return (int)((x * y) % MOD);
}

int sqr(int x) {
    return mul(x, x);
}

int add(int x, int y) {
    x += y;
    if (x >= MOD) x -= MOD;
    return x;
}

int sub(int x, int y) {
    x -= y;
    if (x < 0) x += MOD;
    return x;
}

int C2(int n) {
    return mul(mul(n, n - 1), RM);
}

int count(int* v, int* c, int n, int m, int countIfEqual) {
    int* t = (int*)calloc(m, sizeof(int));
    int R2 = 0, RT = 0, R2T = 0, r = 0;

    for (int i = 0; i < m; ++i) R2 = add(R2, sqr(c[i]));

    for (int i = 0; i < n; ++i) {
        int x = v[i];
        int rx = c[x] - t[x];
        int r2 = sub(R2, sqr(rx));
        int rt = sub(RT, mul(rx, t[x]));
        int r2t = sub(R2T, mul(sqr(rx), t[x]));
        int p = n - i - rx;
        int sumt = i - t[x];

        int temp = mul(sub(sqr(p), r2), sumt);
        temp = sub(temp, mul(2LL * p % MOD, rt));
        temp = add(temp, mul(2, r2t));
        temp = mul(temp, mul(t[x], RM));
        r = add(r, temp);
        r = add(r, mul(C2(t[x]), C2(p)));
        rx -= 1;
        r = add(r, mul(C2(t[x]), mul(rx, p)));

        if (countIfEqual) {
            r = add(r, mul(mul(t[x], sumt), mul(rx, p)));
            r = add(r, mul(C2(t[x]), C2(rx)));
        }

        t[x]++;
        R2 = add(r2, sqr(rx));
        RT = add(rt, mul(rx, t[x]));
        R2T = add(r2t, mul(sqr(rx), t[x]));
    }

    free(t);
    return r;
}

int cmpInt(const void* a, const void* b) {
    return (*(int*)a) - (*(int*)b);
}

int subsequencesWithMiddleMode(int* nums, int numsSize) {
    int* copy = (int*)malloc(numsSize * sizeof(int));
    memcpy(copy, nums, numsSize * sizeof(int));

    qsort(copy, numsSize, sizeof(int), cmpInt);
    int* uniq = (int*)malloc(numsSize * sizeof(int));
    int m = 0;
    for (int i = 0; i < numsSize; ++i) {
        if (i == 0 || copy[i] != copy[i - 1]) {
            uniq[m++] = copy[i];
        }
    }

    int* comp = (int*)malloc(numsSize * sizeof(int));
    for (int i = 0; i < numsSize; ++i) {
        int left = 0, right = m - 1, mid;
        while (left <= right) {
            mid = (left + right) / 2;
            if (uniq[mid] == nums[i]) {
                comp[i] = mid;
                break;
            } else if (uniq[mid] < nums[i]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    int* countArr = (int*)calloc(m, sizeof(int));
    for (int i = 0; i < numsSize; ++i) {
        countArr[comp[i]]++;
    }

    int r1 = count(comp, countArr, numsSize, m, 1);

    for (int i = 0; i < numsSize / 2; ++i) {
        int tmp = comp[i];
        comp[i] = comp[numsSize - 1 - i];
        comp[numsSize - 1 - i] = tmp;
    }

    int r2 = count(comp, countArr, numsSize, m, 0);

    free(copy);
    free(uniq);
    free(comp);
    free(countArr);

    return add(r1, r2);
}
