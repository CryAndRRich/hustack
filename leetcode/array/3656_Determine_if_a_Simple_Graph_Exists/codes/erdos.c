#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int cmp_desc(const void* a, const void* b) {
    return (*(int*)b - *(int*)a);
}

bool simpleGraphExists(int* degrees, int n) {
    qsort(degrees, n, sizeof(int), cmp_desc);

    long long total = 0;
    for (int i = 0; i < n; i++) total += degrees[i];
    if (total % 2 == 1) return false;

    long long* prefix = calloc(n+1, sizeof(long long));
    for (int i = 0; i < n; i++) prefix[i+1] = prefix[i] + degrees[i];

    for (int k = 1; k <= n; k++) {
        long long left = prefix[k];
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (degrees[mid] >= k) lo = mid + 1;
            else hi = mid;
        }
        int j = lo;

        long long right = 1LL * k * (k - 1);
        if (j > k) right += 1LL * k * (j - k);
        right += prefix[n] - prefix[j > k ? j : k];

        if (left > right) {
            free(prefix);
            return false;
        }
    }
    free(prefix);
    return true;
}
