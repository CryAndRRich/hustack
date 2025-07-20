#include <stdio.h>
#include <stdlib.h>

int main() {
    int n;
    scanf("%d", &n);
    int *a = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }

    int *tails = malloc(n * sizeof(int));
    int length = 0;

    for (int i = 0; i < n; i++) {
        int x = a[i];
        int lo = 0, hi = length;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (tails[mid] < x) lo = mid + 1;
            else hi = mid;
        }
        if (lo == length) {
            tails[length++] = x;
        } else {
            tails[lo] = x;
        }
    }

    printf("%d", length);

    free(a);
    free(tails);
    return 0;
}
