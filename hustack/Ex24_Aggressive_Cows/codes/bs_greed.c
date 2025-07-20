#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAXN 100005

int pos[MAXN];

int cmp(const void* a, const void* b) {
    int x = *(int*)a;
    int y = *(int*)b;
    return x - y;
}

bool greedy(int* pos, int N, int C, int dist) {
    int count = 1;
    int last_pos = pos[0];
    for (int i = 1; i < N; ++i) {
        if (pos[i] - last_pos >= dist) {
            count++;
            last_pos = pos[i];
        }
        if (count >= C) return true;
    }
    return false;
}

int binary_search(int* pos, int N, int C) {
    qsort(pos, N, sizeof(int), cmp);

    int low = 1;
    int high = pos[N - 1] - pos[0];
    int ans = 0;

    while (low <= high) {
        int mid = (low + high) / 2;
        if (greedy(pos, N, C, mid)) {
            ans = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return ans;
}

int main() {
    int T;
    scanf("%d", &T);

    while (T--) {
        int N, C;
        scanf("%d %d", &N, &C);

        for (int i = 0; i < N; ++i) {
            scanf("%d", &pos[i]);
        }

        int res = binary_search(pos, N, C);
        printf("%d", res);
    }

    return 0;
}
