#include <stdio.h>
#include <stdlib.h>

int cmp(const void *a, const void *b) {
    int *segA = *(int**)a;
    int *segB = *(int**)b;
    if (segA[1] == segB[1])
        return segA[0] - segB[0];
    return segA[1] - segB[1];
}


int binarySearch(int **segments, int left, int right, int target) {
    int ans = -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (segments[mid][1] < target) {
            ans = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return ans;
}

int main() {
    int n;
    scanf("%d", &n);
    
    int **segments = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        segments[i] = (int*)malloc(2 * sizeof(int));
        scanf("%d %d", &segments[i][0], &segments[i][1]);
    }
    
    qsort(segments, n, sizeof(int*), cmp);
    
    int *dp = (int*)malloc((n + 1) * sizeof(int));
    dp[0] = 0;
    for (int i = 0; i < n; i++) {
        int ind = binarySearch(segments, 0, i - 1, segments[i][0]);
        int candidate = 1; 
        if (ind != -1) {
            candidate += dp[ind + 1];
        }
        dp[i + 1] = dp[i] > candidate ? dp[i] : candidate;
    }
    
    printf("%d", dp[n]);
    
    for (int i = 0; i < n; i++) {
        free(segments[i]);
    }
    free(segments);
    free(dp);
    
    return 0;
}
