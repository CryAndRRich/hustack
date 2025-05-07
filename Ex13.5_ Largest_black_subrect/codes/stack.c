#include <stdio.h>
#include <string.h>

#define MAXN 1000
#define MAXM 1000

int A[MAXN][MAXM];
int h[MAXM];
int st[MAXM + 1];

int largestRectangleHistogram(int m) {
    int top = 0;
    int maxArea = 0;
    for (int i = 0; i <= m; ++i) {
        int curH = (i < m ? h[i] : 0);
        while (top > 0 && h[st[top-1]] > curH) {
            int height = h[st[--top]];
            int width = (top == 0 ? i : (i - st[top-1] - 1));
            int area = height * width;
            if (area > maxArea) maxArea = area;
        }
        st[top++] = i;
    }
    return maxArea;
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            scanf("%d", &A[i][j]);

    memset(h, 0, sizeof(int) * m);
    int best = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            h[j] = (A[i][j] == 1 ? h[j] + 1 : 0);
        }
        int area = largestRectangleHistogram(m);
        if (area > best) best = area;
    }
    printf("%d", best);
    return 0;
}
