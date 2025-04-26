#include <stdio.h>
#include <stdlib.h>

int main() {
    int n, m;
    scanf("%d %d", &n, &m);

    int **A = (int**)malloc((n + 1) * sizeof(int*));
    int **S = (int**)malloc((n + 1) * sizeof(int*));
    for (int i = 0; i <= n; i++) {
        A[i] = (int*)malloc((m + 1) * sizeof(int));
        S[i] = (int*)malloc((m + 1) * sizeof(int));
    }

    for (int i = 0; i <= n; i++) {
        S[i][0] = 0;
    }
    for (int j = 0; j <= m; j++) {
        S[0][j] = 0;
    }

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            scanf("%d", &A[i][j]);
        }
    }

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            S[i][j] = A[i][j] + S[i - 1][j] + S[i][j - 1] - S[i - 1][j - 1];
        }
    }

    int maxArea = 0;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (S[i][j] == i * j) {  
                int area = i * j;
                if (area > maxArea) {
                    maxArea = area;
                }
            }
        }
    }

    printf("%d", maxArea);

    for (int i = 0; i <= n; i++) {
        free(A[i]);
        free(S[i]);
    }
    free(A);
    free(S);

    return 0;
}
