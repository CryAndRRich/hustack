#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int n, m;
bool graph[101][101];
int ans = 0;
int best[101];
int curr[101];

void bron_kerbosch(int R[], int r_size, int P[], int p_size, int X[], int x_size) {
    if (p_size == 0 && x_size == 0) {
        if (r_size > ans) {
            ans = r_size;
            for (int i = 0; i < r_size; i++)
                best[i] = R[i];
        }
        return;
    }

    int p_copy[101], newP[101], newX[101], newR[101];
    for (int i = 0; i < p_size; i++) p_copy[i] = P[i];

    for (int i = 0; i < p_size; i++) {
        int v = p_copy[i];

        int nr_size = r_size;
        for (int j = 0; j < r_size; j++) newR[j] = R[j];
        newR[nr_size++] = v;

        int np_size = 0, nx_size = 0;
        for (int j = 0; j < p_size; j++) {
            if (graph[v][P[j]])
                newP[np_size++] = P[j];
        }
        for (int j = 0; j < x_size; j++) {
            if (graph[v][X[j]])
                newX[nx_size++] = X[j];
        }

        bron_kerbosch(newR, nr_size, newP, np_size, newX, nx_size);

        int new_p_size = 0;
        for (int j = 0; j < p_size; j++) {
            if (P[j] != v) P[new_p_size++] = P[j];
        }
        p_size = new_p_size;
        X[x_size++] = v;
    }
}

int main() {
    scanf("%d %d", &n, &m);
    for (int i = 0; i < m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        graph[u][v] = true;
        graph[v][u] = true;
    }

    int P[101], X[101];
    for (int i = 1; i <= n; i++) P[i - 1] = i;
    bron_kerbosch(curr, 0, P, n, X, 0);

    printf("%d\n", ans);
    for (int i = 0; i < ans; i++) {
        printf("%d ", best[i]);
    }
    return 0;
}
