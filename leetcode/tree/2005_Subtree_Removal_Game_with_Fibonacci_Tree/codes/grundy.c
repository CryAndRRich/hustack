#include <stdbool.h>
#include <stdlib.h>

bool findGameWinner(int n) {
    if (n == 1) return false;
    if (n == 2) return true;
    int* g = (int*)calloc(n + 1, sizeof(int));
    g[0] = 1;
    g[1] = 0;
    g[2] = 1;
    for (int i = 3; i <= n; i++) {
        g[i] = (g[i - 1] + 1) ^ (g[i - 2] + 1);
    }
    bool result = g[n] != 0;
    free(g);
    return result;
}
