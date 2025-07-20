#include <stdlib.h>
#include <string.h>
#include <limits.h>

int dfs(int* debt, int sz, int start) {
    while (start < sz && debt[start] == 0) start++;
    if (start == sz) return 0;

    int res = INT_MAX;
    for (int i = start + 1; i < sz; i++) {
        if (debt[start] * debt[i] < 0) {
            debt[i] += debt[start];
            int next = dfs(debt, sz, start + 1);
            if (next + 1 < res) res = next + 1;
            debt[i] -= debt[start];
        }
    }
    return res;
}

int minTransfers(int** transactions, int transactionsSize, int* transactionsColSize){
    int bal[12] = {0}; 
    for (int i = 0; i < transactionsSize; i++) {
        int from = transactions[i][0], to = transactions[i][1], amt = transactions[i][2];
        bal[from] -= amt;
        bal[to] += amt;
    }

    int debt[12], sz = 0;
    for (int i = 0; i < 12; i++) {
        if (bal[i] != 0) debt[sz++] = bal[i];
    }

    return dfs(debt, sz, 0);
}
