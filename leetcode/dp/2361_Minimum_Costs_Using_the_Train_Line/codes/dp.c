#include <stdlib.h>
#include <string.h>

long long* minimumCosts(int* regular, int regularSize, int* express, int expressSize, int expressCost, int* returnSize) {
    int n = regularSize;
    long long* dpReg = (long long*)malloc((n + 1) * sizeof(long long));
    long long* dpExp = (long long*)malloc((n + 1) * sizeof(long long));
    long long* res = (long long*)malloc(n * sizeof(long long));
    dpReg[0] = 0;
    dpExp[0] = expressCost;
    for (int i = 1; i <= n; i++) {
        long long fromReg = dpReg[i-1] + regular[i-1];
        long long fromExp = dpExp[i-1] + regular[i-1];
        dpReg[i] = fromReg < fromExp ? fromReg : fromExp;

        long long fromExpStay = dpExp[i-1] + express[i-1];
        long long fromRegSwitch = dpReg[i-1] + expressCost + express[i-1];
        dpExp[i] = fromExpStay < fromRegSwitch ? fromExpStay : fromRegSwitch;

        res[i-1] = dpReg[i] < dpExp[i] ? dpReg[i] : dpExp[i];
    }
    free(dpReg);
    free(dpExp);
    *returnSize = n;
    return res;
}
