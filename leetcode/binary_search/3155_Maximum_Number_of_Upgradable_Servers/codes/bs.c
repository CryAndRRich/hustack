#include <stdlib.h>

int* maxUpgrades(int* count, int countSize, int* upgrade, int upgradeSize, int* sell, int sellSize, int* money, int moneySize, int* returnSize) {
    int* result = (int*)malloc(sizeof(int) * countSize);

    for (int i = 0; i < countSize; ++i) {
        int low = 0, high = count[i], best = 0;
        while (low <= high) {
            int mid = (low + high) / 2;
            int needSell = count[i] - mid;
            long long total = (long long)money[i] + (long long)needSell * sell[i];
            if ((long long)mid * upgrade[i] <= total) {
                best = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        result[i] = best;
    }

    *returnSize = countSize;
    return result;
}
