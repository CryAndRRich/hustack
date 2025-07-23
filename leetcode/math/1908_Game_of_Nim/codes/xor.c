#include <stdbool.h>

bool nimGame(int* piles, int pilesSize) {
    int xorSum = 0;
    for (int i = 0; i < pilesSize; i++) {
        xorSum ^= piles[i];
    }
    return xorSum != 0;
}
