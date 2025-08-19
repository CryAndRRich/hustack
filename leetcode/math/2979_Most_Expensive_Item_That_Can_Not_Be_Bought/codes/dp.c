#include <stdio.h>
#include <stdbool.h>
#include <string.h>

int mostExpensiveItem(int primeOne, int primeTwo) {
    int max_val = primeOne * primeTwo;
    bool dp[max_val + 1];
    memset(dp, 0, sizeof(dp));
    dp[0] = true;

    for (int i = 1; i <= max_val; ++i) {
        if (i >= primeOne && dp[i - primeOne])
            dp[i] = true;
        if (i >= primeTwo && dp[i - primeTwo])
            dp[i] = true;
    }

    for (int i = max_val; i >= 0; --i) {
        if (!dp[i])
            return i;
    }

    return 0; 
}
