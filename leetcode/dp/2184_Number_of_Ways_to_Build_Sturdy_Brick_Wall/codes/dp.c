#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOD 1000000007

int rows[1<<10], rowCount;
int compat[1<<10][1<<10], compCount[1<<10];

void gen(int pos, int mask, int width, int* bricks, int bricksSize) {
    if (pos == width) { rows[rowCount++] = mask; return; }
    for (int i = 0; i < bricksSize; i++) {
        int b = bricks[i];
        if (pos + b <= width) {
            if (pos + b < width) gen(pos + b, mask | (1 << (pos + b)), width, bricks, bricksSize);
            else gen(pos + b, mask, width, bricks, bricksSize);
        }
    }
}

int buildWall(int height, int width, int* bricks, int bricksSize) {
    rowCount = 0;
    gen(0, 0, width, bricks, bricksSize);
    for (int i = 0; i < rowCount; i++) compCount[i] = 0;
    for (int i = 0; i < rowCount; i++)
        for (int j = 0; j < rowCount; j++)
            if ((rows[i] & rows[j]) == 0)
                compat[i][compCount[i]++] = j;
    long long dp[1<<10], ndp[1<<10];
    for (int i = 0; i < rowCount; i++) dp[i] = 1;
    for (int h = 0; h < height - 1; h++) {
        for (int i = 0; i < rowCount; i++) ndp[i] = 0;
        for (int i = 0; i < rowCount; i++)
            for (int k = 0; k < compCount[i]; k++)
                ndp[ compat[i][k] ] = (ndp[ compat[i][k] ] + dp[i]) % MOD;
        for (int i = 0; i < rowCount; i++) dp[i] = ndp[i];
    }
    long long ans = 0;
    for (int i = 0; i < rowCount; i++) ans = (ans + dp[i]) % MOD;
    return (int)ans;
}
