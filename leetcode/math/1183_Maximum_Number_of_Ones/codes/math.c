#include <stdio.h>

int min(int a, int b) {
    return a < b ? a : b;
}

int maximumNumberOfOnes(int width, int height, int sideLength, int maxOnes) {
    int hBlock = height / sideLength;
    int wBlock = width / sideLength;
    int hRemain = height % sideLength;
    int wRemain = width % sideLength;

    int answer = maxOnes * hBlock * wBlock;
    int remain = maxOnes;

    int cnt1 = min(hRemain * wRemain, remain);
    answer += (hBlock + wBlock + 1) * cnt1;
    remain -= cnt1;

    int cnt2, cnt3;
    if (hBlock > wBlock) {
        cnt2 = min((wRemain * sideLength) - (hRemain * wRemain), remain);
        answer += hBlock * cnt2;
        remain -= cnt2;

        cnt3 = min((hRemain * sideLength) - (hRemain * wRemain), remain);
        answer += wBlock * cnt3;
        remain -= cnt3;
    } else {
        cnt2 = min((hRemain * sideLength) - (hRemain * wRemain), remain);
        answer += wBlock * cnt2;
        remain -= cnt2;

        cnt3 = min((wRemain * sideLength) - (hRemain * wRemain), remain);
        answer += hBlock * cnt3;
        remain -= cnt3;
    }

    return answer;
}
