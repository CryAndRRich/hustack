#include <stdlib.h>
#include <limits.h>

long long maxCoins(int* lane1, int lane1Size, int* lane2, int lane2Size) {
    int n = lane1Size;
    long long NEG = LLONG_MIN / 4;
    long long dpPrev[2][3];
    for (int l = 0; l < 2; l++) {
        for (int s = 0; s < 3; s++) {
            dpPrev[l][s] = NEG;
        }
    }
    long long ans = NEG;
    for (int i = 0; i < n; i++) {
        long long dpCurr[2][3];
        for (int l = 0; l < 2; l++) {
            for (int s = 0; s < 3; s++) {
                dpCurr[l][s] = NEG;
            }
        }
        for (int l = 0; l < 2; l++) {
            int gain = (l == 0? lane1[i] : lane2[i]);
            for (int s = 0; s < 3; s++) {
                long long best = NEG;
                int startS = (l == 0 ? 0 : 1);
                if (s == startS) {
                    if ((long long)gain > best) best = (long long)gain;
                }
                if (dpPrev[l][s] != NEG) {
                    long long cand = dpPrev[l][s] + (long long)gain;
                    if (cand > best) best = cand;
                }
                if (s >= 1 && dpPrev[1 - l][s - 1] != NEG) {
                    long long cand = dpPrev[1 - l][s - 1] + (long long)gain;
                    if (cand > best) best = cand;
                }
                dpCurr[l][s] = best;
                if (best != NEG && best > ans) ans = best;
            }
        }
        for (int l = 0; l < 2; l++) {
            for (int s = 0; s < 3; s++) {
                dpPrev[l][s] = dpCurr[l][s];
            }
        }
    }
    return ans;
}
