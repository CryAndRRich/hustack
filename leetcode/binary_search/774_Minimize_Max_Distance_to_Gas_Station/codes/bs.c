#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

bool isPossible(int* stations, int size, int k, double D) {
    int count = 0;
    for (int i = 0; i < size - 1; ++i) {
        double dist = stations[i + 1] - stations[i];
        count += (int)(dist / D);
    }
    return count <= k;
}

double minmaxGasDist(int* stations, int stationsSize, int k) {
    double lo = 0, hi = stations[stationsSize - 1] - stations[0];
    double eps = 1e-6;

    while (hi - lo > eps) {
        double mid = (lo + hi) / 2;
        if (isPossible(stations, stationsSize, k, mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return lo;
}
