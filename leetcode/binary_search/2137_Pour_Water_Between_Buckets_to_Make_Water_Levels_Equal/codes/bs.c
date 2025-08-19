#include <stdio.h>
#include <math.h>

double equalizeWater(int* buckets, int bucketsSize, int loss) {
    double lo = 0.0, hi = 0.0;
    for (int i = 0; i < bucketsSize; i++)
        if (buckets[i] > hi) hi = buckets[i];
    double eps = 1e-6;
    double factor = (100.0 - loss) / 100.0;

    while (hi - lo > eps) {
        double mid = (lo + hi) / 2.0;
        double need = 0.0, have = 0.0;
        for (int i = 0; i < bucketsSize; i++) {
            if (buckets[i] < mid) need += mid - buckets[i];
            else have += buckets[i] - mid;
        }
        if (have * factor >= need)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}
