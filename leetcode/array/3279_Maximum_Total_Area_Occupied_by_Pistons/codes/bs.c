#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int time;
    int delta;
} Event;

int cmpEvent(const void* a, const void* b) {
    return ((Event*)a) -> time - ((Event*)b) -> time;
}

long long maxArea(int height, int* positions, int positionsSize, char* directions) {
    int n = positionsSize;
    int cntU = 0;
    for (int i = 0; i < n; i++) if (directions[i] == 'U') cntU++;

    long long initSum = 0;
    for (int i = 0; i < n; i++) initSum += positions[i];

    auto int can(long long target) {
        Event* events = (Event*)malloc(sizeof(Event) * (2 * n + 1));
        int m = 0;
        long long mx = initSum;
        int u = cntU;

        for (int i = 0; i < n; i++) {
            if (directions[i] == 'U') {
                events[m++] = (Event){height - positions[i], -1};
                events[m++] = (Event){2 * height - positions[i], 1};
            } else {
                events[m++] = (Event){height + positions[i], -1};
                events[m++] = (Event){positions[i], 1};
            }
        }
        events[m++] = (Event){0, 0};

        qsort(events, m, sizeof(Event), cmpEvent);

        for (int i = 0; i < m - 1; i++) {
            if (mx >= target) {
                free(events);
                return 1;
            }
            mx += (long long)(events[i + 1].time - events[i].time) * (2 * u - n);
            u += events[i + 1].delta;
        }
        free(events);
        return mx >= target;
    }

    long long low = initSum;
    long long high = (long long)n * height;
    long long ans = low;

    while (low <= high) {
        long long mid = (low + high) / 2;
        if (can(mid)) {
            ans = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return ans;
}
