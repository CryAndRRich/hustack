#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int time;
    int index;
} Event;

int cmpEvent(const void* a, const void* b) {
    Event *ea = (Event*)a;
    Event *eb = (Event*)b;
    if (ea -> time != eb -> time) return ea -> time - eb -> time;
    return ea -> index - eb -> index;
}

long long maxArea(int height, int* positions, int positionsSize, char* directions) {
    int n = positionsSize;
    long long total_pos = 0;
    int velocity = 0;
    Event *events = malloc(sizeof(Event) * n * 2);
    int ecount = 0;

    for (int i = 0; i < n; i++) {
        total_pos += positions[i];
        if (directions[i] == 'U') {
            velocity += 1;
            events[ecount++] = (Event){height - positions[i], i};
            events[ecount++] = (Event){height - positions[i] + height, i};
        } else {
            velocity -= 1;
            events[ecount++] = (Event){positions[i], i};
            events[ecount++] = (Event){positions[i] + height, i};
        }
    }

    qsort(events, ecount, sizeof(Event), cmpEvent);

    long long best = total_pos;
    int current_time = 0;

    for (int i = 0; i < ecount;) {
        int t = events[i].time;
        long long dt = (long long)t - current_time;
        total_pos += (long long)velocity * dt;
        current_time = t;
        if (total_pos > best) best = total_pos;

        while (i < ecount && events[i].time == t) {
            int idx = events[i].index;
            if (directions[idx] == 'U') {
                directions[idx] = 'D';
                velocity -= 2;
            } else {
                directions[idx] = 'U';
                velocity += 2;
            }
            i++;
        }
    }

    free(events);
    return best;
}
