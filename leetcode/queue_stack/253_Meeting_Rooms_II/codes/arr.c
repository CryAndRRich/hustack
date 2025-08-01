#include <stdlib.h>

int compareInt(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int minMeetingRooms(int** intervals, int intervalsSize, int* intervalsColSize) {
    int* starts = (int*)malloc(sizeof(int) * intervalsSize);
    int* ends = (int*)malloc(sizeof(int) * intervalsSize);
    for (int i = 0; i < intervalsSize; i++) {
        starts[i] = intervals[i][0];
        ends[i] = intervals[i][1];
    }
    qsort(starts, intervalsSize, sizeof(int), compareInt);
    qsort(ends, intervalsSize, sizeof(int), compareInt);

    int rooms = 0, end_ptr = 0;
    for (int i = 0; i < intervalsSize; i++) {
        if (starts[i] < ends[end_ptr])
            rooms++;
        else
            end_ptr++;
    }

    free(starts);
    free(ends);
    return rooms;
}
