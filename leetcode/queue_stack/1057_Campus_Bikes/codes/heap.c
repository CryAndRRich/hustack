#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

typedef struct {
    int dist, worker, bike;
} Entry;

void swap(Entry* a, Entry* b) {
    Entry tmp = *a; *a = *b; *b = tmp;
}

void heapifyDown(Entry* heap, int size, int i) {
    int min = i, l = 2*i+1, r = 2*i+2;
    if (l < size && (
        heap[l].dist < heap[min].dist ||
        (heap[l].dist == heap[min].dist && heap[l].worker < heap[min].worker) ||
        (heap[l].dist == heap[min].dist && heap[l].worker == heap[min].worker && heap[l].bike < heap[min].bike)
    )) min = l;
    if (r < size && (
        heap[r].dist < heap[min].dist ||
        (heap[r].dist == heap[min].dist && heap[r].worker < heap[min].worker) ||
        (heap[r].dist == heap[min].dist && heap[r].worker == heap[min].worker && heap[r].bike < heap[min].bike)
    )) min = r;
    if (min != i) {
        swap(&heap[i], &heap[min]);
        heapifyDown(heap, size, min);
    }
}

void heapifyUp(Entry* heap, int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (
            heap[i].dist < heap[p].dist ||
            (heap[i].dist == heap[p].dist && heap[i].worker < heap[p].worker) ||
            (heap[i].dist == heap[p].dist && heap[i].worker == heap[p].worker && heap[i].bike < heap[p].bike)
        ) {
            swap(&heap[i], &heap[p]);
            i = p;
        } else break;
    }
}

void push(Entry* heap, int* size, Entry val) {
    heap[*size] = val;
    heapifyUp(heap, (*size)++);
}

Entry pop(Entry* heap, int* size) {
    Entry res = heap[0];
    heap[0] = heap[--(*size)];
    heapifyDown(heap, *size, 0);
    return res;
}

int* assignBikes(int** workers, int workersSize, int* workersColSize,
                 int** bikes, int bikesSize, int* bikesColSize, int* returnSize) {
    Entry* heap = malloc(sizeof(Entry) * workersSize * bikesSize);
    int heapSize = 0;

    for (int i = 0; i < workersSize; ++i) {
        for (int j = 0; j < bikesSize; ++j) {
            int d = abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1]);
            push(heap, &heapSize, (Entry){d, i, j});
        }
    }

    bool* assigned = calloc(workersSize, sizeof(bool));
    bool* used = calloc(bikesSize, sizeof(bool));
    int* res = malloc(sizeof(int) * workersSize);
    for (int i = 0; i < workersSize; ++i) res[i] = -1;

    while (heapSize > 0) {
        Entry top = pop(heap, &heapSize);
        if (!assigned[top.worker] && !used[top.bike]) {
            res[top.worker] = top.bike;
            assigned[top.worker] = true;
            used[top.bike] = true;
        }
    }

    free(heap);
    free(assigned);
    free(used);
    *returnSize = workersSize;
    return res;
}
