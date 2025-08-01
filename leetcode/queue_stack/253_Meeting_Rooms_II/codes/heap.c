#include <stdio.h>
#include <stdlib.h>

void swap(int* a, int* b) { 
    int t = *a; *a = *b; *b = t; 
}

void heapifyDown(int heap[], int i, int size) {
    int smallest = i;
    int l = 2 * i + 1, r = 2 * i + 2;
    if (l < size && heap[l] < heap[smallest]) smallest = l;
    if (r < size && heap[r] < heap[smallest]) smallest = r;
    if (smallest != i) {
        swap(&heap[i], &heap[smallest]);
        heapifyDown(heap, smallest, size);
    }
}

void heapifyUp(int heap[], int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (heap[p] <= heap[i]) break;
        swap(&heap[p], &heap[i]);
        i = p;
    }
}

void push(int heap[], int* size, int val) {
    heap[(*size)++] = val;
    heapifyUp(heap, *size - 1);
}

int pop(int heap[], int* size) {
    int val = heap[0];
    heap[0] = heap[--(*size)];
    heapifyDown(heap, 0, *size);
    return val;
}

int compare(const void* a, const void* b) {
    return ((int**)a)[0][0] - ((int**)b)[0][0];
}

int minMeetingRooms(int** intervals, int intervalsSize, int* intervalsColSize) {
    if (intervalsSize == 0) return 0;

    qsort(intervals, intervalsSize, sizeof(int*), compare);

    int heap[intervalsSize];
    int heapSize = 0;

    push(heap, &heapSize, intervals[0][1]);

    for (int i = 1; i < intervalsSize; i++) {
        if (heapSize > 0 && intervals[i][0] >= heap[0])
            pop(heap, &heapSize);
        push(heap, &heapSize, intervals[i][1]);
    }

    return heapSize;
}
