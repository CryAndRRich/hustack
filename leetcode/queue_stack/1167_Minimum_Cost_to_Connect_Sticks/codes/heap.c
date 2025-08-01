#include <stdio.h>
#include <stdlib.h>

void heapify(int heap[], int n, int i) {
    int smallest = i, left = 2*i+1, right = 2*i+2;

    if (left < n && heap[left] < heap[smallest]) smallest = left;
    if (right < n && heap[right] < heap[smallest]) smallest = right;

    if (smallest != i) {
        int temp = heap[i]; heap[i] = heap[smallest]; heap[smallest] = temp;
        heapify(heap, n, smallest);
    }
}

void buildMinHeap(int heap[], int n) {
    for (int i = n/2 - 1; i >= 0; i--)
        heapify(heap, n, i);
}

int extractMin(int heap[], int* size) {
    int root = heap[0];
    heap[0] = heap[--(*size)];
    heapify(heap, *size, 0);
    return root;
}

void insertHeap(int heap[], int* size, int val) {
    int i = (*size)++;
    heap[i] = val;
    while (i != 0 && heap[(i - 1)/2] > heap[i]) {
        int temp = heap[i];
        heap[i] = heap[(i - 1)/2];
        heap[(i - 1)/2] = temp;
        i = (i - 1)/2;
    }
}

int connectSticks(int* sticks, int sticksSize) {
    if (sticksSize <= 1) return 0;

    int heap[10000], size = sticksSize;
    for (int i = 0; i < sticksSize; i++) heap[i] = sticks[i];

    buildMinHeap(heap, size);
    int cost = 0;

    while (size > 1) {
        int first = extractMin(heap, &size);
        int second = extractMin(heap, &size);
        int newStick = first + second;
        cost += newStick;
        insertHeap(heap, &size, newStick);
    }

    return cost;
}
