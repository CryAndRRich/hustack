#include <stdio.h>
#include <stdlib.h>

typedef long long ll;

void heapifyDown(ll heap[], int size, int i) {
    int smallest = i;
    int left = 2*i + 1, right = 2*i + 2;

    if (left < size && heap[left] < heap[smallest]) smallest = left;
    if (right < size && heap[right] < heap[smallest]) smallest = right;

    if (smallest != i) {
        ll temp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = temp;
        heapifyDown(heap, size, smallest);
    }
}

void heapifyUp(ll heap[], int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap[i] >= heap[parent]) break;
        ll temp = heap[i];
        heap[i] = heap[parent];
        heap[parent] = temp;
        i = parent;
    }
}

ll extractMin(ll heap[], int* size) {
    ll min = heap[0];
    heap[0] = heap[--(*size)];
    heapifyDown(heap, *size, 0);
    return min;
}

void insertHeap(ll heap[], int* size, ll val) {
    heap[(*size)++] = val;
    heapifyUp(heap, *size - 1);
}

ll minEliminationTime(int* timeReq, int timeReqSize, int splitTime) {
    ll heap[timeReqSize * 2];
    int heapSize = 0;

    for (int i = 0; i < timeReqSize; i++)
        insertHeap(heap, &heapSize, timeReq[i]);

    while (heapSize > 1) {
        ll t1 = extractMin(heap, &heapSize);
        ll t2 = extractMin(heap, &heapSize);
        insertHeap(heap, &heapSize, t2 + splitTime);
    }

    return heap[0];
}
