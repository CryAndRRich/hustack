#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAXN 100005

typedef struct {
    int val;
    int id;
} Entry;

typedef struct {
    Entry data[MAXN];
    int size;
} Heap;

typedef struct {
    Entry stack[MAXN];
    bool deleted[MAXN]; 
    Heap heap;
    int top;
    int next_id;
} MaxStack;

int cmp(Entry a, Entry b) {
    if (a.val != b.val) return a.val - b.val;
    return a.id - b.id;
}

void heapSwap(Heap* h, int i, int j) {
    Entry tmp = h -> data[i];
    h -> data[i] = h -> data[j];
    h -> data[j] = tmp;
}

void heapPush(Heap* h, Entry e) {
    int i = h -> size++;
    h -> data[i] = e;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (cmp(h -> data[p], h -> data[i]) >= 0) break;
        heapSwap(h, i, p);
        i = p;
    }
}

Entry heapTop(Heap* h) {
    return h -> data[0];
}

void heapPop(Heap* h) {
    h -> data[0] = h -> data[--h -> size];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, largest = i;
        if (l < h -> size && cmp(h -> data[l], h -> data[largest]) > 0) largest = l;
        if (r < h -> size && cmp(h -> data[r], h -> data[largest]) > 0) largest = r;
        if (largest == i) break;
        heapSwap(h, i, largest);
        i = largest;
    }
}

MaxStack* maxStackCreate() {
    MaxStack* ms = (MaxStack*)malloc(sizeof(MaxStack));
    ms -> top = 0;
    ms -> heap.size = 0;
    ms -> next_id = 0;
    memset(ms -> deleted, 0, sizeof(ms -> deleted));
    return ms;
}

void maxStackPush(MaxStack* ms, int x) {
    Entry e = {x, ms -> next_id++};
    ms -> stack[ms -> top++] = e;
    heapPush(&ms -> heap, e);
}

int maxStackPop(MaxStack* ms) {
    while (ms -> top > 0 && ms -> deleted[ms -> stack[ms -> top - 1].id])
        ms -> top--;
    Entry e = ms -> stack[--ms -> top];
    ms -> deleted[e.id] = true;
    return e.val;
}

int maxStackTop(MaxStack* ms) {
    while (ms -> top > 0 && ms -> deleted[ms -> stack[ms -> top - 1].id])
        ms -> top--;
    return ms -> stack[ms -> top - 1].val;
}

int maxStackPeekMax(MaxStack* ms) {
    while (ms -> heap.size > 0 && ms -> deleted[heapTop(&ms -> heap).id])
        heapPop(&ms -> heap);
    return heapTop(&ms -> heap).val;
}

int maxStackPopMax(MaxStack* ms) {
    while (ms -> heap.size > 0 && ms -> deleted[heapTop(&ms -> heap).id])
        heapPop(&ms -> heap);
    Entry e = heapTop(&ms -> heap);
    heapPop(&ms -> heap);
    ms -> deleted[e.id] = true;
    return e.val;
}

void maxStackFree(MaxStack* ms) {
    free(ms);
}


/**
 * Your MaxStack struct will be instantiated and called as such:
 * MaxStack* obj = maxStackCreate();
 * maxStackPush(obj, x);
 
 * int param_2 = maxStackPop(obj);
 
 * int param_3 = maxStackTop(obj);
 
 * int param_4 = maxStackPeekMax(obj);
 
 * int param_5 = maxStackPopMax(obj);
 
 * maxStackFree(obj);
*/