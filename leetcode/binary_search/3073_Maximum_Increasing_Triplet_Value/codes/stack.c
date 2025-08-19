#include <stdio.h>
#include <stdlib.h>

static int* global_nums;

typedef struct {
    int *data;
    int top;
} Stack;

void initStack(Stack *s, int n) {
    s -> data = (int*)malloc(sizeof(int) * n);
    s -> top = -1;
}

void push(Stack *s, int val) {
    s -> data[++(s -> top)] = val;
}

void pop(Stack *s) {
    if (s -> top >= 0) s -> top--;
}

int peek(Stack *s) {
    return s -> data[s -> top];
}

int isEmpty(Stack *s) {
    return s -> top == -1;
}

int cmp(const void *a, const void *b) {
    int i = *(int*)a;
    int j = *(int*)b;
    if (global_nums[i] != global_nums[j]) return global_nums[i] - global_nums[j];
    return j - i;  
}

int maximumTripletValue(int *nums, int n) {
    global_nums = nums;

    int *max_right = (int*)calloc(n, sizeof(int));
    int right_max = 0;
    for (int k = n - 1; k >= 0; k--) {
        if (nums[k] >= right_max) {
            right_max = nums[k];
        } else {
            max_right[k] = right_max;
        }
    }

    int *idx = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) idx[i] = i;

    qsort(idx, n, sizeof(int), cmp);

    Stack st;
    initStack(&st, n);

    int max_triplet = 0;
    for (int t = 0; t < n; t++) {
        int j = idx[t];
        while (!isEmpty(&st) && peek(&st) > j) {
            pop(&st);
        }
        if (!isEmpty(&st) && max_right[j] >= 0) {
            int val = nums[peek(&st)] - nums[j] + max_right[j];
            if (val > max_triplet) max_triplet = val;
        }
        push(&st, j);
    }

    free(max_right);
    free(idx);
    free(st.data);

    return max_triplet;
}
