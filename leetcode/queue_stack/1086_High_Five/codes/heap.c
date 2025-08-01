/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int id;
    int scores[1000];
    int count;
} Student;

int cmp(const void* a, const void* b) {
    return (*(int*)b - *(int*)a);
}

int cmpById(const void* a, const void* b) {
    int* s1 = *(int**)a;
    int* s2 = *(int**)b;
    return s1[0] - s2[0];
}

int** highFive(int** items, int itemsSize, int* itemsColSize, int* returnSize, int** returnColumnSizes) {
    Student students[1001] = {0};

    for (int i = 0; i < itemsSize; ++i) {
        int id = items[i][0], score = items[i][1];
        students[id].id = id;
        students[id].scores[students[id].count++] = score;
    }

    int** result = (int**)malloc(1001 * sizeof(int*));
    *returnColumnSizes = (int*)malloc(1001 * sizeof(int));
    *returnSize = 0;

    for (int id = 1; id <= 1000; ++id) {
        if (students[id].count >= 5) {
            qsort(students[id].scores, students[id].count, sizeof(int), cmp);
            int sum = 0;
            for (int j = 0; j < 5; ++j)
                sum += students[id].scores[j];
            result[*returnSize] = (int*)malloc(2 * sizeof(int));
            result[*returnSize][0] = id;
            result[*returnSize][1] = sum / 5;
            (*returnColumnSizes)[*returnSize] = 2;
            (*returnSize)++;
        }
    }

    qsort(result, *returnSize, sizeof(int*), cmpById);
    return result;
}
