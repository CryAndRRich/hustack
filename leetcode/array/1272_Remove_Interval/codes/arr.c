/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
#include <stdlib.h>

int** removeInterval(int** intervals, int intervalsSize, int* intervalsColSize, int* toBeRemoved, int toBeRemovedSize, int* returnSize, int** returnColumnSizes) {
    
    int** res = (int**)malloc(sizeof(int*) * intervalsSize * 2); 
    *returnColumnSizes = (int*)malloc(sizeof(int) * intervalsSize * 2);
    int count = 0;

    int r1 = toBeRemoved[0], r2 = toBeRemoved[1];

    for (int i = 0; i < intervalsSize; i++) {
        int a = intervals[i][0], b = intervals[i][1];

        if (b <= r1 || a >= r2) {
            res[count] = (int*)malloc(sizeof(int) * 2);
            res[count][0] = a;
            res[count][1] = b;
            (*returnColumnSizes)[count++] = 2;
        } else {
            if (a < r1) {
                res[count] = (int*)malloc(sizeof(int) * 2);
                res[count][0] = a;
                res[count][1] = r1 < b ? r1 : b;
                (*returnColumnSizes)[count++] = 2;
            }
            if (b > r2) {
                res[count] = (int*)malloc(sizeof(int) * 2);
                res[count][0] = r2 > a ? r2 : a;
                res[count][1] = b;
                (*returnColumnSizes)[count++] = 2;
            }
        }
    }

    *returnSize = count;
    return res;
}
