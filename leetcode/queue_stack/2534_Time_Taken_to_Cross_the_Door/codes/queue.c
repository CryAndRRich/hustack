#include <stdlib.h>

int* timeTaken(int* arrival, int arrivalSize, int* state, int stateSize, int* returnSize) {
    int n = arrivalSize;
    int* res = (int*)malloc(n * sizeof(int));
    int* enterQ = (int*)malloc(n * sizeof(int));
    int* exitQ = (int*)malloc(n * sizeof(int));
    
    int enFront = 0, enRear = 0;
    int exFront = 0, exRear = 0;

    int time = 0, i = 0, lastUsed = -1;

    for (int j = 0; j < n; ++j) {
        res[j] = -1;
    }

    while (i < n || enFront < enRear || exFront < exRear) {
        while (i < n && arrival[i] <= time) {
            if (state[i] == 0) {
                enterQ[enRear++] = i;
            } else {
                exitQ[exRear++] = i;
            }
            i++;
        }

        if (exFront < exRear && (lastUsed == 1 || lastUsed == -1 || enFront == enRear)) {
            int person = exitQ[exFront++];
            res[person] = time;
            lastUsed = 1;
        } else if (enFront < enRear) {
            int person = enterQ[enFront++];
            res[person] = time;
            lastUsed = 0;
        } else {
            lastUsed = -1; 
        }

        time++;
    }

    free(enterQ);
    free(exitQ);
    *returnSize = n;
    return res;
}
