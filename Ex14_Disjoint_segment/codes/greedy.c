#include <stdio.h>
#include <stdlib.h>

int cmp(const void *a, const void *b) {
    int *segA = *(int **)a;
    int *segB = *(int **)b;
    if (segA[1] == segB[1])
        return segA[0] - segB[0];
    return segA[1] - segB[1];
}

int main(){
    int n;
    scanf("%d", &n);

    int **segments = (int**)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        segments[i] = (int *)malloc(2 * sizeof(int));
        scanf("%d %d", &segments[i][0], &segments[i][1]);
    }

    qsort(segments, n, sizeof(int*), cmp);

    int count = 0;
    int last_end = -1; 
    for (int i = 0; i < n; i++){
        if (segments[i][0] > last_end){
            count += 1;
            last_end = segments[i][1];
        }
    }
    
    printf("%d", count);
    
    for (int i = 0; i < n; i++){
        free(segments[i]);
    }
    free(segments);
    
    return 0;
}
