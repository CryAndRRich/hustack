#include <stdlib.h>
#include <string.h>

typedef struct {
    int n;
    int ***matrix;
    int *count;
} Matrix3D;

Matrix3D* matrix3DCreate(int n) {
    Matrix3D* obj = (Matrix3D*)malloc(sizeof(Matrix3D));
    obj -> n = n;
    obj -> count = (int*)calloc(n, sizeof(int));
    obj -> matrix = (int***)malloc(n * sizeof(int**));
    for (int i = 0; i < n; i++) {
        obj -> matrix[i] = (int**)malloc(n * sizeof(int*));
        for (int j = 0; j < n; j++) {
            obj -> matrix[i][j] = (int*)calloc(n, sizeof(int));
        }
    }
    return obj;
}

void matrix3DSetCell(Matrix3D* obj, int x, int y, int z) {
    if (obj -> matrix[x][y][z] == 0) {
        obj -> matrix[x][y][z] = 1;
        obj -> count[x]++;
    }
}

void matrix3DUnsetCell(Matrix3D* obj, int x, int y, int z) {
    if (obj -> matrix[x][y][z] == 1) {
        obj -> matrix[x][y][z] = 0;
        obj -> count[x]--;
    }
}

int matrix3DLargestMatrix(Matrix3D* obj) {
    int maxCount = 0;
    for (int i = 0; i < obj -> n; i++) {
        if (obj -> count[i] > maxCount) {
            maxCount = obj -> count[i];
        }
    }
    for (int i = obj -> n - 1; i >= 0; i--) {
        if (obj -> count[i] == maxCount) return i;
    }
    return -1;
}

void matrix3DFree(Matrix3D* obj) {
    for (int i = 0; i < obj -> n; i++) {
        for (int j = 0; j < obj -> n; j++) {
            free(obj -> matrix[i][j]);
        }
        free(obj -> matrix[i]);
    }
    free(obj -> matrix);
    free(obj -> count);
    free(obj);
}
