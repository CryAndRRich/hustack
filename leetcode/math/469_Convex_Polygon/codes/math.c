#include <stdbool.h>

int crossProduct(int A[2], int B[2], int C[2]) {
    return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0]);
}

bool isConvex(int** points, int pointsSize, int* pointsColSize) {
    int direction = 0;

    for (int i = 0; i < pointsSize; ++i) {
        int* A = points[i];
        int* B = points[(i + 1) % pointsSize];
        int* C = points[(i + 2) % pointsSize];
        int cross = crossProduct(A, B, C);

        if (cross != 0) {
            if (direction == 0)
                direction = (cross > 0) ? 1 : -1;
            else if ((cross > 0 && direction < 0) || (cross < 0 && direction > 0))
                return false;
        }
    }
    return true;
}
