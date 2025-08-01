int smallestCommonElement(int** mat, int matSize, int* matColSize) {
    int count[10001] = {0};

    for (int i = 0; i < matSize; i++) {
        for (int j = 0; j < matColSize[i]; j++) {
            count[mat[i][j]]++;
        }
    }

    for (int i = 1; i <= 10000; i++) {
        if (count[i] == matSize) {
            return i;
        }
    }

    return -1;
}
