int minCostII(int** costs, int costsSize, int* costsColSize) {
    if (costsSize == 0 || *costsColSize == 0) return 0;

    int n = costsSize;
    int k = *costsColSize;

    int* prev = (int*)calloc(k, sizeof(int));

    for (int i = 0; i < n; i++) {
        int* curr = (int*)malloc(k * sizeof(int));
        int min1 = INT_MAX, min2 = INT_MAX, ind1 = -1;

        for (int j = 0; j < k; j++) {
            if (prev[j] < min1) {
                min2 = min1;
                min1 = prev[j];
                ind1 = j;
            } else if (prev[j] < min2) {
                min2 = prev[j];
            }
        }

        for (int j = 0; j < k; j++) {
            curr[j] = costs[i][j] + (j == ind1 ? min2 : min1);
        }

        free(prev);
        prev = curr;
    }

    int result = INT_MAX;
    for (int i = 0; i < k; i++) {
        if (prev[i] < result) result = prev[i];
    }

    free(prev);
    return result;
}
